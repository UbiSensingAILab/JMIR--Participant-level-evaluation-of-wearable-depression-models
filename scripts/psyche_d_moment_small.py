#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), CC BY-NC 4.0, a separate institution's protocol,
# independent of this project's own ethics protocol (REB25-1587_MOD1).
# `research_datasets/` is gitignored; the per-row sequence/embedding caches
# written below live there and are never committed.
"""MOMENT-Small on PSYCHE-D, using the only real sequence the release has.

PSYCHE-D has no raw per-minute or per-day wearable signal (see
psyche_d_ts_vs_baseline_data_availability_report.md). What it does have is
one row per participant per MONTH (35,694 rows, 4,948 participants, months
1-12), each holding that month's wearable summaries. For every labeled
quarter-end row (n = 10,866) we build that participant's own month-by-month
history up to and including that month, on 19 core sleep/steps summary
channels, and feed it to the frozen MOMENT-1-small backbone (embedding mode).
The history is at most 12 months. MOMENT reads its 512-step window in patches
of 8 steps and only counts a patch as observed when all 8 steps are, so each
month's value is held for exactly 8 steps (sample-and-hold): one month = one
MOMENT patch. Months sit at the end of the window, a missing month masks its
whole patch, and every earlier position is masked with input_mask, so the
model never sees padding as data. No future months are used.

Caveat, disclosed in the report: these are sequences of monthly summaries,
not raw sensor samples, and MOMENT normalises each channel per sample
(RevIN), so the embedding carries the SHAPE of each person's monthly trend,
not its level. The levels are already in the 150 engineered features.

Three steps, because no single local venv has both momentfm and lightgbm
(and nothing is installed into the live backend venv .venv311):

  sequences  (.venv-psyche-d-fl-mlp)  parquet -> research_datasets/.../moment_small_sequences.npz
  embed      (.venv311, momentfm)     sequences -> moment_small_embeddings.npz
  evaluate   (.venv-psyche-d-fl-mlp)  embeddings + features -> results JSON + src/data JSON

evaluate uses the identical StratifiedGroupKFold(5, shuffle, 42) folds,
the same REPRO.phase2_dart_factory() classifier, balanced sample weights and
mean imputation as the frozen baseline (arm B), and recomputes arm B on the
same folds as a parity check against its published 0.7090. Aggregate metrics
only; no participant IDs leave research_datasets/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
CACHE_DIR = ROOT / "research_datasets" / "old" / "psyche_d" / "derived"
SEQ_PATH = CACHE_DIR / "moment_small_sequences.npz"
EMB_PATH = CACHE_DIR / "moment_small_embeddings.npz"
RESULTS_PATH = SCRIPT_DIR / "psyche_d_moment_small_results.json"
UI_PATH = ROOT / "src" / "data" / "psycheDMomentSmall.json"

SEQ_LEN = 512
N_MONTHS = 12
PATCH = 8
TARGET_SENS = 0.554  # Makhmutova et al. 2022 reported sensitivity
PAPER = {"sensitivity": 0.554, "specificity": 0.653, "auprc": 0.31, "prevalence": 0.207}

# Monthly level/spread/count summaries only. The per-period regression
# score/intercept/coeff columns are left out: they already describe a trend
# inside the month, which is what the sequence is meant to show.
CHANNELS = [
    "steps_awake_mean", "steps_mvpa_iqr", "steps_lpa_iqr", "steps_awake_sum_iqr",
    "steps__active_day_count_", "steps__sedentary_day_count_",
    "steps_mvpa_sum_recent", "steps_lpa_sum_recent", "steps_rolling_6_median_recent",
    "sleep_asleep_weekday_mean", "sleep_asleep_weekend_mean",
    "sleep_in_bed_weekday_mean", "sleep_in_bed_weekend_mean",
    "sleep_ratio_asleep_in_bed_weekday_mean", "sleep_asleep_iqr",
    "sleep_main_start_hour_adj_median", "sleep_main_start_hour_adj_iqr",
    "sleep__hypersomnia_count_", "sleep__hyposomnia_count_",
]


def step_sequences():
    sys.path.insert(0, str(SCRIPT_DIR))
    import audit_psyche_d_baseline_reproduction as REPRO

    raw = REPRO.load_raw()
    assert all(c in raw.columns for c in CHANNELS)
    # Global z-score so channels are on comparable scales before MOMENT's own
    # per-sample RevIN; missing values inside an observed month are filled
    # with the channel mean (0 after z-scoring) and still marked observed.
    vals = raw[CHANNELS].astype(float)
    vals = (vals - vals.mean()) / vals.std(ddof=0)
    vals = vals.fillna(0.0)

    by_user = {}
    for uid, mth, row in zip(raw["user_id"], raw["user_mth"], vals.to_numpy(np.float32)):
        by_user.setdefault(uid, {})[int(mth)] = row

    labeled = raw[raw["ref_label"]]
    n, c = len(labeled), len(CHANNELS)
    x = np.zeros((n, c, SEQ_LEN), np.float32)
    mask = np.zeros((n, SEQ_LEN), np.float32)
    n_obs = np.zeros(n, np.int16)
    for i, (uid, m) in enumerate(zip(labeled["user_id"], labeled["user_mth"])):
        months = by_user[uid]
        for k in range(1, int(m) + 1):  # months 1..m, never later
            start = SEQ_LEN - (int(m) - k + 1) * PATCH
            if k in months:
                x[i, :, start:start + PATCH] = months[k][:, None]
                mask[i, start:start + PATCH] = 1.0
                n_obs[i] += 1
    assert (mask[:, -PATCH:] == 1).all(), "every labeled row's own month must be observed"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SEQ_PATH, x=x, mask=mask, n_obs=n_obs, row_index=labeled.index.to_numpy().astype(str))
    print(f"wrote {SEQ_PATH} x={x.shape} observed months per sample: mean {n_obs.mean():.2f}, "
          f"min {n_obs.min()}, max {n_obs.max()}")


def step_embed(batch_size: int):
    import torch
    from momentfm import MOMENTPipeline

    d = np.load(SEQ_PATH)
    x, mask = d["x"], d["mask"]
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-small", model_kwargs={"task_name": "embedding"})
    model.init()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MOMENT-1-small loaded on {device}, {n_params / 1e6:.1f}M frozen params")

    out = np.zeros((len(x), 512), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[s:s + batch_size]).to(device)
            mb = torch.from_numpy(mask[s:s + batch_size]).to(device)
            out[s:s + batch_size] = model(x_enc=xb, input_mask=mb).embeddings.float().cpu().numpy()
            if (s // batch_size) % 20 == 0:
                print(f"  {s + len(xb)}/{len(x)}  {time.time() - t0:.0f}s", flush=True)
    assert np.isfinite(out).all(), "non-finite embeddings"
    np.savez_compressed(EMB_PATH, emb=out, row_index=d["row_index"], device=str(device), n_params=n_params,
                        seconds=time.time() - t0)
    print(f"wrote {EMB_PATH} {out.shape} in {time.time() - t0:.0f}s")


def matched(y, p):
    grid = np.round(np.arange(0.005, 0.9951, 0.0025), 4)
    t = max(t for t in grid if ((p >= t) & (y == 1)).sum() / (y == 1).sum() >= TARGET_SENS)
    pr = p >= t
    tp = int((pr & (y == 1)).sum()); fp = int((pr & (y == 0)).sum())
    tn = int((~pr & (y == 0)).sum()); fn = int((~pr & (y == 1)).sum())
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    return {"decision_threshold": float(t), "accuracy": (tp + tn) / len(y), "sensitivity": sens,
            "specificity": spec, "balanced_accuracy": (sens + spec) / 2, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def paired_delta_ci(y, groups, a, b, n_boot=1000, seed=0):
    """Participant-level bootstrap of AUROC(a) - AUROC(b)."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[g] for g in rng.choice(uniq, len(uniq))])
        if y[idx].min() == y[idx].max():
            continue
        deltas.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    return [float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))]


def step_evaluate():
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.utils.class_weight import compute_sample_weight

    sys.path.insert(0, str(SCRIPT_DIR))
    import audit_psyche_d_baseline_reproduction as REPRO
    from train_psyche_d_severity_change import bootstrap_ci

    t0 = time.time()
    raw = REPRO.load_raw()
    labeled = raw[raw["ref_label"]]
    e = np.load(EMB_PATH)
    assert (e["row_index"] == labeled.index.to_numpy().astype(str)).all(), "embedding rows out of order"
    seq = np.load(SEQ_PATH)

    base_cols = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
    phq_cols = ["phq9_score_start", "phq9_cat_start"]
    emb = pd.DataFrame(e["emb"], index=labeled.index, columns=[f"moment_{i}" for i in range(512)])
    # Control: the exact same monthly sequences MOMENT saw, one value per
    # month per channel (missing months = NaN), handed straight to LightGBM.
    xs, ms = seq["x"][:, :, ::PATCH][:, :, -N_MONTHS:], seq["mask"][:, ::PATCH][:, -N_MONTHS:]
    raw_seq = pd.DataFrame(np.where(ms[:, None, :] == 1, xs, np.nan).reshape(len(xs), -1), index=labeled.index,
                           columns=[f"seq_{c}_{k}" for c in range(len(CHANNELS)) for k in range(N_MONTHS)])
    X = pd.concat([labeled[base_cols + phq_cols], emb, raw_seq], axis=1)
    y = ((labeled["phq9_cat_end"] - labeled["phq9_cat_start"]) > 0).astype(int).to_numpy()
    groups = labeled["user_id"].to_numpy()

    arms = {
        "moment_only": list(emb.columns),
        "moment_plus_baseline_phq": list(emb.columns) + phq_cols,
        "moment_plus_frozen_baseline_features": base_cols + phq_cols + list(emb.columns),
        "frozen_baseline_arm_b_recomputed": base_cols + phq_cols,
        "baseline_phq_only_recomputed": phq_cols,
        "control_raw_monthly_sequence_no_moment": list(raw_seq.columns),
        "control_wearable_features_only": base_cols,
    }
    oof = {a: np.full(len(y), np.nan) for a in arms}
    skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
    for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, groups=groups)):
        assert not set(groups[tr]) & set(groups[te])
        sw = compute_sample_weight(class_weight="balanced", y=y[tr])
        for a, cols in arms.items():
            imp = SimpleImputer(strategy="mean", keep_empty_features=True)
            clf = REPRO.phase2_dart_factory()
            clf.fit(imp.fit_transform(X.iloc[tr][cols]), y[tr], sample_weight=sw)
            oof[a][te] = clf.predict_proba(imp.transform(X.iloc[te][cols]))[:, 1]
        print(f"fold {f + 1}/{REPRO.N_SPLITS} done {time.time() - t0:.0f}s", flush=True)

    res = {}
    for a, p in oof.items():
        assert not np.isnan(p).any()
        lo, hi, _ = bootstrap_ci(y, p, roc_auc_score)
        alo, ahi, _ = bootstrap_ci(y, p, average_precision_score)
        s = REPRO.summarize(a, p, y)
        res[a] = {"n_features": len(arms[a]), "auroc": s["auroc"], "auroc_ci": [lo, hi], "auprc": s["auprc"],
                  "auprc_ci": [alo, ahi], "at_0_5": {k: s[k] for k in ("accuracy", "sensitivity", "specificity")},
                  "matched_sensitivity": matched(y, p)}
    armb = oof["frozen_baseline_arm_b_recomputed"]
    deltas = {a: {"delta_auroc": res[a]["auroc"] - res["frozen_baseline_arm_b_recomputed"]["auroc"],
                  "ci95_participant_bootstrap": paired_delta_ci(y, groups, oof[a], armb)}
              for a in ("moment_only", "moment_plus_baseline_phq", "moment_plus_frozen_baseline_features")}

    out = {
        "meta": {
            "script": "scripts/psyche_d_moment_small.py",
            "model": "AutonLab/MOMENT-1-small, frozen, embedding mode (512-d), no adapter training",
            "backbone_params_millions": round(float(e["n_params"]) / 1e6, 1),
            "embed_device": str(e["device"]), "embed_seconds": round(float(e["seconds"]), 1),
            "n_samples": int(len(y)), "n_positive": int(y.sum()), "n_participants": int(len(np.unique(groups))),
            "channels": CHANNELS,
            "observed_months_per_sample": {"mean": float(seq["n_obs"].mean()), "min": int(seq["n_obs"].min()),
                                           "max": int(seq["n_obs"].max())},
            "protocol": "StratifiedGroupKFold(5, shuffle, random_state=42), pooled out-of-fold, "
                        "REPRO.phase2_dart_factory(), balanced sample weights, mean imputation; same as arm B",
            "matched_sensitivity_rule": f"highest threshold on a 0.0025 grid with pooled OOF sensitivity >= {TARGET_SENS}",
            "runtime_sec": round(time.time() - t0, 1),
        },
        "paper": {**PAPER, "derived_accuracy": PAPER["sensitivity"] * PAPER["prevalence"]
                  + PAPER["specificity"] * (1 - PAPER["prevalence"])},
        "arms": res,
        "delta_vs_frozen_baseline": deltas,
    }
    RESULTS_PATH.write_text(json.dumps(out, indent=1))
    UI_PATH.write_text(json.dumps(out, indent=1))
    print(json.dumps(deltas, indent=1))
    print("wrote", RESULTS_PATH, UI_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["sequences", "embed", "evaluate"])
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    {"sequences": step_sequences, "embed": lambda: step_embed(a.batch_size), "evaluate": step_evaluate}[a.step]()
