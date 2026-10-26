#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stage 6 branch R evaluation (run in .venv-psyche-d-fl-mlp). Uses the frozen MOMENT
embeddings cached by stage06b. Same repeated nested grouped CV (5x5, inner 3-fold) as the
frozen baseline, paired participant bootstrap against the frozen OOF predictions.
Arms: MOMENT(HR-min) alone, MOMENT(5-min multichannel) alone, MOMENT(HR-min) + baseline
questionnaires, MOMENT(HR-min) + frozen features, raw flattened control (24-hour HR profile),
and a small TCN trained on the same 512-minute HR windows. MiniRocket: not run (sktime is not
installed in any project environment).
Embeddings are reduced to 16 principal components INSIDE each training fold (PCA fit on train).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrv_phq9_protocol_common as C  # noqa: E402

EMB = C.CACHE / "hrv_phq9_protocol_moment_embeddings.npz"
WIN = C.CACHE / "hrv_phq9_protocol_windows.npz"
N_PC = 16
LR_GRID = [{"m__C": c} for c in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]]


def lr_pipe(n_emb: int, n_other: int):
    parts = []
    if n_emb:
        parts.append(("emb", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler()), ("pca", PCA(N_PC, random_state=C.SEED))]),
                      list(range(n_emb))))
    if n_other:
        parts.append(("oth", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), list(range(n_emb, n_emb + n_other))))
    return Pipeline([("ct", ColumnTransformer(parts)),
                     ("m", LogisticRegression(class_weight="balanced", max_iter=5000, solver="liblinear"))])


def hr_profile(seg):
    hm = pd.read_pickle(C.HRM_MINUTE_CACHE)
    t = pd.to_datetime(hm["minute"] * 60000 + C.KST_OFFSET_MS, unit="ms")
    hm["study_day"] = (t.dt.normalize() - C.STUDY_DAY1).dt.days + 1
    hm["hour"] = t.dt.hour
    hm = hm[hm["study_day"].between(1, 28)]
    hm["segment"] = np.where(hm["study_day"] <= 14, "A", "B")
    prof = hm.groupby(["deviceId", "segment", "hour"])["hr_mean"].mean().unstack("hour")
    prof.columns = [f"hr_h{h:02d}" for h in prof.columns]
    return seg[["deviceId", "segment"]].merge(prof.reset_index(), on=["deviceId", "segment"], how="left").drop(columns=["deviceId", "segment"]).to_numpy(float)


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 16
        self.net = nn.Sequential(
            nn.Conv1d(1, ch, 7, padding=3, dilation=1), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(ch, ch, 7, padding=6, dilation=2), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(ch, ch, 7, padding=12, dilation=4), nn.ReLU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(ch, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def norm_windows(X, M):
    # per-window instance normalisation over observed minutes (same idea as MOMENT's RevIN)
    out = np.zeros_like(X)
    for i in range(len(X)):
        m = M[i] > 0
        v = X[i, 0, m]
        mu, sd = v.mean(), v.std() + 1e-6
        out[i, 0] = np.where(m, (X[i, 0] - mu) / sd, 0.0)
    return out


def tcn_cv(seg, Xw, owner, y, g):
    oof = np.full((len(y), C.OUTER_REPEATS), np.nan)
    wy = y[owner]
    for rep in range(C.OUTER_REPEATS):
        for k, (tr, te) in enumerate(C.outer_folds(y, g, rep)):
            torch.manual_seed(C.SEED + 10 * rep + k)
            trw = np.isin(owner, tr)
            tew = np.isin(owner, te)
            model = TCN()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            pw = torch.tensor([(wy[trw] == 0).sum() / max((wy[trw] == 1).sum(), 1)], dtype=torch.float32)
            lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
            Xt = torch.from_numpy(Xw[trw]); yt = torch.from_numpy(wy[trw].astype(np.float32))
            gen = torch.Generator().manual_seed(C.SEED + rep)
            for _ in range(15):
                model.train()
                perm = torch.randperm(len(Xt), generator=gen)
                for s in range(0, len(Xt), 64):
                    ix = perm[s:s + 64]
                    opt.zero_grad(); l = lossf(model(Xt[ix]), yt[ix]); l.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pw_te = torch.sigmoid(model(torch.from_numpy(Xw[tew]))).numpy()
            ow = owner[tew]
            for i in te:
                if (ow == i).any():
                    oof[i, rep] = pw_te[ow == i].mean()
    return oof


def main():
    t = time.time()
    torch.set_num_threads(2)
    seg = C.build_segments()
    E = np.load(EMB, allow_pickle=True)
    assert (E["device"] == seg["deviceId"].to_numpy()).all() and (E["segment"] == seg["segment"].to_numpy()).all()
    y, g = seg["y"].to_numpy(), seg["deviceId"].to_numpy()
    fz = C.load_json("hrv_phq9_protocol_frozen_config.json")
    frozen_oof = np.load(C.CACHE / "hrv_phq9_protocol_frozen_oof.npz")["oof"]
    pf = np.nanmean(frozen_oof, axis=1)
    Xfz, _, _ = C.Xy(seg, fz["features"])
    Xq, _, _ = C.Xy(seg, C.arm_columns(seg, "Q_baseline_questionnaires_demo"))
    hr, m5 = E["hr_mean"], E["m5_mean"]
    arms = {
        "MOMENT_HRmin_alone": (np.hstack([hr]), hr.shape[1], 0),
        "MOMENT_M5_multichannel_alone": (np.hstack([m5]), m5.shape[1], 0),
        "MOMENT_HRmin_plus_baseline_questionnaires": (np.hstack([hr, Xq]), hr.shape[1], Xq.shape[1]),
        "MOMENT_HRmin_plus_frozen_features": (np.hstack([hr, Xfz]), hr.shape[1], Xfz.shape[1]),
        "control_raw_flattened_24h_HR_profile": (hr_profile(seg), 0, 24),
    }
    results, oofs = {}, {}
    for name, (X, ne, no) in arms.items():
        res = C.nested_cv(lr_pipe(ne, no), LR_GRID, X, y, g)
        oofs[name] = res["oof"]
        results[name] = {"n_inputs": X.shape[1], "mean_auroc": res["mean_auroc"], "sd_auroc": res["sd_auroc"],
                         **C.summarize_oof(y, res["oof"], g),
                         "paired_vs_frozen": C.paired_boot_diff(y, np.nanmean(res["oof"], axis=1), pf, g)}
        print(name, round(res["mean_auroc"], 4), results[name]["auroc_ci"], results[name]["paired_vs_frozen"], flush=True)
    W = np.load(WIN)
    Xw = norm_windows(W["hr_X"], W["hr_M"]).astype(np.float32)
    toof = tcn_cv(seg, Xw, W["hr_owner"], y, g)
    pt = np.nanmean(toof, axis=1)
    pt = np.where(np.isnan(pt), np.nanmean(pt), pt)
    results["TCN_HRmin_windows"] = {"n_inputs": "512-minute HR windows", "mean_auroc": float(np.nanmean([C.safe_auc(y[~np.isnan(toof[:, r])], toof[~np.isnan(toof[:, r]), r]) for r in range(C.OUTER_REPEATS)])),
                                    **C.summarize_oof(y, pt[:, None], g), "paired_vs_frozen": C.paired_boot_diff(y, pt, pf, g),
                                    "arch": "3 dilated Conv1d(16) + global avg pool, 15 epochs, Adam 1e-3, pos_weight; segment prob = mean window prob"}
    print("TCN", results["TCN_HRmin_windows"]["mean_auroc"], results["TCN_HRmin_windows"]["auroc_ci"], flush=True)
    results["MiniRocket"] = {"status": "not run", "reason": "sktime / MiniRocket not installed in any project venv"}
    np.savez(C.CACHE / "hrv_phq9_protocol_stage06_oof.npz", **oofs, TCN=toof)
    meta = C.load_json("hrv_phq9_protocol_stage06b_moment_embed_results.json")
    out = {"branch": "R", "embedding_meta": meta, "frozen_reference": {"arm": fz["arm"], "model": fz["model"],
           "auroc_repeat_avg_pred": fz["fixed_params_cv"]["auroc_repeat_avg_pred"]},
           "pca_components": N_PC, "classifier": "LR balanced, C in {1e-3..100}, nested inner 3-fold",
           "ppg": "not used: 4.7 GB / 682M rows; a 5-minute mean of raw green-light PPG intensity carries no validated HR/HRV meaning, and beat-level PPG processing was outside this run's time and memory budget",
           "results": results, "seconds": round(time.time() - t, 1)}
    C.write_json("hrv_phq9_protocol_stage06_results.json", out)


if __name__ == "__main__":
    main()
