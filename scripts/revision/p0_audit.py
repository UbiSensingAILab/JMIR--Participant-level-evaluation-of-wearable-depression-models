#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; participant-level predictions are written there, never to results/.
"""JMIR revision, Phase 0 audit (tasks 0.1, 0.2, 0.4; 0.3 is a code reading).

0.1  rebuild the frozen PSYCHE-D OOF predictions (LightGBM-DART, 152 inputs) and
     confirm the fixed 70/15/15 FL split sizes; check saved FL prediction files
     align row-for-row with the split so participant IDs can be attached.
0.2  recompute the Table 3 CIs both ways (row bootstrap = the ablation script's
     `bootstrap_ci`; participant bootstrap = psyche_d_participant_bootstrap_arms.py)
     and compare to the manuscript values.
0.4  map the 150 non-PHQ inputs to wearable / screener / lifestyle-medication.

Seeds: CV split random_state=42, model random_state=42, bootstrap rng seed=42,
2000 resamples. Outputs: results/revision/P0_*.csv and P0_AUDIT.md.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import load_split_and_frame  # noqa: E402

OUT = ROOT / "results" / "revision"
CACHE = ROOT / "research_datasets" / "revision_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
N_BOOT, BOOT_SEED = 2000, 42
PHQ = ["phq9_score_start", "phq9_cat_start"]
MANUSCRIPT = {"arm_B": (0.709, 0.698, 0.721), "phq_only": (0.696, 0.684, 0.708), "diff": (0.013, 0.005, 0.021)}


def group_of(col: str) -> str:
    if col in PHQ:
        return "baseline_phq9"
    if col in REPRO.SCREENER_COLS:
        return "screener"
    if col.startswith(("life", "med", "nonmed")):
        return "lifestyle_medication"
    if col.startswith(("sleep", "steps")):
        return "wearable"
    raise ValueError(col)


def wearable_sub(col: str) -> str:
    kind = "sleep" if col.startswith("sleep") else "steps"
    reg = col.endswith(("__score", "__intercept", "__coeff", "__score_", "__intercept_", "__coeff_"))
    return f"{kind}_{'regression' if reg else 'statistics'}"


# ---------------- 0.4 feature map ----------------
raw = REPRO.load_raw()
base = [c for c in raw.columns if c not in REPRO.LEAKAGE_COLS and c not in ("user_id", "user_mth", "ref_label")]
assert len(base) == 150
fmap = pd.DataFrame({"feature": base + PHQ})
fmap["group"] = fmap["feature"].map(group_of)
fmap["subgroup"] = [wearable_sub(c) if g == "wearable" else g for c, g in zip(fmap["feature"], fmap["group"])]
fmap["timing_vs_end_phq9"] = fmap["group"].map({
    "baseline_phq9": "precedes (start of 3-month period)", "screener": "precedes (enrolment, static)",
    "lifestyle_medication": "concurrent (end month)", "wearable": "concurrent (8-14 days before end PHQ-9)"})
fmap.to_csv(OUT / "P0_4_psyche_d_feature_groups.csv", index=False)
counts = fmap["group"].value_counts().to_dict()
sub_counts = fmap["subgroup"].value_counts().to_dict()
assert counts == {"wearable": 111, "screener": 29, "lifestyle_medication": 10, "baseline_phq9": 2}, counts

# ---------------- 0.1 rebuild frozen OOF ----------------
lab = raw[raw["ref_label"]]
g = lab["user_id"].to_numpy()
y = ((lab["phq9_cat_end"] - lab["phq9_cat_start"]) > 0).astype(int).to_numpy()
arms = {"arm_B": base + PHQ, "phq_only": PHQ}
oof = {k: np.full(len(y), np.nan) for k in arms}
fold = np.full(len(y), -1)
skf = StratifiedGroupKFold(n_splits=REPRO.N_SPLITS, shuffle=True, random_state=REPRO.RANDOM_STATE)
for f, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y, g)):
    fold[te] = f
    sw = compute_sample_weight("balanced", y[tr])
    for k, cols in arms.items():
        X = lab[cols].to_numpy(dtype=float)
        imp = SimpleImputer(strategy="mean", keep_empty_features=True)
        clf = REPRO.phase2_dart_factory()
        clf.fit(imp.fit_transform(X[tr]), y[tr], sample_weight=sw)
        oof[k][te] = clf.predict_proba(imp.transform(X[te]))[:, 1]
pd.DataFrame({"user_id": g, "row_id": lab.index, "fold": fold, "y": y, "oof_arm_B": oof["arm_B"], "oof_phq_only": oof["phq_only"]}).to_csv(
    CACHE / "P0_1_psyche_d_oof_frozen.csv", index=False)
auroc = {k: float(roc_auc_score(y, v)) for k, v in oof.items()}

# FL split
cache, labeled, feature_cols, yy, train_idx, val_idx, test_idx = load_split_and_frame()
split_rows = {"train": (len(train_idx), labeled.loc[train_idx, "user_id"].nunique()),
              "validation": (len(val_idx), labeled.loc[val_idx, "user_id"].nunique()),
              "test": (len(test_idx), labeled.loc[test_idx, "user_id"].nunique())}
y_test_split = yy.loc[test_idx].to_numpy()
align = {}
for name, f in [("FedAvg (step23)", "psyche_d_fedavg_vs_fedtree_step23_results.json"),
                ("FedXgbBagging (step4)", "psyche_d_fedavg_vs_fedtree_step4_results.json"),
                ("TreeAvg (step5)", "psyche_d_fedavg_vs_fedtree_step5_results.json"),
                ("Centralized LightGBM + MLP (step1)", "psyche_d_fedavg_vs_fedtree_step1_results.json"),
                ("Centralized XGBoost (s4)", "psyche_d_treeavg_audit_extend_s4_results.json")]:
    d = json.loads((ROOT / "scripts" / f).read_text())
    ys = [r["y_test"] for r in d["per_seed_results"]] if "per_seed_results" in d else [d["y_test"]]
    align[name] = all(np.array_equal(np.asarray(v), y_test_split) for v in ys)

# ---------------- 0.2 row vs participant bootstrap ----------------
def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


rng = np.random.default_rng(BOOT_SEED)
row_b = {k: [] for k in arms}
for _ in range(N_BOOT):
    ix = rng.integers(0, len(y), len(y))
    for k in arms:
        row_b[k].append(roc_auc_score(y[ix], oof[k][ix]))
users, inv = np.unique(g, return_inverse=True)
by_user = [np.where(inv == i)[0] for i in range(len(users))]
rng = np.random.default_rng(BOOT_SEED)
par_b = {k: [] for k in arms}
for _ in range(N_BOOT):
    ix = np.concatenate([by_user[i] for i in rng.integers(0, len(users), len(users))])
    for k in arms:
        par_b[k].append(roc_auc_score(y[ix], oof[k][ix]))
# the original ablation script bootstrapped each arm with its own fresh rng(42), so its
# row-level CIs are reproduced by restarting the rng per arm
row_sep = {}
for k in arms:
    r = np.random.default_rng(BOOT_SEED)
    vals = []
    for _ in range(N_BOOT):
        ix = r.integers(0, len(y), len(y))
        vals.append(roc_auc_score(y[ix], oof[k][ix]))
    row_sep[k] = ci(vals)

rows = []
for k in arms:
    m = MANUSCRIPT[k]
    rows.append({"quantity": f"AUROC {k}", "point": round(auroc[k], 4), "manuscript": f"{m[0]} ({m[1]}-{m[2]})",
                 "row_level_ci_ablation_script": "%.4f-%.4f" % row_sep[k], "participant_level_ci": "%.4f-%.4f" % ci(par_b[k])})
d_row = np.array(row_b["arm_B"]) - np.array(row_b["phq_only"])
d_par = np.array(par_b["arm_B"]) - np.array(par_b["phq_only"])
m = MANUSCRIPT["diff"]
rows.append({"quantity": "paired diff arm_B - phq_only", "point": round(auroc["arm_B"] - auroc["phq_only"], 4),
             "manuscript": f"{m[0]} ({m[1]}-{m[2]})", "row_level_ci_ablation_script": "%.4f-%.4f (paired row resample)" % ci(d_row),
             "participant_level_ci": "%.4f-%.4f" % ci(d_par)})
ci_df = pd.DataFrame(rows)
ci_df.to_csv(OUT / "P0_2_psyche_d_ci_level.csv", index=False)

inv_df = pd.DataFrame([
    {"item": "Frozen PSYCHE-D OOF (arm B, 152 inputs)", "status": f"rebuilt, AUROC {auroc['arm_B']:.4f}", "file": "research_datasets/revision_cache/P0_1_psyche_d_oof_frozen.csv (gitignored)"},
    {"item": "PHQ-9-only OOF (2 inputs)", "status": f"rebuilt, AUROC {auroc['phq_only']:.4f}", "file": "same"},
    *[{"item": f"FL split {s}", "status": f"{r} rows, {p} participants", "file": "scripts/psyche_d_fedavg_vs_fedtree_split_cache.pkl"} for s, (r, p) in split_rows.items()],
    *[{"item": f"Saved test predictions: {n}", "status": "row order matches split, participant IDs attachable" if ok else "MISMATCH", "file": ""} for n, ok in align.items()],
])
inv_df.to_csv(OUT / "P0_1_inventory.csv", index=False)

res = {"seeds": {"cv_random_state": REPRO.RANDOM_STATE, "model_random_state": REPRO.RANDOM_STATE, "bootstrap_seed": BOOT_SEED, "n_boot": N_BOOT,
                 "fl_split_seed": C.SPLIT_SEED, "fl_partition_seeds": C.PARTITION_SEEDS},
       "n_rows": int(len(y)), "n_participants": int(len(users)), "auroc": auroc, "split": split_rows, "fl_prediction_alignment": align,
       "feature_groups": counts, "wearable_subgroups": {k: v for k, v in sub_counts.items() if k.startswith(("sleep", "steps"))},
       "ci_table": rows}
(OUT / "P0_audit_results.json").write_text(json.dumps(res, indent=2, default=str))
print(json.dumps(res, indent=1, default=str))
