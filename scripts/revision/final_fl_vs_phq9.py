#!/usr/bin/env python3
"""Part 2 item 1d: baseline PHQ-9-only model trained on the same 70% training portion as the
federated models, scored on the fixed test set, and paired with TreeAvg (P+S+L 41, corrected
per-round-seed implementation, selected eta) pooled over the 3 partition seeds.

Same recipe as final_fl.centralized (frozen LightGBM-DART factory, mean imputation, balanced
sample weights) and the same participant-level bootstrap (2000 resamples, rng 42) and pooling
rule as final_fl.pooled. Reuses the saved TreeAvg test predictions
(research_datasets/revision_cache/T2_T5_fl_test_predictions_seedvary.npz)."""
import json

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

import p2_fl_core as F
from p2_fl_core import C, ROOT, Boot, REPRO

CACHE = ROOT / "research_datasets" / "revision_cache"
FIN = ROOT / "results" / "revision" / "final"
SEEDS = C.PARTITION_SEEDS
P = ["phq9_score_start", "phq9_cat_start"]


def main():
    D = F.Data()
    ytr = D.yc(D.train_df)
    Xtr = D.train_df[P].to_numpy(dtype=np.float64)
    Xte = D.test_df[P].to_numpy(dtype=np.float64)
    imp = SimpleImputer(strategy="mean", keep_empty_features=True)
    clf = REPRO.phase2_dart_factory()
    clf.fit(imp.fit_transform(Xtr), ytr, sample_weight=compute_sample_weight("balanced", ytr))
    p_phq = clf.predict_proba(imp.transform(Xte))[:, 1]

    z = np.load(CACHE / "T2_T5_fl_test_predictions_seedvary.npz", allow_pickle=True)  # own cache; users stored as objects
    assert np.array_equal(z["y"], D.y_test) and np.array_equal(z["users"], D.test_users)
    B = Boot(D.test_users, D.y_test)
    met_phq, d_phq = B.metric(p_phq)
    tre = {s: z[f"P+S+L 41|IID|TreeAvg|{s}"] for s in SEEDS}
    ds, per = [], []
    for s in SEEDS:
        met, d = B.metric(tre[s])
        ds.append(d - d_phq)
        per.append({"partition_seed": s, "treeavg_auroc": met["auroc"], "treeavg_auroc_ci": met["auroc_ci"],
                    "delta": met["auroc"] - met_phq["auroc"], "ci": Boot.ci(d - d_phq)})
    pooled_ci = Boot.ci(np.mean(ds, axis=0))
    pooled_delta = float(np.mean([r["delta"] for r in per]))
    res = {
        "description": "Baseline PHQ-9 only (phq9_score_start, phq9_cat_start), frozen LightGBM-DART recipe, trained on the 70% "
                       "training portion (same as federated models), scored on the test set; paired with TreeAvg P+S+L 41 "
                       "(per-round seeds, selected eta) pooled over partition seeds 1-3.",
        "n_train_rows": int(len(ytr)), "n_train_participants": int(len(D.cache["train_participants"])),
        "n_test_rows": int(len(D.y_test)), "n_test_participants": int(B.n_participants),
        "baseline_phq9_70pct": met_phq,
        "per_seed": per,
        "pooled": {"comparison": "TreeAvg (P+S+L 41) - baseline PHQ-9 (70%-trained)", "delta": pooled_delta, "ci": pooled_ci,
                   "ci_excludes_0": bool(pooled_ci[0] > 0 or pooled_ci[1] < 0)},
        "bootstrap": {"n": F.N_BOOT, "seed": F.BOOT_SEED, "level": "participant"},
        "source_predictions": "research_datasets/revision_cache/T2_T5_fl_test_predictions_seedvary.npz",
        "script": "scripts/revision/final_fl_vs_phq9.py",
    }
    (FIN / "FL_vs_PHQ9_70pct.json").write_text(json.dumps(res, indent=1))
    f = lambda v: f"{v:.3f}"  # noqa: E731
    md = ["# FL vs baseline PHQ-9 (70%-trained), test set", "",
          f"- Baseline PHQ-9 (70%-trained, frozen DART recipe): AUROC {f(met_phq['auroc'])} ({f(met_phq['auroc_ci'][0])}-{f(met_phq['auroc_ci'][1])}); "
          f"AUPRC {f(met_phq['auprc'])} ({f(met_phq['auprc_ci'][0])}-{f(met_phq['auprc_ci'][1])})"]
    for r in per:
        md.append(f"- Seed {r['partition_seed']}: TreeAvg {f(r['treeavg_auroc'])}; TreeAvg − PHQ-9 {r['delta']:+.3f} ({r['ci'][0]:+.3f} to {r['ci'][1]:+.3f})")
    md.append(f"- Pooled over 3 seeds: TreeAvg − baseline PHQ-9 {pooled_delta:+.3f} ({pooled_ci[0]:+.3f} to {pooled_ci[1]:+.3f})")
    (FIN / "FL_vs_PHQ9_70pct.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
