#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): IRAS 321104, independent of REB25-1587_MOD1. Research data.
"""Protocol Stage 9 (robustness), SMALL-N branch. Run with .venv-psyche-d-fl-mlp.

1. 3 CV seeds (42 from Stage 3, plus 43 and 44 here) for every screened model on the
   frozen feature set; mean and population SD of the headline AUROC.
2. Same-data confirmation: every headline row uses all 73 participants / 10 positives.
3. Fold-to-fold spread of the frozen model (the SMALL-N replacement for a confirmatory
   grouped CV), plus leave-one-participant-out (nested inner 3-fold) as a second CV.
4. Pooled-OOF numbers only (there is no fixed split anywhere in this run).
5. Leave-one-site-out: not applicable (single site).
Supplementary label checks with the frozen pipeline: GDS-15 >= 5 (secondary target) and
PHQ-9 >= 5 (mild or worse).

Writes scripts/resilient_protocol_stage09_results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resilient_protocol_common as C  # noqa: E402

OUT = C.SCRIPT_DIR / "resilient_protocol_stage09_results.json"


def lopo(model, X, y, seed=C.SEED):
    s = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.setdiff1d(np.arange(len(y)), [i])
        params, _, _ = C.nested_select(model, X[tr], y[tr], seed + i)
        s[i] = C._fit_predict(model, params, X[tr], y[tr], X[[i]], seed + i)[0][0]
    return s


def main():
    t0 = time.time()
    d = C.load_features()
    F = d["features"]
    uids = F.index.to_numpy()
    y = d["y_phq10"]
    frozen = json.loads((C.SCRIPT_DIR / "resilient_protocol_frozen_config.json").read_text())
    s23 = json.loads((C.SCRIPT_DIR / "resilient_protocol_stage02_03_results.json").read_text())
    X = F[frozen["features"]].to_numpy(dtype=float)
    model = frozen["model"]
    res = {"seeds": [42, 43, 44], "models": {}}
    for name in C.MODEL_NAMES:
        aucs = [s23["screen"][name]["auroc"]]
        for sd in (43, 44):
            r = C.repeated_nested_cv(name, X, y, uids, seed=sd)
            aucs.append(float(roc_auc_score(y, r["oof"].mean(axis=0))))
        res["models"][name] = {"aurocs": aucs, "mean": float(np.mean(aucs)), "sd": float(np.std(aucs))}
        print(f"[seeds] {name}: {np.round(aucs, 3)} ({time.time() - t0:.0f}s)", flush=True)
    folds = [f["test_auc"] for f in s23["screen"][model]["folds"] if f["test_auc"] is not None]
    res["frozen_fold_spread"] = {"n_folds": len(folds), "mean": float(np.mean(folds)), "sd": float(np.std(folds)),
                                 "min": float(np.min(folds)), "max": float(np.max(folds)), "positives_per_test_fold": sorted({f["pos_test"] for f in s23["screen"][model]["folds"]})}
    s = lopo(model, X, y)
    lo, hi, _, _ = C.boot_ci(y, s)
    res["frozen_lopo"] = {"auroc": float(roc_auc_score(y, s)), "auroc_ci": [lo, hi]}
    print("[lopo]", res["frozen_lopo"], flush=True)
    res["same_data_check"] = {"n": int(len(y)), "n_pos": int(y.sum()), "note": "every headline number in Stages 3-9 uses these same 73 participants"}
    res["leave_one_site_out"] = "not applicable: single site"
    # A12 follow-up: enrolment date alone separates the label (AUROC in Stage 2). Does the
    # frozen score just track enrolment date / season?
    from scipy.stats import spearmanr
    s45 = json.loads((C.SCRIPT_DIR / "resilient_protocol_stage04_05_results.json").read_text())
    score = np.array(s45["frozen_oof"]).mean(axis=0)
    date = F["phq_date_ord"].to_numpy()
    month = np.array([__import__("datetime").date.fromordinal(int(v)).month for v in date])
    winter = np.isin(month, [11, 12, 1, 2]).astype(int)
    rho, p = spearmanr(score, date)
    # within-season AUROC of the frozen score (stratified check)
    within = {}
    for name, m in [("Nov-Feb", winter == 1), ("Mar-Oct", winter == 0)]:
        within[name] = {"n": int(m.sum()), "n_pos": int(y[m].sum()),
                        "auroc": float(roc_auc_score(y[m], score[m])) if len(np.unique(y[m])) == 2 else None}
    res["enrolment_date_check"] = {"date_only_auroc": float(roc_auc_score(y, date)), "spearman_score_vs_date": float(rho), "p": float(p),
                                   "winter_enrolment_pos_rate": float(y[winter == 1].mean()), "other_pos_rate": float(y[winter == 0].mean()),
                                   "within_season_auroc": within}
    print("[date]", res["enrolment_date_check"], flush=True)
    res["supplementary_labels"] = {}
    for lab, yy in [("GDS-15 >= 5", d["y_gds5"]), ("PHQ-9 >= 5", d["y_phq5"])]:
        r = C.repeated_nested_cv(model, X, yy, uids, seed=C.SEED)
        sm = C.summarize(yy, r["oof"])
        sm["n_pos"] = int(yy.sum())
        sm.pop("auprc_per_repeat_mean", None)
        res["supplementary_labels"][lab] = sm
        print(f"[label] {lab}: {sm['auroc']:.3f} {sm['auroc_ci']}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    C.write_json(OUT, res)


if __name__ == "__main__":
    main()
