#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stage 9: robustness (SMALL-N branch).
 1. 3 seeds for every model on the frozen arm (seed s changes the model random_state AND the
    outer-fold shuffle), nested grouped CV; mean and population SD of the pooled AUROC.
 2. Fold-to-fold spread of the frozen model (5 folds x 5 repeats).
 3. Leave-one-participant-out (LOPO) for the frozen configuration (fixed hyperparameters).
 4. Secondary target PHQ-9 >= 10 with the frozen configuration (sensitivity analysis).
 5. Same-data check: every headline number uses the same 98 segments / 36 positives.
Output: hrv_phq9_protocol_stage09_results.json
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrv_phq9_protocol_common as C  # noqa: E402


def seed_run(seg, cols, model, s):
    X, y, g = C.Xy(seg, cols)
    base, grid = C.model_specs(seed=s)[model]
    oof = np.full(len(y), np.nan)
    cv = StratifiedGroupKFold(C.OUTER_SPLITS, shuffle=True, random_state=1000 + s)
    for k, (tr, te) in enumerate(cv.split(X, y, g)):
        params, _ = C.inner_select(base, grid, X[tr], y[tr], g[tr], seed=2000 + 10 * s + k)
        e = clone(base).set_params(**params)
        C._fit(e, X[tr], y[tr])
        oof[te] = C._proba(e, X[te])
    return model, s, C.safe_auc(y, oof)


def main():
    t = time.time()
    seg = C.build_segments()
    fz = C.load_json("hrv_phq9_protocol_frozen_config.json")
    cols = fz["features"]
    X, y, g = C.Xy(seg, cols)
    models = list(C.model_specs().keys())
    rr = Parallel(n_jobs=3, verbose=3)(delayed(seed_run)(seg, cols, m, s) for m in models for s in range(3))
    seeds = {}
    for m in models:
        v = [a for (mm, s, a) in rr if mm == m]
        seeds[m] = {"aurocs": v, "mean": float(np.mean(v)), "sd_population": float(np.std(v))}

    base = C.model_specs()[fz["model"]][0]
    params = fz["hyperparameters"]
    fold_aucs = []
    for rep in range(C.OUTER_REPEATS):
        for tr, te in C.outer_folds(y, g, rep):
            e = clone(base).set_params(**params)
            C._fit(e, X[tr], y[tr])
            fold_aucs.append(C.safe_auc(y[te], C._proba(e, X[te])))
    fa = np.array(fold_aucs, float)

    lopo = np.full(len(y), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        e = clone(base).set_params(**params)
        C._fit(e, X[tr], y[tr])
        lopo[te] = C._proba(e, X[te])
    lopo_ci, _ = C.boot_ci(y, lopo, g)

    y10 = seg["y10"].to_numpy()
    oof10 = np.full((len(y), C.OUTER_REPEATS), np.nan)
    for rep in range(C.OUTER_REPEATS):
        for tr, te in C.outer_folds(y10, g, rep):
            e = clone(base).set_params(**params)
            C._fit(e, X[tr], y10[tr])
            oof10[te, rep] = C._proba(e, X[te])
    s10 = C.summarize_oof(y10, oof10, g)
    s10["per_repeat_auroc"] = [C.safe_auc(y10, oof10[:, r]) for r in range(C.OUTER_REPEATS)]

    out = {"frozen": {"arm": fz["arm"], "model": fz["model"]},
           "seeds_all_models_frozen_arm": seeds,
           "frozen_fold_spread": {"n_folds": int(len(fa)), "n_defined": int(np.sum(~np.isnan(fa))), "mean": float(np.nanmean(fa)),
                                  "sd": float(np.nanstd(fa)), "min": float(np.nanmin(fa)), "max": float(np.nanmax(fa))},
           "lopo": {"auroc": C.safe_auc(y, lopo), "auroc_ci": lopo_ci,
                    "note": "pooled over 49 single-participant test folds; LOPO pooled AUROC can be biased low for small N"},
           "secondary_target_phq9_ge10": {"n_pos": int(y10.sum()), "pos_participants": int(seg.groupby('deviceId')['y10'].max().sum()), **s10},
           "same_data_check": {"n_rows": int(len(y)), "n_pos": int(y.sum()), "n_participants": int(len(set(g)))},
           "seconds": round(time.time() - t, 1)}
    C.write_json("hrv_phq9_protocol_stage09_results.json", out)
    print(seeds, out["frozen_fold_spread"], out["lopo"], {k: s10[k] for k in ("auroc_repeat_avg_pred", "auroc_ci")})


if __name__ == "__main__":
    main()
