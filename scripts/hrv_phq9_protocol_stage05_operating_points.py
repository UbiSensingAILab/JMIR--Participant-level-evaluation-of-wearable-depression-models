#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stage 5: operating points for the frozen configuration (SMALL-N branch).
For every outer fold (5 folds x 5 repeats) the threshold is chosen on the inner 3-fold OOF
predictions of that fold's TRAINING participants only, then applied once to the outer test fold.
Confusion counts are pooled over the 5 outer folds of a repeat; metrics are averaged over repeats.
Rules: default = max balanced accuracy on linspace(0.005, 0.995, 199); max-accuracy = max accuracy
on the same grid; matched sensitivity = highest threshold (0.0025 grid) whose inner sensitivity
>= 0.80 (clinical target, no published reference exists). Full 0.01 sweep on the repeat-averaged
OOF predictions is descriptive only.
Output: hrv_phq9_protocol_stage05_results.json, src/... is NOT touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrv_phq9_protocol_common as C  # noqa: E402

GRID = np.linspace(0.005, 0.995, 199)
FINE = np.arange(0.0, 1.0 + 1e-9, 0.0025)


def conf(y, p, t):
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fn, tn, fp


def metrics(tp, fn, tn, fp):
    n = tp + fn + tn + fp
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    return {"accuracy": (tp + tn) / n, "sensitivity": sens, "specificity": spec,
            "balanced_accuracy": np.nanmean([sens, spec]),
            "ppv": tp / (tp + fp) if tp + fp else np.nan, "npv": tn / (tn + fn) if tn + fn else np.nan,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp}


def pick(y, p, rule):
    if rule == "default":
        vals = [metrics(*conf(y, p, t))["balanced_accuracy"] for t in GRID]
        return float(GRID[int(np.nanargmax(vals))])
    if rule == "max_accuracy":
        vals = [metrics(*conf(y, p, t))["accuracy"] for t in GRID]
        return float(GRID[int(np.nanargmax(vals))])
    if rule == "matched_sensitivity":
        ok = [t for t in FINE if metrics(*conf(y, p, t))["sensitivity"] >= C.CLINICAL_TARGET_SENS]
        return float(max(ok)) if ok else 0.0
    raise ValueError(rule)


def main():
    seg = C.build_segments()
    fz = C.load_json("hrv_phq9_protocol_frozen_config.json")
    X, y, g = C.Xy(seg, fz["features"])
    base = C.model_specs()[fz["model"]][0]
    res = C.nested_cv(base, [fz["hyperparameters"]], X, y, g, fixed_params=fz["hyperparameters"], return_inner_oof=True)
    rules = ["default", "max_accuracy", "matched_sensitivity"]
    per_rep = {r: [] for r in rules}
    thr = {r: [] for r in rules}
    for rep in range(C.OUTER_REPEATS):
        counts = {r: np.zeros(4, dtype=int) for r in rules}
        for f in [i for i in res["inner"] if i["repeat"] == rep]:
            tr, te, ip = f["train_idx"], f["test_idx"], f["inner_oof"]
            ytr = y[tr]
            ok = ~np.isnan(ip)
            for r in rules:
                t = pick(ytr[ok], ip[ok], r)
                thr[r].append(t)
                counts[r] += np.array(conf(y[te], res["oof"][te, rep], t))
        for r in rules:
            per_rep[r].append(metrics(*counts[r]))
    summary = {}
    for r in rules:
        keys = ["accuracy", "sensitivity", "specificity", "balanced_accuracy", "ppv", "npv"]
        summary[r] = {k: float(np.nanmean([m[k] for m in per_rep[r]])) for k in keys}
        summary[r].update({k + "_sd_over_repeats": float(np.nanstd([m[k] for m in per_rep[r]])) for k in ["accuracy", "sensitivity", "specificity"]})
        summary[r]["threshold_median"] = float(np.median(thr[r]))
        summary[r]["threshold_range"] = [float(np.min(thr[r])), float(np.max(thr[r]))]
        summary[r]["repeat0_counts"] = {k: per_rep[r][0][k] for k in ["tp", "fn", "tn", "fp"]}
    summary["matched_sensitivity"]["reference_sensitivity"] = C.CLINICAL_TARGET_SENS
    summary["matched_sensitivity"]["reference_source"] = "clinical target sensitivity 0.80 (no published model on this dataset and target)"
    pavg = np.nanmean(res["oof"], axis=1)
    sweep = [{"threshold": round(float(t), 2), **metrics(*conf(y, pavg, t))} for t in np.arange(0.01, 1.0, 0.01)]
    maj = float(1 - y.mean())
    reach = [s for s in sweep if s["accuracy"] > maj]
    out = {"frozen": {"arm": fz["arm"], "model": fz["model"]}, "majority_rate": maj, "pos_rate": float(y.mean()),
           "selection": "thresholds chosen on inner OOF of each outer training fold, applied once to the outer test fold",
           "operating_points": summary, "per_repeat": per_rep,
           "sweep_repeat_avg_oof": sweep,
           "accuracy_above_majority_reachable_in_sweep": bool(reach),
           "best_sweep_accuracy": max(s["accuracy"] for s in sweep),
           "best_sweep_accuracy_at": max(sweep, key=lambda s: s["accuracy"]),
           "note": "sweep is descriptive (on pooled OOF predictions, not validation-selected)"}
    C.write_json("hrv_phq9_protocol_stage05_results.json", out)
    for r in rules:
        print(r, {k: round(v, 3) if isinstance(v, float) else v for k, v in summary[r].items()})
    print("majority", maj, "best sweep acc", out["best_sweep_accuracy"])


if __name__ == "__main__":
    main()
