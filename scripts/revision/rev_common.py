"""Shared helpers for the JMIR analysis revision (scripts/revision/p1_*.py).

Participant-level bootstrap: resample participants with replacement, keep all of
each participant's rows together; the SAME draws are reused for every model so
differences are paired. 2000 resamples, numpy default_rng(42)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision"
CACHE = ROOT / "research_datasets" / "revision_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
N_BOOT = 2000
BOOT_SEED = 42


def participant_draws(groups: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> list[np.ndarray]:
    users, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.r_[0, np.cumsum(np.bincount(inv))]
    rows_of = [order[starts[i]:starts[i + 1]] for i in range(len(users))]
    rng = np.random.default_rng(seed)
    return [np.concatenate([rows_of[i] for i in rng.integers(0, len(users), len(users))]) for _ in range(n_boot)]


def ci(v) -> list[float]:
    v = np.asarray(v, dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def _safe(fn, y, p):
    return fn(y, p) if 0 < y.sum() < len(y) else np.nan


def boot_metric(y, p, draws, fn=roc_auc_score, mask=None) -> np.ndarray:
    out = np.empty(len(draws))
    for b, ix in enumerate(draws):
        if mask is not None:
            ix = ix[mask[ix]]
        out[b] = _safe(fn, y[ix], p[ix])
    return out


def summarize(y, p, draws, mask=None) -> dict:
    yy, pp = (y, p) if mask is None else (y[mask], p[mask])
    ba = boot_metric(y, p, draws, roc_auc_score, mask)
    bp = boot_metric(y, p, draws, average_precision_score, mask)
    return {"n": int(len(yy)), "n_pos": int(yy.sum()), "auroc": float(roc_auc_score(yy, pp)), "auroc_ci": ci(ba),
            "auprc": float(average_precision_score(yy, pp)), "auprc_ci": ci(bp), "_boot_auroc": ba}


def paired(a: dict, b: dict) -> dict:
    d = a["_boot_auroc"] - b["_boot_auroc"]
    lo, hi = ci(d)
    return {"delta": a["auroc"] - b["auroc"], "ci": [lo, hi], "ci_excludes_0": bool(lo > 0 or hi < 0)}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def fmt(m: dict) -> str:
    return f"{m['auroc']:.3f} ({m['auroc_ci'][0]:.3f}-{m['auroc_ci'][1]:.3f})"
