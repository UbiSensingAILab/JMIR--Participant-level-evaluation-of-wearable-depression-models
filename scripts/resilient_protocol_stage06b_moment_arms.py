#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): IRAS 321104, independent of REB25-1587_MOD1. Research data.
"""Protocol Stage 6 (branch R), part b: MOMENT arms under the frozen model family and the
same repeated nested CV, paired bootstrap vs the frozen baseline. Run with
.venv-psyche-d-fl-mlp, after stage06a (embeddings) and stage04_05 (frozen config).

Arms: MOMENT HR only (512-d); MOMENT all 4 modalities (2048-d); MOMENT 4 + demographics;
MOMENT 4 + frozen features; raw-sequence control = flattened hourly HR (512) + hourly
Steps (512) arrays, own-mean gap filled, no MOMENT. MiniRocket / TCN sequence baselines:
not run (sktime is not installed in either venv, and a TCN has no realistic chance to
train on 58 sequences per fold with 8 positives; reported as "not run").

Writes scripts/resilient_protocol_stage06_results.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resilient_protocol_common as C  # noqa: E402

OUT = C.SCRIPT_DIR / "resilient_protocol_stage06_results.json"


def main():
    d = C.load_features()
    F = d["features"]
    uids = F.index.to_numpy()
    y = d["y_phq10"]
    frozen = json.loads((C.SCRIPT_DIR / "resilient_protocol_frozen_config.json").read_text())
    s45 = json.loads((C.SCRIPT_DIR / "resilient_protocol_stage04_05_results.json").read_text())
    model = frozen["model"]
    E = pickle.loads(C.MOMENT_PKL.read_bytes())
    assert list(E["uids"]) == list(uids), "embedding order must match feature table"
    m4 = np.concatenate([E["hr"], E["steps"], E["sleep_physio"], E["sleep_state"]], axis=1)
    raw = np.stack([np.concatenate([
        d["hourly_hr"][u] if d["hourly_hr"][u] is not None else np.full(512, np.nan),
        d["hourly_steps"][u] if d["hourly_steps"][u] is not None else np.full(512, np.nan)]) for u in uids])
    Xf = F[frozen["features"]].to_numpy(dtype=float)
    Xd = F[d["demo_cols"]].to_numpy(dtype=float)
    arms = {
        "MOMENT HR only": E["hr"],
        "MOMENT 4 modalities": m4,
        "MOMENT 4 + demographics": np.concatenate([m4, Xd], axis=1),
        "MOMENT 4 + frozen features": np.concatenate([m4, Xf], axis=1),
        "Raw flattened HR+Steps (control)": raw,
    }
    base = np.array(s45["frozen_oof"]).mean(axis=0)
    res = {"model": model, "branch": "R", "frozen_auroc": frozen["auroc"], "arms": {},
           "missing_embeddings": {k: int(np.isnan(E[k]).any(axis=1).sum()) for k in ["hr", "steps", "sleep_physio", "sleep_state"]},
           "sequence_baselines": "not run: MiniRocket needs sktime (not installed in .venv311 or .venv-psyche-d-fl-mlp); a TCN on 58 training sequences with 8 positives per fold was judged not trainable",
           "pooling": "one 512-hour window per participant per modality, so per-participant pooling is the identity"}
    for name, X in arms.items():
        t0 = time.time()
        r = C.repeated_nested_cv(model, X.astype(float), y, uids, seed=C.SEED)
        s = C.summarize(y, r["oof"])
        s["n_features"] = int(X.shape[1])
        s["paired_vs_frozen"] = C.paired_boot_diff(y, r["oof"].mean(axis=0), base)
        s["seconds"] = round(time.time() - t0, 1)
        s["oof_avg"] = r["oof"].mean(axis=0).tolist()
        res["arms"][name] = s
        print(f"[moment] {name}: {s['auroc']:.3f} [{s['auroc_ci'][0]:.3f},{s['auroc_ci'][1]:.3f}] d={s['paired_vs_frozen']['delta']:+.3f} {s['paired_vs_frozen']['ci']} ({s['seconds']}s)", flush=True)
    C.write_json(OUT, res)


if __name__ == "__main__":
    main()
