#!/usr/bin/env python3
# Research data (PSYCHE-D, HRV and PHQ-9, RESILIENT) stay in research_datasets/ (gitignored).
"""Reproduce every JMIR revision result (Phase 0, Phases 1-3, Tier 1, final run).

Usage (from the repo root):
    python3 scripts/revision/run_all_revision.py            # all analyses, cached MOMENT embeddings
    python3 scripts/revision/run_all_revision.py --moment   # also regenerate MOMENT embeddings (.venv311)
    python3 scripts/revision/run_all_revision.py --from final_fl.py   # resume at a step

Each step runs in its own process because torch and LightGBM/XGBoost crash or deadlock in
one process on this Mac (OpenMP clash); OMP_NUM_THREADS=1 is set for every step.
Later steps read outputs of earlier ones, so keep the order. Logs go to results/revision/logs/.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TREES = str(ROOT / ".venv-psyche-d-fl-mlp" / "bin" / "python")
MOMENT = str(ROOT / ".venv311" / "bin" / "python")
R = "scripts/revision/"

# (python, script, args, extra env, approximate runtime on the 2026-10-16 machine)
MOMENT_STEPS = [
    (TREES, "scripts/psyche_d_moment_small.py", ["sequences"], {}, "~1 min"),
    (MOMENT, "scripts/psyche_d_moment_small.py", ["embed"], {}, "~5-10 min"),
    (MOMENT, "scripts/hrv_phq9_protocol_stage06b_moment_embed.py", [], {}, "~5 min"),
    (MOMENT, "scripts/resilient_protocol_stage06a_moment_embed.py", [], {}, "~5 min"),
]
STEPS = [
    (TREES, R + "p0_audit.py", [], {}, "~2 min"),
    (TREES, R + "p1_cv.py", [], {}, "~5 min"),
    (TREES, R + "p1_temporal.py", [], {}, "<1 min"),
    (TREES, R + "p1_holdout.py", [], {}, "~2 min"),
    (TREES, R + "p2_fl_rerun.py", [], {}, "~1 min"),
    (TREES, R + "p2_fl_noniid.py", [], {}, "~1 min"),
    (TREES, R + "p2_mb_selected.py", [], {}, "<1 min"),
    (TREES, R + "p3_small_datasets.py", [], {}, "~3.5 min"),
    (TREES, R + "r_tier1_cv.py", [], {}, "~11 min"),
    (TREES, R + "r_tier1_split.py", [], {}, "~3 min"),
    (TREES, R + "r_tier1_fl.py", [], {}, "<1 min"),
    (TREES, R + "final_fl.py", [], {}, "~2 min"),
    (TREES, R + "final_fl.py", [], {"FL_VARY_SEED": "1"}, "~2 min"),
    (TREES, R + "final_outcome_calib.py", [], {}, "~1 min"),
    (TREES, R + "final_package.py", [], {}, "<1 min"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moment", action="store_true", help="regenerate MOMENT embeddings first (.venv311)")
    ap.add_argument("--from", dest="start", default=None, help="start at the first step whose script name contains this")
    a = ap.parse_args()
    steps = (MOMENT_STEPS if a.moment else []) + STEPS
    if a.start:
        steps = steps[next(i for i, s in enumerate(steps) if a.start in s[1]):]
    logs = ROOT / "results" / "revision" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for py, script, args, env, eta in steps:
        tag = Path(script).stem + ("_" + "_".join(args) if args else "") + ("_seedvary" if env.get("FL_VARY_SEED") else "")
        print(f"[{time.time() - t0:6.0f}s] {tag} (expected {eta})", flush=True)
        with open(logs / f"{tag}.log", "w") as fh:
            r = subprocess.run([py, str(ROOT / script), *args], cwd=ROOT / Path(script).parent, stdout=fh, stderr=subprocess.STDOUT,
                               env={**os.environ, "OMP_NUM_THREADS": "1", **env})
        if r.returncode:
            sys.exit(f"{tag} failed (exit {r.returncode}); see {logs / (tag + '.log')}")
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
