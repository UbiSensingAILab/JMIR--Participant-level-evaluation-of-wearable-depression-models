#!/usr/bin/env python3
"""Reproduce every analysis in the manuscript, from the 3 public data sets to the results package.

Usage (from the repository root):
    python3 reproduce.py                 # run all steps in order
    python3 reproduce.py --list          # show the steps
    python3 reproduce.py --from 12       # resume at step 12
    python3 reproduce.py --only 3 4      # run selected steps

Each step runs in its own process with OMP_NUM_THREADS=1 (torch and LightGBM/XGBoost in one process can
deadlock on macOS). Logs go to results/logs/<step>.log. See README.md for setup and data placement.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TREES = str(ROOT / ".venv-psyche-d-fl-mlp" / "bin" / "python")  # analysis environment
MOMENT = str(ROOT / ".venv311" / "bin" / "python")  # MOMENT-1-small environment

# (group, python, script relative to scripts/, args, extra env)
STEPS = [
    # A. PSYCHE-D: split, centralized and federated baselines, audits
    ("A PSYCHE-D", TREES, "psyche_d_fedavg_vs_fedtree_step1_centralized.py", [], {}),
    ("A PSYCHE-D", TREES, "psyche_d_fedavg_vs_fedtree_step23_fedavg.py", [], {}),
    ("A PSYCHE-D", TREES, "psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging.py", [], {}),
    ("A PSYCHE-D", TREES, "psyche_d_fedavg_vs_fedtree_step5_tree_avg.py", [], {}),
    ("A PSYCHE-D", TREES, "psyche_d_treeavg_audit_extend_s4_centralized_xgboost.py", [], {}),
    ("A PSYCHE-D", TREES, "psyche_d_treeavg_audit_extend_s5_lr_grid.py", [], {}),
    ("A PSYCHE-D", TREES, "audit_psyche_d_baseline.py", [], {}),
    ("A PSYCHE-D", TREES, "psyche_d_permutation_importance.py", [], {}),
    ("A PSYCHE-D", TREES, "psyche_d_moment_small.py", ["sequences"], {}),
    ("A PSYCHE-D", MOMENT, "psyche_d_moment_small.py", ["embed"], {}),
    # B. HRV and PHQ-9 data set: protocol stages 0-12
    ("B HRV", MOMENT, "hrv_phq9_protocol_stage00_01_profile_inventory.py", [], {}),  # numpy 1 pickle, read by step 16
    ("B HRV", TREES, "hrv_phq9_protocol_stage02_03_leakage_baseline.py", [], {}),
    ("B HRV", TREES, "hrv_phq9_protocol_stage04_ablation_freeze.py", [], {}),
    ("B HRV", TREES, "hrv_phq9_protocol_stage05_operating_points.py", [], {}),
    ("B HRV", MOMENT, "hrv_phq9_protocol_stage06a_hrm_minute_bins.py", [], {}),  # numpy 1 pickle, read by step 16
    ("B HRV", MOMENT, "hrv_phq9_protocol_stage06b_moment_embed.py", [], {}),
    ("B HRV", TREES, "hrv_phq9_protocol_stage06c_moment_eval.py", [], {}),
    ("B HRV", TREES, "hrv_phq9_protocol_stage07_08_fl.py", [], {}),
    ("B HRV", TREES, "hrv_phq9_protocol_stage09_robustness.py", [], {}),
    ("B HRV", TREES, "hrv_phq9_protocol_stage10_12_report.py", [], {}),
    # C. RESILIENT: protocol stages 0-12
    ("C RESILIENT", MOMENT, "resilient_protocol_stage00_01_profile_inventory.py", [], {}),  # numpy 1 pickle, read by step 24
    ("C RESILIENT", TREES, "resilient_protocol_stage02_03_leakage_baseline.py", [], {}),
    ("C RESILIENT", TREES, "resilient_protocol_stage04_05_ablation_freeze_thresholds.py", [], {}),
    ("C RESILIENT", MOMENT, "resilient_protocol_stage06a_moment_embed.py", [], {}),
    ("C RESILIENT", TREES, "resilient_protocol_stage06b_moment_arms.py", [], {}),
    ("C RESILIENT", TREES, "resilient_protocol_stage07_08_fl_exploratory.py", [], {}),
    ("C RESILIENT", TREES, "resilient_protocol_stage09_robustness.py", [], {}),
    ("C RESILIENT", TREES, "resilient_protocol_stage10_12_report.py", [], {}),
    # D. Final analyses reported in the manuscript
    ("D Final", TREES, "revision/p0_audit.py", [], {}),
    ("D Final", TREES, "revision/p1_cv.py", [], {}),
    ("D Final", TREES, "revision/p1_temporal.py", [], {}),
    ("D Final", TREES, "revision/p1_holdout.py", [], {}),
    ("D Final", TREES, "revision/p2_fl_rerun.py", [], {}),
    ("D Final", TREES, "revision/p2_fl_noniid.py", [], {}),
    ("D Final", TREES, "revision/p2_mb_selected.py", [], {}),
    ("D Final", TREES, "revision/p3_small_datasets.py", [], {}),
    ("D Final", TREES, "revision/r_tier1_cv.py", [], {}),
    ("D Final", TREES, "revision/r_tier1_split.py", [], {}),
    ("D Final", TREES, "revision/r_tier1_fl.py", [], {}),
    ("D Final", TREES, "revision/final_fl.py", [], {}),
    ("D Final", TREES, "revision/final_fl.py", [], {"FL_VARY_SEED": "1"}),
    ("D Final", TREES, "revision/final_fl_vs_phq9.py", [], {}),
    ("D Final", TREES, "revision/final_outcome_calib.py", [], {}),
    ("D Final", TREES, "revision/final_fig3_deciles.py", [], {}),
    ("D Final", TREES, "revision/final_package.py", [], {}),
    ("D Final", TREES, "revision/manuscript_tools/build_v9_figures.py", [], {}),
]


def label(i, s):
    env = " ".join(f"{k}={v}" for k, v in s[4].items())
    return f"{i:2d}. [{s[0]}] {env + ' ' if env else ''}{s[2]} {' '.join(s[3])}".rstrip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list the steps and exit")
    ap.add_argument("--from", dest="start", type=int, default=1, help="first step to run")
    ap.add_argument("--only", type=int, nargs="+", help="run only these steps")
    a = ap.parse_args()
    if a.list:
        for i, s in enumerate(STEPS, 1):
            print(label(i, s))
        return
    for py in {TREES, MOMENT}:
        if not Path(py).exists():
            sys.exit(f"Missing environment: {py}. See README.md, section 2.")
    logs = ROOT / "results" / "logs"
    # output folders some scripts write to (summaries, report pages, figures)
    for d in (logs, ROOT / "results" / "revision" / "final", ROOT / "scripts" / ".cache", ROOT / "research_datasets" / "revision_cache", ROOT / "src" / "data" / "protocolComparison",
              ROOT / "public" / "docs" / "paper_figures" / "hrv_phq9", ROOT / "public" / "docs" / "paper_figures" / "resilient_protocol"):
        d.mkdir(parents=True, exist_ok=True)
    chosen = [i for i in range(1, len(STEPS) + 1) if (a.only and i in a.only) or (not a.only and i >= a.start)]
    t0 = time.time()
    for i in chosen:
        grp, py, script, args, env = STEPS[i - 1]
        path = ROOT / "scripts" / script
        tag = f"{i:02d}_{Path(script).stem}{'_' + '_'.join(args) if args else ''}{'_seedvary' if env else ''}"
        print(f"[{time.time() - t0:6.0f}s] {label(i, STEPS[i - 1])}", flush=True)
        with open(logs / f"{tag}.log", "w") as fh:
            r = subprocess.run([py, str(path), *args], cwd=path.parent, stdout=fh, stderr=subprocess.STDOUT,
                               env={**os.environ, "OMP_NUM_THREADS": "1", **env})
        if r.returncode:
            sys.exit(f"Step {i} failed (exit {r.returncode}); see {logs / (tag + '.log')}. Resume with --from {i}.")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Results: results/revision/final/ (see README.md, section 5).")


if __name__ == "__main__":
    main()
