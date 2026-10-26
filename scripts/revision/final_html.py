#!/usr/bin/env python3
"""Build the single-file HTML results page of the final analysis run from the
markdown in results/revision/final/ (pandoc, figures embedded). Aggregates only.
Output: results/revision/final/FINAL_RESULTS.html and public/docs/JMIR_Final_Analysis_Results.html."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIN = ROOT / "results" / "revision" / "final"
FIGS = [
    ("fig_R2_forest_paired_differences.png", "PSYCHE-D timing-aware ablation (R2): paired AUROC differences with participant-level 95% CIs."),
    ("fig_T9_calibration_frozen_forecast_selected.png", "Task 9: test-set calibration (10 decile bins) of the frozen, R3 forecasting and development-selected P+S+L models, before and after Platt scaling fitted on validation."),
    ("fig_T5_communication_curves_as_specified.png", "Task 5: validation AUROC vs cumulative communication, original tree implementation (same XGBoost seed every round). Left: frozen 152 inputs; right: P+S+L 41 inputs."),
    ("fig_T5_communication_curves_seed_varies.png", "Task 5: the same with a different XGBoost seed per round and client."),
]


def main():
    parts = [
        "---\ntitle: JMIR Final Analysis Run\nsubtitle: All remaining revision analyses (tasks 1-11), PSYCHE-D, HRV and PHQ-9, RESILIENT. 2026-10-16. Manuscript not edited.\n---\n",
        (FIN / "PLAIN_SUMMARY.md").read_text().split("\n", 1)[1].replace("\n## ", "\n# ", 1) if False else "# Summary\n" + (FIN / "PLAIN_SUMMARY.md").read_text().split("\n", 1)[1].replace("\n## ", "\n## "),
        (FIN / "TASK_RESULTS.md").read_text(),
        "# Figures\n" + "\n".join(f"![{cap}](figures/{f})\n" for f, cap in FIGS),
    ]
    for f in ["TABLE3_input_sets.md", "TABLE4_operating_points.md", "TABLE5_federated.md"]:
        t = (FIN / f).read_text()
        parts.append(t.replace("# Draft", "# Draft", 1).replace("\n## ", "\n### "))
    for f, title in [("CLAIM_CHANGES.md", None), ("METHODS_NOTES.md", None)]:
        t = (FIN / f).read_text()
        parts.append(t.replace("\n## ", "\n### ").replace("\n# ", "\n# ", 1))
    parts.append("# Files\n- Package: `results/revision/final/` (FINAL_NUMBERS.md, TABLE3-5, METHODS_NOTES.md, CLAIM_CHANGES.md, PLAIN_SUMMARY.md, figures/).\n"
                 "- Code: `scripts/revision/final_fl.py`, `final_outcome_calib.py`, `final_package.py`, `final_html.py`; runner `run_all_revision.py` (README_run_all.md).\n"
                 "- Full number list with sources: `results/revision/final/FINAL_NUMBERS.md`.\n")
    md = FIN / "_final_results.md"
    md.write_text("\n\n".join(parts))
    out = FIN / "FINAL_RESULTS.html"
    subprocess.run(["pandoc", str(md), "-o", str(out), "--standalone", "--embed-resources", "--toc", "--toc-depth=1",
                    "--css", str(FIN / "report.css"), "--resource-path", str(FIN), "--metadata", "lang=en"], check=True)
    md.unlink()
    shutil.copy(out, ROOT / "public" / "docs" / "JMIR_Final_Analysis_Results.html")
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
