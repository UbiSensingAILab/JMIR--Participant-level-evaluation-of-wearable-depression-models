# Participant-level evaluation of wearable depression models

Analysis code for the manuscript "Do Consumer Wearable Features Add Value Beyond Questionnaires for Assessing Depressive Symptoms? Participant-Level Secondary Analysis of 3 Public Data Sets" (Falah, Danesh Pajouh, Saeedi, Liang; manuscript prepared for JMIR mHealth and uHealth).

The code applies one standardized participant-level protocol to 3 public data sets (PSYCHE-D, an HRV and PHQ-9 data set, and RESILIENT): leakage audit, model screening, timing-aware input ablation, operating points and calibration, frozen MOMENT-1-small embeddings, and simulated federated learning (FedAvg, FedXgbBagging and TreeAvg) with measured communication.

## Contents

| Path | What it is |
|---|---|
| `scripts/revision/` | Final analyses reported in the manuscript; `run_all_revision.py` runs them in order (see `scripts/revision/README_run_all.md`) |
| `scripts/psyche_d_*`, `scripts/*psyche_d_severity_change*` | PSYCHE-D models, ablation, bootstrap, permutation importance, MOMENT, and the original federated learning experiments |
| `scripts/hrv_phq9_protocol_*` | Protocol stages 0-12 for the HRV and PHQ-9 data set |
| `scripts/resilient_protocol_*` | Protocol stages 0-12 for RESILIENT |
| `scripts/revision/manuscript_tools/build_v9_figures.py` | Manuscript figures from the saved results |
| `env/` | Package versions of the two Python environments |

## Data

No data are included. Download the 3 public data sets from their repositories, under their own licenses, and place them as follows:

| Data set | Source | Expected location |
|---|---|---|
| PSYCHE-D | Zenodo, doi:10.5281/zenodo.5085146 | `research_datasets/old/psyche_d/` (`anon_processed_df.parquet`) |
| HRV and PHQ-9 | Figshare, doi:10.6084/m9.figshare.28509740.v1 | `research_datasets/old/hrv_sleep_diaries/` |
| RESILIENT | Zenodo, doi:10.5281/zenodo.16755408 | `research_datasets/old/resilient/` |

Participant-level derived files (caches, out-of-fold predictions) are not shared. The scripts regenerate them in `scripts/.cache/` and `research_datasets/revision_cache/`.

## Environments

- `.venv-psyche-d-fl-mlp` (Python 3.13): every analysis step; `pip install -r env/requirements-analysis.txt`.
- `.venv311` (Python 3.11): MOMENT-1-small embeddings only; `pip install -r env/requirements-moment.txt`.

Run each step in its own process with `OMP_NUM_THREADS=1` (torch and LightGBM/XGBoost in one process can deadlock on macOS). `run_all_revision.py` does this.

## Order of runs

1. The per-data-set protocol stages (`hrv_phq9_protocol_stage*`, `resilient_protocol_stage*`) in numeric order, and `psyche_d_fedavg_vs_fedtree_step1_centralized.py` (builds the PSYCHE-D federated split).
2. `python3 scripts/revision/run_all_revision.py --moment` for the final analyses (about 50 minutes on an Apple silicon laptop).

## Generative AI use

The code was written and run with Claude Code (Anthropic) under the authors' direction; the authors checked the outputs. The full author prompts and transcripts will be added to `ai_use/` in a later release.

## License

MIT (see `LICENSE`). The data sets keep their own licenses.
