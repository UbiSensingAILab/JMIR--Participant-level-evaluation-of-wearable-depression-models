# Evaluating Wearable Sensing Using Federated Learning for Early Awareness of Depressive Symptoms

Analysis code for the manuscript:

> Falah F, Danesh Pajouh MM, Liang SHL, Saeedi S. *Evaluating Wearable Sensing Using Federated Learning for Early Awareness of Depressive Symptoms: A Participant-Level Analysis of 3 Public Data Sets.* Manuscript prepared for JMIR mHealth and uHealth.

**Authors:** Faraneh Falah¹, MSc; Mohammad Mahdi Danesh Pajouh², MSc; Steve H L Liang², PhD; Sara Saeedi¹, PhD

¹Department of Electrical and Software Engineering, Schulich School of Engineering, University of Calgary, Calgary, AB, Canada  
²UbiSensing AI Research Lab, University of Calgary, Calgary, AB, Canada

**Contact:** Faraneh Falah (farane.falah@ucalgary.ca) and Sara Saeedi (ssaeedi@ucalgary.ca), Department of Electrical and Software Engineering, Schulich School of Engineering, University of Calgary, 2500 University Drive NW, Calgary, AB T2N 1N4, Canada

The code applies one standardized, participant-level protocol to 3 public data sets (PSYCHE-D, an HRV and PHQ-9 data set, and RESILIENT):

- a leakage audit (record-level vs participant-level splits, label shuffle)
- screening of 7 machine learning models
- a timing-aware ablation of questionnaire, contextual, and Fitbit inputs, and forecasting
- operating points, positive predictive values, and calibration (Platt scaling)
- frozen MOMENT-1-small time-series embeddings
- simulated federated learning (FedAvg, FedXgbBagging, and TreeAvg) against centralized and single-client training, with measured communication

No data are included in this repository.

---

## Contents

| Path | Contents |
|---|---|
| `reproduce.py` | Runs every analysis in order, from the raw data sets to the results package |
| `scripts/psyche_d_*`, `scripts/audit_psyche_d_baseline.py` | PSYCHE-D: data split, centralized and federated models, audits, permutation importance, MOMENT |
| `scripts/hrv_phq9_protocol_*` | HRV and PHQ-9 data set: protocol stages 0 to 12 |
| `scripts/resilient_protocol_*` | RESILIENT: protocol stages 0 to 12 |
| `scripts/revision/` | Final analyses reported in the manuscript, the results package, and the manuscript figures |
| `mood_monitoring_agent/` | Frozen MOMENT-1-small wrapper used by the embedding steps |
| `env/` | Exact package versions of the two Python environments |
| `CITATION.cff` | How to cite this code |

---

## Step-by-step reproduction

### Step 1. Requirements

- macOS or Linux. Tested on an Apple M3 laptop with 16 GB RAM; no GPU is needed.
- About 25 GB of free disk space for the data (the unzipped HRV and PHQ-9 sensor files are about 18 GB), 3 GB for the 2 environments, and under 100 MB for outputs.
- Python 3.13 (analysis) and Python 3.11 (MOMENT-1-small only).
- Git.

### Step 2. Get the code and create the 2 environments

```bash
git clone https://github.com/UbiSensingAILab/JMIR--Participant-level-evaluation-of-wearable-depression-models.git
cd JMIR--Participant-level-evaluation-of-wearable-depression-models

# Analysis environment (every step except the MOMENT embeddings)
python3.13 -m venv .venv-psyche-d-fl-mlp
.venv-psyche-d-fl-mlp/bin/pip install -r env/requirements-analysis.txt

# MOMENT-1-small environment (embedding steps only)
python3.11 -m venv .venv311
.venv311/bin/pip install -r env/requirements-moment.txt
```

The environment folder names must stay as shown; `reproduce.py` looks for them in the repository root. The first MOMENT step downloads the `AutonLab/MOMENT-1-small` weights from Hugging Face (about 150 MB).

### Step 3. Download the 3 data sets

Download each data set from its repository and accept its license. Then place the files as shown. Unzip any `.zip` archives in place.

| Data set | Source and license | Place in | Files needed |
|---|---|---|---|
| PSYCHE-D | Zenodo, [doi:10.5281/zenodo.5085146](https://doi.org/10.5281/zenodo.5085146), CC BY-NC 4.0 | `research_datasets/old/psyche_d/` | `anon_processed_df.parquet`, `feature_explanations.tsv` |
| HRV and PHQ-9 | Figshare, [doi:10.6084/m9.figshare.28509740.v1](https://doi.org/10.6084/m9.figshare.28509740.v1), CC BY 4.0 | `research_datasets/old/hrv_sleep_diaries/` | `README.docx`, `survey.csv`, `sleep_diary.csv`, `sensor_hrv.csv`, `sensor_hrv_filtered.csv`, and the unzipped `raw_data/` folder (the analyses read `hrm.csv.gz` and `lit.csv.gz`; the other sensor files are only listed in the data profile) |
| RESILIENT | Zenodo, [doi:10.5281/zenodo.16755408](https://doi.org/10.5281/zenodo.16755408) (version v3) | `research_datasets/old/resilient/` | `Demographics.csv`, `summary_stats_per_participant.csv`, and the unzipped `Resilient_metadata/` and `Sleepmat_Watch_Data/` folders |

Expected layout:

```
research_datasets/old/
├── psyche_d/            anon_processed_df.parquet, feature_explanations.tsv
├── hrv_sleep_diaries/   README.docx, survey.csv, sleep_diary.csv, sensor_hrv*.csv, raw_data/
└── resilient/           Demographics.csv, summary_stats_per_participant.csv, Resilient_metadata/, Sleepmat_Watch_Data/
```

### Step 4. Run the analyses

```bash
python3 reproduce.py --list    # show the 46 steps
python3 reproduce.py           # run them all
```

`reproduce.py` runs 4 groups of steps, each script in its own process:

| Group | Steps | What it does |
|---|---|---|
| A. PSYCHE-D | 1-10 | Participant-level split; centralized LightGBM and XGBoost; original FedAvg, FedXgbBagging, and TreeAvg runs; baseline audit; permutation importance; MOMENT sequences and embeddings |
| B. HRV and PHQ-9 | 11-20 | Protocol stages 0-12: profile, leakage audit, model screen, ablation, operating points, MOMENT, federated learning, robustness, report |
| C. RESILIENT | 21-28 | Protocol stages 0-12, as above (federated learning is exploratory) |
| D. Final analyses | 29-46 | Analyses reported in the manuscript, the results package, and the manuscript figures |

If a step fails, the run stops and prints the log file. Fix the cause and resume with `python3 reproduce.py --from <step>`. Logs are in `results/logs/`.

A full run took about 2 hours on an Apple M3 laptop (16 GB RAM): about 15 minutes for PSYCHE-D, 15 minutes for the HRV and PHQ-9 data set, 55 minutes for RESILIENT, and 30 minutes for the final analyses. The longest single step is the RESILIENT ablation (step 23, about 18 minutes).

### Step 5. Find the results

| Manuscript item | Output file |
|---|---|
| Table 4 (input groups and discrimination) | `results/revision/final/TABLE3_input_sets.md` |
| Table 5 (operating points) | `results/revision/final/TABLE4_operating_points.md` |
| Table 6 (federated learning) | `results/revision/final/TABLE5_federated.md` |
| Figures 1-4 | `results/revision/final/manuscript_figures/Figure1.png` to `Figure4.png` |
| Every number quoted in the Abstract and Results | `results/revision/final/FINAL_NUMBERS.md` |
| Multimedia Appendix 2 tables (CSV) | `results/revision/*.csv` (for example `T2_T4_per_seed_auroc_seedvary.csv`, `T3_local_only_seedvary.csv`, `T5_communication_summary_seedvary.csv`) |

The draft table files are numbered as in the analysis plan (Tables 3-5); in the manuscript they are Tables 4-6. Tables 1-3 of the manuscript describe prior work, the data sets, and input timing, and are not computed.

---

## Notes

- **Federated learning is simulated.** The 2825 PSYCHE-D training participants are split at random into 5 clients (3 partition seeds), with shared validation and test portions; a non-IID check uses birth-year clients.
- **Tree implementation.** The manuscript's primary federated results use a different XGBoost seed for each client tree in each round (`FL_VARY_SEED=1`, step 41). Step 40 reruns the original implementation, which reused one seed in every round; its results are reported in Multimedia Appendix 2.
- **Participant privacy.** Participant-level intermediate files (caches, out-of-fold predictions) are written to `scripts/.cache/` and `research_datasets/revision_cache/` and are excluded from version control by `.gitignore`. Do not commit them.
- **Reproducibility check.** Seeds are fixed in every script. A clean run of `reproduce.py` from the raw data sets, on the authors' machine with the package versions in `env/`, reproduced the manuscript tables (`TABLE3`-`TABLE5`) exactly, and 60 of the 67 results CSV files byte for byte. The other differences were at or below the third decimal place (for example, one HRV and PHQ-9 label-shuffle mean of 0.459 vs 0.462 reported in Multimedia Appendix 2). Other platforms or package versions can change results in the last decimal places.
- **Thread settings.** Each step runs with `OMP_NUM_THREADS=1`, because torch and LightGBM or XGBoost in one process can deadlock on macOS.

## Citation

Please cite the manuscript and this repository (see `CITATION.cff`).

## License

The code is released under the MIT License (see `LICENSE`). The data sets keep their own licenses.
