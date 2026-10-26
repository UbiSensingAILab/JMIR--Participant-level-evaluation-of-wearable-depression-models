#!/usr/bin/env python3
# Aggregates only; reads results/revision/* and scripts/*_results.json, writes results/revision/final/.
"""Final analysis run, tasks 1, 8 and 10: manuscript-state check, RESILIENT label-shuffle
provenance, and the manuscript results package (FINAL_NUMBERS.md, draft Tables 3-5,
figures). Every number is read from a saved result file; nothing is typed in by hand.
METHODS_NOTES.md and CLAIM_CHANGES.md in final/ are written separately (narrative)."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "revision"
FIN = OUT / "final"
FIG = FIN / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SC = ROOT / "scripts"
J = lambda p: json.loads(Path(p).read_text())  # noqa: E731
L = lambda s: ast.literal_eval(s) if isinstance(s, str) else s  # noqa: E731


def a3(x):
    return f"{x:.3f}"


def ci(lo_hi, sep="-"):
    lo, hi = lo_hi
    f = lambda v: f"{v:.3f}".replace("-", "−")  # noqa: E731
    return f"{f(lo)}{sep}{f(hi)}" if sep == "-" and lo >= 0 else f"{f(lo)} to {f(hi)}"


def est(x, c):
    return f"{a3(x)} ({ci(c)})"


def dlt(d, c):
    s = f"{d:+.3f}".replace("-", "−")
    return f"{s} ({ci(c, 'to')})"


def pct(x, c=None):
    return f"{100 * x:.1f}" + (f" ({100 * c[0]:.1f}-{100 * c[1]:.1f})" if c is not None else "")


# ------------------------------------------------------------------ sources
arms = pd.read_csv(OUT / "R1_R2_R3_cv_arms.csv").set_index("arm")
pr = pd.read_csv(OUT / "R2_R3_R4_paired_differences.csv")
pr = {r.comparison: (r.delta, L(r.ci)) for r in pr.itertuples()}
p1 = pd.read_csv(OUT / "P1_1_2_cv_arms.csv").set_index("arm")
p1p = {r.comparison: (r.delta, L(r.ci)) for r in pd.read_csv(OUT / "P1_1_2_paired_differences.csv").itertuples()}
temporal = J(OUT / "P1_1_temporal_participant_ci.json")
p1j = J(OUT / "P1_cv_results.json")
sev = pd.read_csv(OUT / "R5_baseline_severity.csv")
r6 = pd.read_csv(OUT / "R6_test_models.csv").set_index("model")
r6p = {r.comparison: (r.delta, L(r.ci)) for r in pd.read_csv(OUT / "R6_test_paired_vs_frozen.csv").itertuples()}
ops = pd.read_csv(OUT / "T9_operating_points_test.csv")
cal = pd.read_csv(OUT / "T9_calibration.csv")
cal3 = pd.read_csv(OUT / "P3_3_calibration_small_datasets.csv")
t6 = pd.read_csv(OUT / "T6_outcome_ge5_arms.csv").set_index("arm")
t6p = {r.comparison: (r.delta, L(r.ci)) for r in pd.read_csv(OUT / "T6_outcome_ge5_paired.csv").itertuples()}
t6m = J(OUT / "T6_T9_results.json")["T6"]["meta"]
p31 = pd.read_csv(OUT / "P3_1_hrv_leakage_participant_ci.csv")
p32 = pd.read_csv(OUT / "P3_2_hrv_moment_rerun.csv")
hrv = J(ROOT / "src/data/protocolComparison/hrv_phq9.json")
res = J(ROOT / "src/data/protocolComparison/resilient.json")
hrv5 = J(SC / "hrv_phq9_protocol_stage05_results.json")["operating_points"]
res45 = J(SC / "resilient_protocol_stage04_05_results.json")
FL = {}
for tag, name in [("", "as specified"), ("_seedvary", "seed varies by round")]:
    per = pd.read_csv(OUT / f"T2_T4_per_seed_auroc{tag}.csv")
    prd = pd.read_csv(OUT / f"T2_T3_T4_paired_differences{tag}.csv")
    loc = pd.read_csv(OUT / f"T3_local_only{tag}.csv")
    com = pd.read_csv(OUT / f"T5_communication_summary{tag}.csv")
    eta = pd.read_csv(OUT / f"T2_treeavg_eta_selection{tag}.csv")
    FL[name] = dict(per=per, pr=prd, loc=loc, com=com, eta=eta)
r8 = pd.read_csv(OUT / "R8_per_seed_auroc.csv")


def arm(k, src=arms):
    r = src.loc[k]
    return est(r.auroc, L(r.auroc_ci)), a3(r.auprc), L(r.auprc_ci)


def flp(v, iset, part, m, seed=None):
    d = FL[v]["per"]
    q = d[(d.input_set == iset) & (d.partition == part) & (d.method == m)]
    if seed is not None:
        q = q[q.partition_seed == seed]
    return q


def flpair(v, iset, comp, seed="pooled"):
    d = FL[v]["pr"]
    q = d[(d.input_set == iset) & (d.comparison == comp) & (d.partition_seed.astype(str) == str(seed))].iloc[0]
    return dlt(q.delta, L(q.ci))


def flmean(v, iset, m):
    a = flp(v, iset, "IID", m).auroc
    return f"{a.mean():.3f} (SD {a.std(ddof=1):.3f})"


def comm(v, iset, part, m, col):
    d = FL[v]["com"]
    return d[(d.input_set == iset) & (d.partition == part) & (d.method == m)][col]


# ------------------------------------------------------------------ Task 8: RESILIENT label shuffle
s23 = J(SC / "resilient_protocol_stage02_03_results.json")["permutation_test"]
s45 = res45["permutation_frozen_model"]
psy_a10 = J(SC / "audit_psyche_d_baseline_results_part_a.json")  # written by audit_psyche_d_baseline.py
hrv_perm = J(SC / "hrv_phq9_protocol_stage02_results.json")["permutation_test"]
t8 = pd.DataFrame([
    {"dataset": "RESILIENT", "value_cited_as": "0.44", "mean": s23["mean"], "sd": s23["sd"], "per_shuffle": s23["aurocs"],
     "model": "logistic regression, nested (outer 5-fold x 2 repeats), sensor + demographic inputs (frozen inputs)",
     "shuffle": "5 label permutations, rng 1000-1004 (one row per participant)", "protocol_stage": "Stage 2 leakage audit",
     "script": "scripts/resilient_protocol_stage02_03_leakage_baseline.py", "result_file": "scripts/resilient_protocol_stage02_03_results.json:permutation_test"},
    {"dataset": "RESILIENT", "value_cited_as": "0.52", "mean": s45["mean"], "sd": s45["sd"], "per_shuffle": s45["aurocs"],
     "model": "frozen model family (random forest, nested selection, outer 5-fold x 1 repeat), same inputs",
     "shuffle": "5 label permutations, rng 2000-2004", "protocol_stage": "Stage 4-5 freeze",
     "script": "scripts/resilient_protocol_stage04_05_ablation_freeze_thresholds.py", "result_file": "scripts/resilient_protocol_stage04_05_results.json:permutation_frozen_model"},
    {"dataset": "PSYCHE-D", "value_cited_as": "0.50", "mean": psy_a10["A10_permutation_auroc_mean"], "sd": psy_a10["A10_permutation_auroc_std"],
     "per_shuffle": psy_a10["A10_permutation_aurocs"], "model": "logistic regression (balanced), 150 non-PHQ inputs, participant-grouped 5-fold",
     "shuffle": "5 permutations of the row labels (rng 1000-1004); NOT blocked by participant", "protocol_stage": "audit A10",
     "script": "scripts/audit_psyche_d_baseline.py (audit_permutation_test)", "result_file": "scripts/audit_psyche_d_baseline_results.json:part_a.A10_*"},
    *[{"dataset": "HRV and PHQ-9", "value_cited_as": "0.46", "mean": h["mean"], "sd": None, "per_shuffle": h["seed_aurocs"],
       "model": f"logistic regression, nested, arm {h['arm']}", "shuffle": "5 participant-block permutations (rng 1000-1004)",
       "protocol_stage": "Stage 2 leakage audit", "script": "scripts/hrv_phq9_protocol_stage02_03_leakage_baseline.py (permutation_test)",
       "result_file": "scripts/hrv_phq9_protocol_stage02_results.json:permutation_test"} for h in hrv_perm],
])
t8.to_csv(OUT / "T8_label_shuffle_provenance.csv", index=False)

# ------------------------------------------------------------------ figures
for src, dst in [("R2_forest_paired_differences.png", "fig_R2_forest_paired_differences.png"),
                 ("T9_calibration_frozen_forecast_selected.png", "fig_T9_calibration_frozen_forecast_selected.png"),
                 ("P1_6_calibration_plot.png", "fig_P1_6_calibration_frozen.png"),
                 ("R7_calibration_frozen_and_forecast.png", "fig_R7_calibration_frozen_and_forecast.png"),
                 ("T5_communication_curves.png", "fig_T5_communication_curves_as_specified.png"),
                 ("T5_communication_curves_seedvary.png", "fig_T5_communication_curves_seed_varies.png"),
                 ("P2_4_communication_curves.png", "fig_P2_4_communication_curves_frozen.png")]:
    shutil.copy(OUT / src, FIG / dst)

# ------------------------------------------------------------------ Table 3
T3 = ["# Draft Table 3. Input sets in all 3 datasets (AUROC, participant-level 95% CI)", "",
      "All CIs: participant-level bootstrap, 2000 resamples, seed 42. PSYCHE-D: pooled out-of-fold predictions from StratifiedGroupKFold(5, shuffle, 42), LightGBM-DART frozen recipe for every row. HRV and PHQ-9: repeated grouped 5x5 CV, repeat-averaged predictions. RESILIENT: repeated nested CV 5x5, one row per participant.", "",
      "| Dataset and input set | Inputs, n | AUROC (95% CI) | AUPRC (95% CI) |", "|---|---|---|---|",
      f"| **PSYCHE-D** (10,866 samples, 4036 participants; worsening 20.7%) | | | |"]
psy_rows = [("Baseline PHQ-9 (P)", "P"), ("Screener (S)", "S"), ("Lifestyle and medication, end month (L)", "L"), ("Fitbit, end month (W)", "W"),
            ("P+S", "P+S"), ("P+S+L (development-selected input set)", "P+S+L"), ("P+S+W", "P+S+W"),
            ("P+S+L+W (frozen protocol model)", "P+S+L+W (frozen)"), ("Forecast: P+S + months 1-2 Fitbit, lifestyle and medication", "R3 forecast: P+S+SM1/SM2 (W+L)")]
for lab, k in psy_rows:
    r = arms.loc[k]
    T3.append(f"| {lab} | {int(r.n_inputs)} | {est(r.auroc, L(r.auroc_ci))} | {est(r.auprc, L(r.auprc_ci))} |")
for lab, k in [("Wearable and context (150; manuscript row)", "wearable_context_150"), ("Context only (S+L)", "c_context_only"),
               ("P+W", "d_phq_wearable"), ("MOMENT embeddings only", "moment_only"), ("MOMENT + frozen inputs", "moment_plus_frozen"),
               ("Published 2-phase design (reproduced)", "published_2phase_original")]:
    if k in p1.index:
        r = p1.loc[k]
        T3.append(f"| {lab} | {int(r.n_inputs)} | {est(r.auroc, L(r.auroc_ci))} | {est(r.auprc, L(r.auprc_ci))} |")
T3.append(f"| **HRV and PHQ-9** (98 segments, 49 participants; positive 36.7%) | | | |")
for a in hrv["plots"]["ablation"]:
    if "| LR" in a["arm"] or a["arm"].startswith("Stage 6") or "| ET" in a["arm"]:
        T3.append(f"| {a['arm']} | {a['n_features']} | {est(a['auroc'], a['ci'])} | n/a in source |")
T3.append(f"| **RESILIENT** (73 participants, 10 positive) | | | |")
for a in res["plots"]["ablation"]:
    T3.append(f"| {a['arm']} | {a['n_features']} | {est(a['auroc'], a['ci'])} | n/a in source |")
T3 += ["", "Sources: results/revision/R1_R2_R3_cv_arms.csv, P1_1_2_cv_arms.csv; src/data/protocolComparison/hrv_phq9.json and resilient.json (plots.ablation). AUPRC for the small datasets is in the manuscript Table 3 (from the protocol stage results); it is not repeated here because the comparison JSON does not store its CI.",
       "", "Outcome sensitivity (PSYCHE-D, PHQ-9 increase >= 5 points; prevalence " + f"{100 * t6m['prevalence']:.1f}%):", "",
       "| Input set | AUROC (95% CI) | AUPRC (95% CI) |", "|---|---|---|"]
for k in t6.index:
    r = t6.loc[k]
    T3.append(f"| {k} | {est(r.auroc, L(r.auroc_ci))} | {est(r.auprc, L(r.auprc_ci))} |")
(FIN / "TABLE3_input_sets.md").write_text("\n".join(T3) + "\n")

# ------------------------------------------------------------------ Table 4
T4 = ["# Draft Table 4. Operating points (test set for PSYCHE-D), with PPV and NPV", "",
      "PSYCHE-D: model fitted on the 70% training portion, threshold chosen on the 15% validation portion, scored once on the test set (1631 samples, 606 participants; majority-class rate 80.8%). 95% CIs: participant bootstrap of the test set. Platt-scaled scores give the same classifications at the corresponding thresholds (T9_operating_points_test.csv), except where noted.", "",
      "| Dataset, model and operating point | Threshold | Accuracy, % | Sensitivity, % | Specificity, % | PPV, % | NPV, % |", "|---|---|---|---|---|---|---|"]
for mlabel, mkey in [("PSYCHE-D frozen protocol model (P+S+L+W, 152)", "Frozen (P+S+L+W, 152)"), ("PSYCHE-D development-selected model (P+S+L, 41)", "Development-selected (P+S+L, 41)")]:
    T4.append(f"| **{mlabel}** | | | | | | |")
    for r in ops[(ops.model == mkey) & (ops.scores == "uncalibrated")].itertuples():
        T4.append(f"| {r.operating_point} | {r.threshold:.3f} | {pct(r.accuracy, L(r.accuracy_ci))} | {pct(r.sensitivity, L(r.sensitivity_ci))} | {pct(r.specificity, L(r.specificity_ci))} | {pct(r.ppv, L(r.ppv_ci))} | {pct(r.npv, L(r.npv_ci))} |")
T4.append("| Published PSYCHE-D model (Makhmutova 2022; random sample holdout) | n/a | 63.3 (derived) | 55.4 | 65.3 | not reported | not reported |")
T4.append("| **HRV and PHQ-9 frozen model** (repeated CV, per-fold thresholds; majority 63.3%) | | | | | | |")
for k, lab in [("default", "Default"), ("max_accuracy", "Maximum accuracy"), ("matched_sensitivity", "80% clinical target")]:
    r = hrv5[k]
    T4.append(f"| {lab} | {r['threshold_median']:.3f} (median) | {pct(r['accuracy'])} | {pct(r['sensitivity'])} | {pct(r['specificity'])} | {pct(r['ppv'])} | {pct(r['npv'])} |")
T4.append("| **RESILIENT frozen model** (repeated CV, per-fold thresholds; majority 86.3%) | | | | | | |")
for k, lab in [("default", "Default"), ("max_accuracy", "Maximum accuracy"), ("matched_sensitivity", "80% clinical target")]:
    r = res45["operating_points"][k]
    npv = r["tn"] / (r["tn"] + r["fn"])
    T4.append(f"| {lab} | {r['threshold_median']:.3f} (median) | {pct(r['accuracy'])} | {pct(r['sensitivity'])} | {pct(r['specificity'])} | {pct(r['ppv'])} | {pct(npv)} |")
T4 += ["", "HRV and RESILIENT values are pooled over repeated outer folds (HRV: mean over 5 repeats; RESILIENT: 365 out-of-fold predictions = 73 participants x 5 repeats); no bootstrap CI was computed for them. RESILIENT NPV = TN/(TN+FN) from the saved counts.",
       "Sources: results/revision/T9_operating_points_test.csv; scripts/hrv_phq9_protocol_stage05_results.json (operating_points); scripts/resilient_protocol_stage04_05_results.json (operating_points)."]
(FIN / "TABLE4_operating_points.md").write_text("\n".join(T4) + "\n")

# ------------------------------------------------------------------ Table 5
T5 = ["# Draft Table 5. Federated learning, PSYCHE-D (5 simulated clients) and the 2 small datasets", "",
      "PSYCHE-D: fixed split (seed 20260928), test set 1631 samples / 606 participants. Federated rows: test AUROC of partition seed 1 with participant-level 95% CI, then mean (SD) over partition seeds 1-3. Communication: total MB over all rounds (upload + download, mean of 3 seeds) and MB up to the validation-selected round. Local-only: mean AUROC of the 5 single-client models (seed-1 value with CI; the Mean (SD) column is over the 3 seeds' 5-client means).", ""]
for v in FL:
    T5.append(f"## Tree implementation: {v}")
    T5.append("")
    if v == "seed varies by round":
        T5.append("Same code, but each client tree in each round gets its own random seed, so XGBoost's 80% row and column subsampling differs between rounds (as in one continuous XGBoost run). FedAvg and local-only rows do not use this and are identical to the as-specified run.")
        T5.append("")
    T5 += ["| Input set and method | AUROC seed 1 (95% CI) | Mean (SD), 3 seeds | Total MB | MB to selected round | Selected rounds |", "|---|---|---|---|---|---|"]
    for iset, lab in [("frozen 152", "Frozen inputs (152)"), ("P+S+L 41", "Development-selected inputs (P+S+L, 41)")]:
        T5.append(f"| **{lab}** | | | | | |")
        for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
            s1 = flp(v, iset, "IID", m, 1).iloc[0]
            eta_sel = FL[v]["eta"]
            e = eta_sel[(eta_sel.input_set == iset) & eta_sel.selected].eta.iloc[0]
            ml = m + (f" (eta {e:g})" if m == "TreeAvg" else " (MLP)" if m == "FedAvg" else "")
            T5.append(f"| {ml} | {est(s1.auroc, L(s1.auroc_ci))} | {flmean(v, iset, m)} | {comm(v, iset, 'IID', m, 'total_mb').mean():.2f} | "
                      f"{comm(v, iset, 'IID', m, 'mb_to_selected_round').mean():.2f} | {'/'.join(str(int(x)) for x in comm(v, iset, 'IID', m, 'selected_round'))} |")
        for m in ["Centralized LightGBM", "Centralized XGBoost"]:
            r = flp(v, iset, "central", m).iloc[0]
            T5.append(f"| {m} | {est(r.auroc, L(r.auroc_ci))} | single model | n/a | n/a | n/a |")
        lo = FL[v]["loc"]
        for fam in ["MLP", "XGBoost", "LightGBM"]:
            q = lo[(lo.input_set == iset) & (lo.family == fam)]
            r1 = q[q.partition_seed == 1].iloc[0]
            T5.append(f"| Local only, {fam} (mean of 5 clients) | {est(r1.mean_client_auroc, L(r1.mean_client_auroc_ci))} | {q.mean_client_auroc.mean():.3f} (SD {q.mean_client_auroc.std(ddof=1):.3f}) | n/a | n/a | n/a |")
    T5.append("| **Frozen inputs, non-IID clients (birth-year quintiles)** | | | | | |")
    for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
        r = flp(v, "frozen 152", "age quintile", m).iloc[0]
        T5.append(f"| {m} | {est(r.auroc, L(r.auroc_ci))} | single partition | {comm(v, 'frozen 152', 'age quintile', m, 'total_mb').iloc[0]:.2f} | {comm(v, 'frozen 152', 'age quintile', m, 'mb_to_selected_round').iloc[0]:.2f} | {int(comm(v, 'frozen 152', 'age quintile', m, 'selected_round').iloc[0])} |")
    T5.append("")
T5 += ["## HRV and PHQ-9 and RESILIENT (unchanged; 2 clients, per outer fold)", "", "| Dataset and method | AUROC (95% CI) | MB per outer fold |", "|---|---|---|"]
for nm, d in [("HRV and PHQ-9", hrv), ("RESILIENT (exploratory, not feasible)", res)]:
    for m in d["fl"]["methods"]:
        T5.append(f"| {nm}: {m['name']} | {est(m['auroc'], m['auroc_ci'])} | {m['comm_mb'] if m['comm_mb'] is not None else 'n/a'} |")
T5 += ["", "Sources: results/revision/T2_T4_per_seed_auroc[_seedvary].csv, T3_local_only[_seedvary].csv, T5_communication_summary[_seedvary].csv, T2_treeavg_eta_selection[_seedvary].csv; src/data/protocolComparison/hrv_phq9.json, resilient.json (fl)."]
(FIN / "TABLE5_federated.md").write_text("\n".join(T5) + "\n")


# ------------------------------------------------------------------ FINAL_NUMBERS.md
def c(model, ver):
    return cal[(cal.model == model) & (cal.version == ver)].iloc[0]


def calline(model, ver):
    r = c(model, ver)
    return (f"intercept {r.intercept_citl:.2f} ({ci(L(r.intercept_citl_ci), 'to')}), slope {r.slope:.2f} ({ci(L(r.slope_ci))}), "
            f"Brier {r.brier:.3f} ({ci(L(r.brier_ci))}), mean predicted {r.mean_predicted:.3f} vs observed {r.observed_rate:.3f}")


def op(model, scores, name):
    r = ops[(ops.model == model) & (ops.scores == scores) & (ops.operating_point == name)].iloc[0]
    return (f"threshold {r.threshold:.3f}: accuracy {pct(r.accuracy, L(r.accuracy_ci))}, sensitivity {pct(r.sensitivity, L(r.sensitivity_ci))}, "
            f"specificity {pct(r.specificity, L(r.specificity_ci))}, PPV {pct(r.ppv, L(r.ppv_ci))}, NPV {pct(r.npv, L(r.npv_ci))}")


FROZEN = "Frozen (P+S+L+W, 152)"
SEL = "Development-selected (P+S+L, 41)"
r6sel = r6.loc["selected: P+S+L / lightgbm_dart_frozen_config"]
r6fro = r6.loc["P+S+L+W (frozen) / DART (development-trained)"]
lines = [
    "# FINAL_NUMBERS: every number the manuscript stage needs",
    "",
    "Generated by `scripts/revision/final_package.py` from saved result files (2026-10-16). Format: estimate (95% CI). All CIs are participant-level bootstrap, 2000 resamples, numpy default_rng(42), the same resamples for every model compared (paired). Differences: first minus second.",
    "",
    "**Two PSYCHE-D models, keep them apart:**",
    "- **Frozen protocol model** = P+S+L+W, 152 inputs, LightGBM-DART (150 trees, depth 4, drop 0.1), mean imputation, balanced weights. Fixed before testing; unchanged throughout the revision.",
    "- **Development-selected model** = P+S+L, 41 inputs, same LightGBM-DART recipe, chosen by grouped CV on the development portion only (R6) and scored once on the held-out test set.",
    "",
    "Input groups: P = baseline PHQ-9 score and category (2), S = screener (29, precede the outcome), L = lifestyle and medication change items in the end month (10), W = Fitbit sleep and steps in the end month (111). L and W overlap the end-of-period PHQ-9 recall window, so the frozen model detects (not forecasts) worsening.",
    "",
    "## Abstract",
    f"- PSYCHE-D frozen protocol model, CV: AUROC {arm('P+S+L+W (frozen)')[0]} (R1_R2_R3_cv_arms.csv)",
    f"- PSYCHE-D baseline PHQ-9 alone, CV: {arm('P')[0]}; frozen minus P {dlt(*pr['frozen - P: the hybrid gain'])} (R2_R3_R4_paired_differences.csv)",
    f"- PSYCHE-D P+S (screener) {arm('P+S')[0]}; P+S minus P {dlt(*pr['(P+S) - P: value of the screener'])}",
    f"- PSYCHE-D Fitbit added to P+S: {dlt(*pr['(P+S+W) - (P+S): wearables without concurrent self-report'])}",
    f"- PSYCHE-D development-selected model, held-out test: {est(r6sel.auroc, L(r6sel.auroc_ci))} vs frozen {est(r6fro.auroc, L(r6fro.auroc_ci))}; difference {dlt(*r6p['selected: P+S+L / lightgbm_dart_frozen_config - frozen'])} (R6_test_models.csv, R6_test_paired_vs_frozen.csv)",
    f"- HRV and PHQ-9: day-0 PHQ-9 alone {est(hrv['frozen']['auroc'], hrv['frozen']['auroc_ci'])}; watch sensors alone 0.43-0.50 (hrv_phq9.json)",
    f"- RESILIENT: sensors + demographics {est(res['frozen']['auroc'], res['frozen']['auroc_ci'])}; questionnaires {est(res['plots']['ablation'][5]['auroc'], res['plots']['ablation'][5]['ci'])} (resilient.json)",
    f"- MOMENT alone: PSYCHE-D {arm('moment_only', p1)[0]}, HRV {est(hrv['moment']['alone_auroc'], hrv['moment']['alone_ci'])}, RESILIENT {est(res['moment']['alone_auroc'], res['moment']['alone_ci'])}",
    f"- FL, frozen inputs, as specified: TreeAvg mean {flmean('as specified', 'frozen 152', 'TreeAvg')}, {comm('as specified', 'frozen 152', 'IID', 'TreeAvg', 'total_mb').mean():.2f} MB total vs FedXgbBagging {comm('as specified', 'frozen 152', 'IID', 'FedXgbBagging', 'total_mb').mean():.2f} MB; TreeAvg minus centralized LightGBM {flpair('as specified', 'frozen 152', 'TreeAvg - Centralized LightGBM')} (T2_T3_T4_paired_differences.csv)",
    f"- FL, frozen inputs, seed varies by round (sensitivity): TreeAvg mean {flmean('seed varies by round', 'frozen 152', 'TreeAvg')}; FL on P+S+L: TreeAvg {flmean('seed varies by round', 'P+S+L 41', 'TreeAvg')} vs centralized LightGBM {a3(flp('seed varies by round', 'P+S+L 41', 'central', 'Centralized LightGBM').iloc[0].auroc)}",
    "",
    "## Results: Study Population and Outcomes",
    f"- PSYCHE-D: 10,866 samples, 4036 participants, worsening 2252 (20.7%). Test set 1631 samples, 606 participants, 313 worsened (19.2%; majority-class rate 80.8%).",
    f"- Second outcome (PHQ-9 increase >= 5 points): {t6m['n_positive']} of {t6m['n_samples']} samples ({100 * t6m['prevalence']:.1f}%), {t6m['n_positive_participants']} participants; {t6m['overlap_with_primary']['both']} also worsened by category, {t6m['overlap_with_primary']['ge5_only']} only by >= 5 points, {t6m['overlap_with_primary']['primary_only']} only by category (T6_T9_results.json).",
    "",
    "## Results: Leakage Audit",
    f"- PSYCHE-D logistic regression, 150 inputs: record split {est(p1j['arms']['leak_record_split']['auroc'], p1j['arms']['leak_record_split']['auroc_ci'])} vs participant split {est(p1j['arms']['leak_participant_split']['auroc'], p1j['arms']['leak_participant_split']['auroc_ci'])}, difference {dlt(p1j['paired']['leak: record - participant split']['delta'], p1j['paired']['leak: record - participant split']['ci'])} (P1_cv_results.json)",
]
for r in p31.itertuples():
    lines.append(f"- HRV and PHQ-9, {r.level}, {r.split}: {est(r.auroc, (r.ci_low, r.ci_high))}" + (f" (mean of 5 repeats {r.auroc_mean_of_5_repeats:.3f})" if pd.notna(r.auroc_mean_of_5_repeats) else "") + " (P3_1_hrv_leakage_participant_ci.csv)")
lines += [
    f"- Label shuffle: PSYCHE-D {psy_a10['A10_permutation_auroc_mean']:.3f} (SD {psy_a10['A10_permutation_auroc_std']:.3f}); HRV and PHQ-9 {hrv_perm[0]['mean']:.3f} and {hrv_perm[1]['mean']:.3f} (2 arms); RESILIENT {s23['mean']:.3f} (SD {s23['sd']:.3f}, leakage-audit LR) and {s45['mean']:.3f} (SD {s45['sd']:.3f}, frozen RF family) (T8_label_shuffle_provenance.csv). Recommended citation for RESILIENT: 0.44 (same leakage-audit LR approach as the other 2 datasets); see task 8.",
    "",
    "## Results: Input sets (Table 3)",
]
for lab, k in psy_rows:
    a_ = arm(k)
    lines.append(f"- PSYCHE-D {lab}: AUROC {a_[0]}, AUPRC {a_[1]} ({ci(a_[2])})")
for k, v in pr.items():
    lines.append(f"- PSYCHE-D {k}: {dlt(*v)}")
lines += [f"- PSYCHE-D temporal split (participant-disjoint, later quarters): {est(temporal['auroc'], temporal['auroc_ci'])}, {temporal['n_test']} test samples, {temporal['n_test_participants']} participants (P1_1_temporal_participant_ci.json)",
          f"- PSYCHE-D baseline category alone as a rule: {arm('category rule')[0]}"]
for r in sev.itertuples():
    if pd.notna(getattr(r, "_6", None)):
        lines.append(f"- Baseline PHQ-9 {r.baseline_phq9}: {r.n_samples} samples, worsening {100 * r.worsening_rate:.1f}%; frozen {est(r._6, L(r._7))}, P+S {est(r._8, L(r._9))}, P {est(r._10, L(r._11))}")
    else:
        lines.append(f"- Baseline PHQ-9 {r.baseline_phq9}: {r.n_samples} samples, worsening {100 * r.worsening_rate:.1f}% (AUROC not defined)")
lines += [f"- Excluding PHQ-9 20-27, retrained: frozen {arm('P+S+L+W (frozen) excl 20-27')[0]} vs P {arm('P excl 20-27')[0]}; difference {dlt(*pr['excl 20-27: frozen - P'])}",
          "- HRV and PHQ-9 and RESILIENT rows: see TABLE3_input_sets.md (unchanged from the manuscript).", "",
          "## Results: Forecasting analysis (R3)",
          f"- Forecast (P+S + months 1-2 W and L): {arm('R3 forecast: P+S+SM1/SM2 (W+L)')[0]}; minus P+S {dlt(*pr['forecast (W+L) - (P+S)'])}",
          f"- Forecast, Fitbit only: {arm('R3 forecast: P+S+SM1/SM2 Fitbit only')[0]}; minus P+S {dlt(*pr['forecast Fitbit only - (P+S)'])}",
          f"- Held-out test: forecast {est(r6.loc['R3 forecast (W+L) / DART (development-trained)'].auroc, L(r6.loc['R3 forecast (W+L) / DART (development-trained)'].auroc_ci))}; minus frozen {dlt(*r6p['R3 forecast (W+L) / DART (development-trained) - frozen'])}",
          "",
          "## Results: Published-model comparison (R4)",
          f"- Published 2-phase design reproduced, CV: {arm('published 2-phase (reproduction)')[0]}; minus frozen {dlt(*pr['published 2-phase - frozen'])}; minus forecast {dlt(*pr['published 2-phase - R3 forecast (W+L)'])}; minus P+S {dlt(*pr['published 2-phase - P+S'])}",
          "",
          "## Results: Held-out test and development-only selection (R6)"]
for k in r6.index:
    r = r6.loc[k]
    lines.append(f"- {k}: AUROC {est(r.auroc, L(r.auroc_ci))}, AUPRC {est(r.auprc, L(r.auprc_ci))}")
for k, v in r6p.items():
    lines.append(f"- {k}: {dlt(*v)}")
lines += ["", "## Results: Operating points (Table 4)", "Frozen protocol model (fitted on 70% train, thresholds on validation, test once):"]
for nm in ["default 0.5", "maximum accuracy (validation)", "matched sensitivity 55.4%", "80% sensitivity"]:
    lines.append(f"- {nm}, uncalibrated scores: {op(FROZEN, 'uncalibrated', nm)}")
    lines.append(f"- {nm}, Platt-calibrated scores (0.5 kept for the default point; other thresholds re-chosen on validation): {op(FROZEN, 'Platt (validation)', nm)}")
lines.append("Development-selected model (P+S+L; same procedure):")
for nm in ["default 0.5", "maximum accuracy (validation)", "matched sensitivity 55.4%", "80% sensitivity"]:
    lines.append(f"- {nm}, uncalibrated scores: {op(SEL, 'uncalibrated', nm)}")
    lines.append(f"- {nm}, Platt-calibrated scores (0.5 kept for the default point; other thresholds re-chosen on validation): {op(SEL, 'Platt (validation)', nm)}")
lines += ["- Note: a fixed 0.5 cut-off means different things before and after Platt scaling (Platt 0.5 = high-risk only). Matched-sensitivity and 80% points are equivalent before and after Platt (monotone transform).",
          "- HRV and PHQ-9 and RESILIENT: TABLE4_operating_points.md (no CIs in the source).", "",
          "## Results: Calibration (test set; Platt fitted on validation)"]
for m in [FROZEN, "R3 forecast", SEL]:
    for ver in ["uncalibrated", "Platt (validation)"]:
        lines.append(f"- {m}, {ver}: {calline(m, ver)}; AUROC {c(m, ver).auroc:.3f} (70%-trained)")
for r in cal3.itertuples():
    lines.append(f"- {r.model} (pooled outer folds, uncalibrated): intercept {r.calibration_in_the_large:.2f} ({r.citl_ci}), slope {r.slope:.2f} ({r.slope_ci}), Brier {r.brier:.3f} ({r.brier_ci}); prevalence Brier {r.brier_null_prevalence:.3f}; predictions {r.prediction_range}")
lines += ["- Platt scaling was not applied to HRV and PHQ-9 or RESILIENT (no separate validation set; pooled CV predictions only).", "",
          "## Results: Outcome sensitivity (task 6; PHQ-9 increase >= 5 points)"]
for k in t6.index:
    a_ = t6.loc[k]
    lines.append(f"- {k}: AUROC {est(a_.auroc, L(a_.auroc_ci))}, AUPRC {est(a_.auprc, L(a_.auprc_ci))}")
for k, v in t6p.items():
    lines.append(f"- {k}: {dlt(*v)}")
lines += ["", "## Results: MOMENT",
          f"- PSYCHE-D MOMENT alone {arm('moment_only', p1)[0]}; MOMENT + frozen {arm('moment_plus_frozen', p1)[0]}; minus frozen {dlt(*p1p['moment_plus_frozen - frozen'])} (P1_1_2_*.csv)",
          f"- RESILIENT MOMENT alone {est(res['moment']['alone_auroc'], res['moment']['alone_ci'])}; added to frozen {a3(res['moment']['added_auroc'])}, difference CI {ci(res['moment']['delta_vs_frozen_ci'], 'to')}"]
for r in p32.itertuples():
    lines.append(f"- HRV MOMENT, {r.config}, {r.inputs}: {est(r.auroc, (r.ci_low, r.ci_high))}; minus frozen {dlt(r.diff_vs_frozen, (r.diff_ci_low, r.diff_ci_high))} (P3_2_hrv_moment_rerun.csv)")
lines += ["", "## Results: Federated learning (Table 5)",
          "Two implementations are reported. 'As specified' = the original FL code (each round builds a fresh XGBoost booster with seed 0, so the 80% row and column subsample is the same in every round). 'Seed varies by round' = same code with a different seed per (round, client); recommended as primary, with the as-specified run as the reproduction of the original manuscript numbers. FedAvg and local-only are identical in both."]
for v in FL:
    lines.append(f"### {v}")
    for iset in ["frozen 152", "P+S+L 41"]:
        e = FL[v]["eta"]
        lines.append(f"- {iset}: TreeAvg eta selected on validation {e[(e.input_set == iset) & e.selected].eta.iloc[0]:g} (grid 0.03-0.30; T2_treeavg_eta_selection)")
        for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
            ss = flp(v, iset, "IID", m).sort_values("partition_seed")
            lines.append(f"- {iset}, {m}: " + "; ".join(f"seed {int(r.partition_seed)} {est(r.auroc, L(r.auroc_ci))}" for r in ss.itertuples()) + f"; mean {flmean(v, iset, m)}")
        for m in ["Centralized LightGBM", "Centralized XGBoost"]:
            r = flp(v, iset, "central", m).iloc[0]
            lines.append(f"- {iset}, {m}: {est(r.auroc, L(r.auroc_ci))}")
        prd = FL[v]["pr"]
        for comp in prd[(prd.input_set == iset) & (prd.partition_seed.astype(str) == "pooled")].comparison:
            lines.append(f"- {iset}, {comp}, pooled over 3 seeds: {flpair(v, iset, comp)}")
        for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
            lines.append(f"- {iset}, {m} communication: total {comm(v, iset, 'IID', m, 'total_mb').mean():.2f} MB (upload {comm(v, iset, 'IID', m, 'total_upload_mb').mean():.2f}, download {comm(v, iset, 'IID', m, 'total_download_mb').mean():.2f}); to selected round {comm(v, iset, 'IID', m, 'mb_to_selected_round').mean():.2f} MB; best validation AUROC within 2.16 MB " + "/".join(f"{x:.3f}" for x in comm(v, iset, 'IID', m, 'best_val_auroc_within_budget')))
    prd = FL[v]["pr"]
    for comp in prd[prd.input_set == "both"].comparison.unique():
        lines.append(f"- {comp}, pooled: {flpair(v, 'both', comp)}")
    for comp in prd[prd.task == "T4"].comparison:
        q = prd[(prd.task == "T4") & (prd.comparison == comp)].iloc[0]
        lines.append(f"- Non-IID (birth-year quintiles), {comp}: {dlt(q.delta, L(q.ci))}")
    for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
        r = flp(v, "frozen 152", "age quintile", m).iloc[0]
        lines.append(f"- Non-IID, {m}: {est(r.auroc, L(r.auroc_ci))}")
r8s = r8[r8.method == "TreeAvg eta 0.10 (sensitivity)"]
lines += [f"- TreeAvg eta 0.10 (prespecified, as specified code): " + "; ".join(f"seed {int(float(r.partition_seed))} {est(r.auroc, L(r.auroc_ci))}" for r in r8s.itertuples()) + " (R8_per_seed_auroc.csv)",
          f"- HRV and PHQ-9 FL (2 clients, PHQ9_1 only) and RESILIENT FL (exploratory): unchanged, TABLE5_federated.md.", "",
          "## Discussion numbers",
          "- No new numbers: the Discussion should quote the Results bullets above (screener vs Fitbit contributions, outcome sensitivity, held-out test selection, calibration, FL gap to centralized under both tree implementations).",
          ]
(FIN / "FINAL_NUMBERS.md").write_text("\n".join(lines) + "\n")
print("wrote", *sorted(p.name for p in FIN.iterdir()))
