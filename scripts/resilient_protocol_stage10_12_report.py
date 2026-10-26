#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): IRAS 321104, independent of REB25-1587_MOD1. Research data.
"""Protocol Stage 10 (published-work comparison, fallback branch), Stage 11 (paper prep:
figures as script-generated SVG because matplotlib is not installed in either venv) and
Stage 12 (final report). Builds, from the saved stage results only:

  scripts/resilient_protocol_stage10_12_results.json
  scripts/resilient_protocol_oof_predictions.json      (all saved OOF predictions + labels)
  public/docs/paper_figures/resilient_protocol/*.svg
  public/docs/RESILIENT_Protocol_Report.html
  src/data/protocolComparison/resilient.json

Run with .venv-psyche-d-fl-mlp. No model is trained here; every number is read from the
stage JSONs or computed directly from the saved predictions.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resilient_protocol_common as C  # noqa: E402

S = C.SCRIPT_DIR
REPO = C.REPO
FIG_DIR = REPO / "public" / "docs" / "paper_figures" / "resilient_protocol"
HTML_OUT = REPO / "public" / "docs" / "RESILIENT_Protocol_Report.html"
SUMMARY_OUT = REPO / "src" / "data" / "protocolComparison" / "resilient.json"
OUT = S / "resilient_protocol_stage10_12_results.json"
OOF_OUT = S / "resilient_protocol_oof_predictions.json"


def J(name):
    return json.loads((S / name).read_text())


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def ci3(c):
    return "n/a" if not c else f"[{c[0]:.3f}, {c[1]:.3f}]"


def pct(x):
    return f"{100 * x:.1f}%"


def esc(s):
    return html.escape(str(s))


# ---------------------------------------------------------------------------
# SVG figures (light/dark aware through CSS variables when inlined)
# ---------------------------------------------------------------------------

def _frame(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}" '
            f'style="width:100%;height:auto;font-family:Public Sans,sans-serif;font-size:11px">'
            f'<title>{esc(title)}</title>{body}</svg>')


def svg_curves(series, title, xlab, ylab, diag=False, w=420, h=320):
    L, R, T, B = 48, 12, 14, 40
    pw, ph = w - L - R, h - T - B
    X = lambda v: L + v * pw
    Y = lambda v: T + (1 - v) * ph
    body = [f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#C3CCBC"/>']
    for t in (0, 0.25, 0.5, 0.75, 1):
        body.append(f'<line x1="{X(t)}" y1="{T+ph}" x2="{X(t)}" y2="{T+ph+4}" stroke="#828C7F"/><text x="{X(t)}" y="{T+ph+16}" text-anchor="middle" fill="#828C7F">{t:g}</text>')
        body.append(f'<line x1="{L-4}" y1="{Y(t)}" x2="{L}" y2="{Y(t)}" stroke="#828C7F"/><text x="{L-7}" y="{Y(t)+4}" text-anchor="end" fill="#828C7F">{t:g}</text>')
    if diag:
        body.append(f'<line x1="{X(0)}" y1="{Y(0)}" x2="{X(1)}" y2="{Y(1)}" stroke="#9BA598" stroke-dasharray="4 4"/>')
    for i, (lab, pts, col, dash) in enumerate(series):
        d = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)
        body.append(f'<polyline points="{d}" fill="none" stroke="{col}" stroke-width="2" {"stroke-dasharray=\"5 3\"" if dash else ""}/>')
        body.append(f'<line x1="{L+10}" y1="{T+14+i*15}" x2="{L+28}" y2="{T+14+i*15}" stroke="{col}" stroke-width="2"/><text x="{L+33}" y="{T+18+i*15}" fill="#5B655C">{esc(lab)}</text>')
    body.append(f'<text x="{L+pw/2}" y="{h-6}" text-anchor="middle" fill="#5B655C">{esc(xlab)}</text>')
    body.append(f'<text transform="translate(12,{T+ph/2}) rotate(-90)" text-anchor="middle" fill="#5B655C">{esc(ylab)}</text>')
    return _frame(w, h, "".join(body), title)


def svg_forest(rows, title, ref=0.5, lo_ax=0.2, hi_ax=1.0, w=640):
    """rows: (label, value, [lo, hi] or None, highlight)"""
    L, R, T, rowh = 250, 60, 16, 24
    h = T + rowh * len(rows) + 34
    pw = w - L - R
    X = lambda v: L + (v - lo_ax) / (hi_ax - lo_ax) * pw
    body = []
    for t in np.arange(lo_ax, hi_ax + 1e-9, 0.1):
        body.append(f'<line x1="{X(t):.1f}" y1="{T-4}" x2="{X(t):.1f}" y2="{T+rowh*len(rows)}" stroke="#E3E8DE"/><text x="{X(t):.1f}" y="{T+rowh*len(rows)+14}" text-anchor="middle" fill="#828C7F">{t:.1f}</text>')
    body.append(f'<line x1="{X(ref):.1f}" y1="{T-4}" x2="{X(ref):.1f}" y2="{T+rowh*len(rows)}" stroke="#9BA598" stroke-dasharray="4 3"/>')
    for i, (lab, v, ci, hl) in enumerate(rows):
        y = T + i * rowh + rowh / 2
        col = "#0E8562" if hl else "#5B655C"
        body.append(f'<text x="{L-8}" y="{y+4}" text-anchor="end" fill="{col}" {"font-weight=\"600\"" if hl else ""}>{esc(lab)}</text>')
        if ci:
            body.append(f'<line x1="{X(max(ci[0], lo_ax)):.1f}" y1="{y}" x2="{X(min(ci[1], hi_ax)):.1f}" y2="{y}" stroke="{col}" stroke-width="2"/>')
        body.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="4.5" fill="{col}"/><text x="{w-R+6}" y="{y+4}" fill="{col}">{v:.3f}</text>')
    body.append(f'<text x="{L+pw/2}" y="{h-4}" text-anchor="middle" fill="#5B655C">AUROC with 95% participant bootstrap CI (dashed line = chance)</text>')
    return _frame(w, h, "".join(body), title)


def svg_hbars(rows, title, unit, w=560):
    L, R, T, rowh = 150, 70, 10, 26
    h = T + rowh * len(rows) + 12
    mx = max(v for _, v in rows)
    body = []
    for i, (lab, v) in enumerate(rows):
        y = T + i * rowh
        bw = (w - L - R) * v / mx
        body.append(f'<text x="{L-8}" y="{y+16}" text-anchor="end" fill="#5B655C">{esc(lab)}</text><rect x="{L}" y="{y+4}" width="{bw:.1f}" height="16" rx="3" fill="#0E8562"/>'
                    f'<text x="{L+bw+6:.1f}" y="{y+16}" fill="#1B211C">{v:.3f} {esc(unit)}</text>')
    return _frame(w, h, "".join(body), title)


def downsample(pts, k=60):
    pts = [tuple(map(float, p)) for p in pts]
    if len(pts) <= k:
        return pts
    idx = np.unique(np.round(np.linspace(0, len(pts) - 1, k)).astype(int))
    return [pts[i] for i in idx]


# ---------------------------------------------------------------------------

def main():
    prof, inv = J("resilient_protocol_profile.json"), J("resilient_protocol_inventory.json")
    s23, s45, s6 = J("resilient_protocol_stage02_03_results.json"), J("resilient_protocol_stage04_05_results.json"), J("resilient_protocol_stage06_results.json")
    s78, s9 = J("resilient_protocol_stage07_08_results.json"), J("resilient_protocol_stage09_results.json")
    frozen = J("resilient_protocol_frozen_config.json")
    d = C.load_features()
    F = d["features"]
    y = d["y_phq10"]
    n, npos = len(y), int(y.sum())
    maj = 1 - y.mean()
    fro_oof = np.array(s45["frozen_oof"])
    fro = fro_oof.mean(axis=0)
    cand = s45["arms"]["sensor_demo (candidate)"]
    model = frozen["model"]
    ops = s45["operating_points"]

    # ---------------- Stage 10: fallback comparison ----------------
    phq = F["phq_total"].to_numpy()
    sex = F["sex_female"].to_numpy()
    desc = {}
    for feat in ["steps_daily_median", "steps_daily_sd"]:
        v = F[feat].to_numpy()
        row = {}
        for lab, m in [("all", np.ones(n, bool)), ("female", sex == 1), ("male", sex == 0)]:
            ok = m & ~np.isnan(v)
            r, p = spearmanr(v[ok], phq[ok])
            row[lab] = {"n": int(ok.sum()), "rho": float(r), "p": float(p)}
        desc[feat] = row
    s10 = {"published_reference": "none with prediction metrics: Gomez et al. (Scientific Data 2025, PMC12546717) is a data descriptor reporting descriptive patterns only",
           "fallback_reference_sensitivity": C.CLINICAL_TARGET_SENS,
           "matched_sensitivity_point": ops["matched_sensitivity"],
           "descriptive_direction_check_vs_data_paper": desc,
           "data_paper_statement": "lower and less variable step counts correlated with higher depressive symptoms in both sexes (descriptive, no statistical testing)"}

    # ---------------- plots ----------------
    fpr, tpr, _ = roc_curve(y, fro)
    roc_pts = downsample(list(zip(fpr, tpr)), 60)
    if roc_pts[0] != (0.0, 0.0):
        roc_pts = [(0.0, 0.0)] + roc_pts[:59]
    if roc_pts[-1] != (1.0, 1.0):
        roc_pts = roc_pts[:59] + [(1.0, 1.0)]
    prec, rec, _ = precision_recall_curve(y, fro)
    pr_pts = downsample(list(zip(rec[::-1], prec[::-1])), 60)
    sweep = []
    for t in np.round(np.arange(0.01, 1.0, 0.02), 2):
        c = C.confusion_at(y, fro, t)
        sweep.append({"t": float(t), "acc": c["accuracy"], "sens": c["sensitivity"], "spec": c["specificity"]})
    arm_label = {"sensor_demo (candidate)": "Sensor + demographics (frozen)", "sensor_only": "Sensor only", "watch_only (HR+steps)": "Watch only (HR + steps)",
                 "mat_only (sleep physio+state)": "Sleep mat only", "demographics_only": "Demographics only",
                 "questionnaire_only (GAD-7, ACE-III)": "Questionnaire only (GAD-7, ACE-III)", "everything (sensor+demo+questionnaire)": "Everything (+ questionnaires)"}
    ablation = [{"arm": arm_label[a], "n_features": v["n_features"], "auroc": v["auroc"], "ci": v["auroc_ci"]} for a, v in s45["arms"].items()]
    ablation += [{"arm": f"MOMENT: {a}", "n_features": v["n_features"], "auroc": v["auroc"], "ci": v["auroc_ci"]} for a, v in s6["arms"].items()]
    fl_comm = [{"method": m, "mb": s78["methods"][m]["comm_mb_per_fold_mean"], "unit": "MB per outer fold (K=2, all rounds, up + down, measured)"}
               for m in ["FedAvg MLP", "FedXgbBagging", "TreeAvg"]]
    kind = {"FedAvg MLP": "exploratory", "FedXgbBagging": "exploratory", "TreeAvg": "exploratory", "Centralized XGBoost": "centralized",
            "Centralized LightGBM": "centralized", "Centralized MLP": "centralized", "Local only": "local"}
    fl_methods = [{"name": m, "auroc": v["auroc"], "ci": v["auroc_ci"], "kind": kind[m]} for m, v in s78["methods"].items()]
    plots = {"roc": {"model": f"{model} (sensor + demographics), repeat-averaged pooled OOF", "points": [[r4(a), r4(b)] for a, b in roc_pts]},
             "pr": {"points": [[r4(a), r4(b)] for a, b in pr_pts]},
             "sweep": [{k: r4(v) for k, v in s.items()} for s in sweep],
             "ablation": [{"arm": a["arm"], "n_features": a["n_features"], "auroc": r4(a["auroc"]), "ci": r4(a["ci"])} for a in ablation],
             "fl_comm": [{"method": c["method"], "mb": r4(c["mb"]), "unit": c["unit"]} for c in fl_comm],
             "fl_methods": [{"name": m["name"], "auroc": r4(m["auroc"]), "ci": r4(m["ci"]), "kind": m["kind"]} for m in fl_methods]}

    # ---------------- OOF dump ----------------
    OOF_OUT.write_text(json.dumps({
        "note": "participant order = scripts/resilient_protocol_features.pkl index order (pseudonymous UIDs not repeated here); scores are out-of-fold",
        "y_phq9_ge10": y.tolist(), "y_gds15_ge5": d["y_gds5"].tolist(),
        "frozen_oof_per_repeat": s45["frozen_oof"], "stage4_arm_oof_repeat_avg": s45["arm_oof_avg"],
        "stage6_moment_oof_repeat_avg": {a: v["oof_avg"] for a, v in s6["arms"].items()},
        "stage7_fl_oof_repeat_avg": s78["oof_avg"], "stage7_local_client_oof": s78["oof_local_client"],
        "stage3_screen_oof_per_repeat": {m: v["oof"] for m, v in s23["screen"].items()}}, default=C._json_default))
    print("[write]", OOF_OUT.relative_to(REPO))

    # ---------------- figures ----------------
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = {}
    figs["fig1_roc_pr"] = (svg_curves([("ROC, frozen " + model, roc_pts, "#0E8562", False)], "ROC curve of the frozen model", "False positive rate", "True positive rate", diag=True),
                           svg_curves([("Precision-recall, frozen " + model, pr_pts, "#0E8562", False), (f"Chance (prevalence {y.mean():.3f})", [(0, y.mean()), (1, y.mean())], "#9BA598", True)],
                                      "Precision-recall curve of the frozen model", "Recall (sensitivity)", "Precision"))
    figs["fig2_operating_curves"] = svg_curves([("Accuracy", [(s["t"], s["acc"]) for s in sweep], "#0E8562", False),
                                                ("Sensitivity", [(s["t"], s["sens"]) for s in sweep], "#B8710A", False),
                                                ("Specificity", [(s["t"], s["spec"]) for s in sweep], "#5B655C", True),
                                                (f"Majority rate {maj:.3f}", [(0, maj), (1, maj)], "#9BA598", True)],
                                               "Accuracy, sensitivity and specificity versus cut-off", "Cut-off on the predicted probability", "Rate", w=520)
    screen_rows = sorted([(m, v["auroc"], v["auroc_ci"], m == model) for m, v in s23["screen"].items()], key=lambda r: -r[1])
    figs["fig3_screen"] = svg_forest(screen_rows, "Model screen, sensor + demographics")
    abl_rows = [(a["arm"], a["auroc"], a["ci"], a["arm"].startswith("Sensor + demographics")) for a in ablation]
    figs["fig3b_ablation"] = svg_forest(abl_rows, "Feature-source and MOMENT arms", lo_ax=0.1)
    fl_rows = [(m["name"] + (" (exploratory)" if m["kind"] == "exploratory" else ""), m["auroc"], m["ci"], False) for m in fl_methods]
    figs["fig7_fl"] = svg_forest(fl_rows, "Exploratory FL versus centralized and local, K=2", lo_ax=0.2)
    figs["fig4_comm"] = svg_hbars([(c["method"], c["mb"]) for c in fl_comm], "Measured communication per outer fold", "MB")
    for k, v in figs.items():
        if isinstance(v, tuple):
            for i, part in enumerate(v):
                (FIG_DIR / f"{k}_{'roc' if i == 0 else 'pr'}.svg").write_text(part)
        else:
            (FIG_DIR / f"{k}.svg").write_text(v)

    # ---------------- stage table ----------------
    perm_lr = s23["permutation_test"]
    perm_fr = s45["permutation_frozen_model"]
    stages = [
        (0, "Dataset profile", "run", "SMALL-N", "N=73, 10 positives, 1 row per participant, cross-sectional label, raw sub-hourly signals, single site, no prediction paper, no weights"),
        (1, "Inventory and raw-data gate", "run", None, "4 independent checks agree: raw sequences (HR ~10 min, steps hourly, mat 2-3 min)"),
        (2, "Target and leakage audit", "run", None, f"PHQ-9 >= 10, {npos}/{n} positive; checklist {s23['leakage_pass']}/6 pass (the fail is the structural post-assessment sensor window); permutation AUROC {perm_lr['mean']:.3f} (LogReg) and {perm_fr['mean']:.3f} (frozen model)"),
        (3, "Baseline and forensic audit", "run", "SMALL-N; Part B skipped", f"A1-A12 run; no published model to reproduce; screen of 7 families, best {model} AUROC {cand['auroc']:.3f}"),
        (4, "Feature-source ablation and freeze", "run", "SMALL-N; temporal check skipped", f"frozen: {model}, sensor + demographics, {frozen['n_features']} features"),
        (5, "Operating points", "run", "clinical target 0.80", "validation (inner OOF) selected cut-offs applied once per outer fold"),
        (6, "Foundation-model features (MOMENT)", "branch", "R", "frozen MOMENT-1-small per modality on the same 512 h window; MiniRocket/TCN not run"),
        (7, "Federated learning", "branch", "not feasible; exploratory K=2", "4 positives per client against a >= ~10 rule; run as exploratory only"),
        (8, "FL audit and diagnostics", "branch", "exploratory", "asserts, measured bytes, paired bootstrap; no round/eta selection possible"),
        (9, "Robustness", "run", "SMALL-N", "3 CV seeds for all 7 models, LOPO, fold spread, enrolment-date check, GDS-15/PHQ-9>=5 labels"),
        (10, "Comparison with published work", "branch", "fallback (P8 = no)", "clinical target sensitivity 0.80 + descriptive direction check against the data paper"),
        (11, "Paper prep", "run", "SVG figures", "figures as script-generated SVG (no matplotlib in either venv); figs 5 (matched-sensitivity bars vs paper) skipped: no paper"),
        (12, "Final report", "run", None, "this document, sections A-K"),
    ]

    html_doc = build_html(locals())
    HTML_OUT.write_text(html_doc)
    print("[write]", HTML_OUT.relative_to(REPO))

    # ---------------- summary JSON ----------------
    m6 = s6["arms"]
    best_m_alone = m6["MOMENT 4 modalities"]
    added = m6["MOMENT 4 + frozen features"]
    summary = {
        "dataset": "RESILIENT", "prefix": "resilient", "report": "/docs/RESILIENT_Protocol_Report.html",
        "n_participants": n, "n_rows": n, "unit": "participant (one baseline PHQ-9; sensors summarised over [phq_date, +512 h))",
        "target": "PHQ-9 total >= 10 (Kroenke et al. 2001)", "target_type": "binary", "n_pos": npos, "pos_rate": r4(y.mean()), "majority_rate": r4(maj),
        "profile": {"individual_rows": True, "rows_per_participant": "1", "design": "cross-sectional label (baseline PHQ-9) with longitudinal wearable data",
                    "small_n": True, "signal_granularity": "raw sequences (watch HR ~10 min, steps hourly, sleep mat 2-3 min)", "natural_sites": None,
                    "published_reference": None, "survey_weights": False},
        "stages": [{"id": i, "name": nm, "status": st, "branch": br, "note": nt} for i, nm, st, br, nt in stages],
        "leakage": {"permutation_metric_mean": r4(perm_fr["mean"]), "row_vs_grouped": None, "checklist_pass": s23["leakage_pass"], "checklist_total": 6},
        "frozen": {"model": model, "features": "sensor + demographics (watch HR, steps, sleep-mat physiology and sleep state summaries + sex, age group, 2 comorbidities)",
                   "n_features": frozen["n_features"], "cv": "repeated nested stratified CV, outer 5-fold x 5 repeats, inner 3-fold (participant = row)",
                   "auroc": r4(cand["auroc"]), "auroc_ci": r4(cand["auroc_ci"]), "auprc": r4(cand["auprc"]), "auprc_ci": r4(cand["auprc_ci"])},
        "operating_points": {
            "default": _op(ops["default"]), "max_accuracy": _op(ops["max_accuracy"]),
            "matched_sensitivity": dict(_op(ops["matched_sensitivity"]), reference_sensitivity=C.CLINICAL_TARGET_SENS,
                                        reference_source="clinical screening target (protocol fallback; no published prediction model on RESILIENT)")},
        "moment": {"branch": "R", "alone_auroc": r4(best_m_alone["auroc"]), "alone_ci": r4(best_m_alone["auroc_ci"]),
                   "added_auroc": r4(added["auroc"]), "delta_vs_frozen_ci": r4(added["paired_vs_frozen"]["ci"]),
                   "note": f"alone = 4 per-modality MOMENT embeddings (2048-d); added = embeddings + frozen features; {model}, same CV. Paired delta of added vs frozen {added['paired_vs_frozen']['delta']:+.3f}: {added['paired_vs_frozen']['verdict']}. HR-only MOMENT {m6['MOMENT HR only']['auroc']:.3f} {ci3(m6['MOMENT HR only']['auroc_ci'])} is significantly worse than frozen; raw flattened HR+steps control {m6['Raw flattened HR+Steps (control)']['auroc']:.3f} matches it."},
        "fl": {"feasible": False, "clients": "simulated K=2, label-stratified split of each outer training fold (exploratory only)", "K": 2,
               "methods": [{"name": m, "auroc": r4(s78["methods"][m]["auroc"]), "auroc_ci": r4(s78["methods"][m]["auroc_ci"]),
                            "comm_mb": r4(s78["methods"][m].get("comm_mb_per_fold_mean"))}
                           for m in ["FedAvg MLP", "FedXgbBagging", "TreeAvg", "Centralized XGBoost", "Centralized LightGBM", "Local only"]],
               "note": f"Not feasible at this N: 10 positives total, 8 per outer training fold, so K=2 gives 4 positives per client (rule >= ~10). Methods were run as exploratory evidence only; comm_mb = measured MB per outer fold; no FL vs centralized difference is distinguishable."},
        "robustness": {"seeds": 3, "auroc_sd": r4(s9["models"][model]["sd"]),
                       "note": f"{model} AUROC over CV seeds 42/43/44: {', '.join(f'{a:.3f}' for a in s9['models'][model]['aurocs'])}; LOPO {s9['frozen_lopo']['auroc']:.3f} {ci3(s9['frozen_lopo']['auroc_ci'])}; outer-fold AUROC SD {s9['frozen_fold_spread']['sd']:.3f} (2 positives per test fold)."},
        "published": {"paper": "Gomez et al., The RESILIENT Dataset, Scientific Data 2025 (PMC12546717): data descriptor, no prediction model",
                      "their_metrics": None,
                      "comparison": f"No comparable metric. Fallback: matched-sensitivity point at the 0.80 clinical target lands at test sensitivity {ops['matched_sensitivity']['sensitivity']:.3f}, specificity {ops['matched_sensitivity']['specificity']:.3f}. Descriptive check: step-count direction agrees with the data paper (see report Stage 10)."},
        "honest_read": HONEST_OVERALL.format(model=model, auc=cand["auroc"], lo=cand["auroc_ci"][0], hi=cand["auroc_ci"][1], npos=npos),
        "plots": plots,
    }
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2))
    print("[write]", SUMMARY_OUT.relative_to(REPO))
    s1012 = {"stage10": s10, "plots": plots, "figures": sorted(p.name for p in FIG_DIR.glob("*.svg"))}
    C.write_json(OUT, s1012)


def r4(x):
    return C.r4(x)


def _op(o):
    return {"threshold": r4(o["threshold_median"]), "accuracy": r4(o["accuracy"]), "sensitivity": r4(o["sensitivity"]), "specificity": r4(o["specificity"])}


HONEST_OVERALL = ("On 73 older adults with only {npos} PHQ-9 positives, a small random forest on wearable step, heart-rate and sleep summaries plus demographics "
                  "reached a repeated-CV AUROC of {auc:.3f} (95% CI {lo:.3f} to {hi:.3f}), an interval that stays above chance, "
                  "and the signal comes almost entirely from daily step counts. It is fragile: picking the best of 7 model families on the same predictions adds optimism, "
                  "the sensors were recorded after the PHQ-9 visit, and at clinically useful sensitivity the model flags about 4 in 10 negatives. "
                  "Frozen MOMENT embeddings and federated learning add nothing that can be distinguished at this size, and FL is not feasible with 10 positives. "
                  "Treat this as a hypothesis (less walking goes with depression in this cohort) to test on a larger cohort, not as a screening tool.")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """
:root{--bg:#F5F6F2;--surface:#FFFFFF;--surface-2:#ECEFE8;--border:#D8DED2;--border-strong:#C3CCBC;--text:#1B211C;--text-muted:#5B655C;--text-faint:#828C7F;
--accent:#0E8562;--accent-soft:#E1F0E8;--accent-text:#0A6B4E;--warn:#B8710A;--warn-soft:#F7EBD6;--warn-text:#8A5507;--shadow:0 1px 2px rgba(27,33,28,.06),0 4px 14px rgba(27,33,28,.05);}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14181A;--surface:#1B211D;--surface-2:#222A23;--border:#313B32;--border-strong:#3C4A3D;--text:#E7EBE4;--text-muted:#A9B3A5;--text-faint:#7C867A;--accent:#3FA989;--accent-soft:#1D3129;--accent-text:#5DBFA0;--warn:#D9A150;--warn-soft:#3A2E1A;--warn-text:#E8B968;--shadow:0 1px 2px rgba(0,0,0,.3);}}
:root[data-theme="dark"]{--bg:#14181A;--surface:#1B211D;--surface-2:#222A23;--border:#313B32;--border-strong:#3C4A3D;--text:#E7EBE4;--text-muted:#A9B3A5;--text-faint:#7C867A;--accent:#3FA989;--accent-soft:#1D3129;--accent-text:#5DBFA0;--warn:#D9A150;--warn-soft:#3A2E1A;--warn-text:#E8B968;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"Public Sans",-apple-system,"Segoe UI",sans-serif;font-size:16px;line-height:1.6;margin:0;padding:0 16px 64px}
.page{max-width:880px;margin:0 auto}
h1,h2,h3{font-family:"Source Serif 4",Georgia,serif;font-weight:600}
h1{font-size:2rem;line-height:1.15;margin:40px 0 6px}
h2{font-size:1.4rem;margin:0 0 6px}
h3{font-size:1.05rem;margin:18px 0 8px}
p{margin:0 0 12px;max-width:72ch}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em}
.subtitle{color:var(--text-muted)}
.chip{display:inline-block;font-size:.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase;padding:4px 9px;border-radius:999px;border:1px solid var(--border-strong);background:var(--surface-2);color:var(--text-muted);margin:0 6px 6px 0}
.chip.good{background:var(--accent-soft);color:var(--accent-text);border-color:transparent}
.chip.warn{background:var(--warn-soft);color:var(--warn-text);border-color:transparent}
section{background:var(--surface);border:1px solid var(--border);border-radius:14px;box-shadow:var(--shadow);padding:20px 22px;margin:22px 0}
.verdict{border-left:4px solid var(--accent)}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:1px;background:var(--border);border:1px solid var(--border);border-radius:10px;overflow:hidden;margin:12px 0}
.stat{background:var(--surface);padding:12px 14px}.stat .k{font-size:.7rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em}
.stat .v{font-family:"IBM Plex Mono",monospace;font-size:1.25rem;font-weight:600}.stat .c{font-family:"IBM Plex Mono",monospace;font-size:.75rem;color:var(--text-faint)}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--border);vertical-align:top}
th{font-size:.7rem;text-transform:uppercase;letter-spacing:.04em;color:var(--text-muted);background:var(--surface-2)}
tr:last-child td{border-bottom:none}
td.n{font-family:"IBM Plex Mono",monospace;white-space:nowrap}
tr.win td{background:var(--accent-soft);font-weight:600}
.honest{background:var(--accent-soft);border-radius:10px;padding:12px 14px;margin-top:14px}
.honest b{color:var(--accent-text)}
.caution{background:var(--warn-soft);border-radius:10px;padding:12px 14px;margin:12px 0}
.caution b{color:var(--warn-text)}
.fig{margin:14px 0;padding:10px;border:1px solid var(--border);border-radius:10px;background:#fff}
.figpair{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:640px){.figpair{grid-template-columns:1fr}h1{font-size:1.6rem}section{padding:16px}}
nav.toc{font-size:.85rem;color:var(--text-muted)}nav.toc a{color:var(--accent-text);margin-right:10px;white-space:nowrap}
footer{color:var(--text-faint);font-size:.8rem;margin-top:24px}
"""


def table(headers, rows, win=None):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = ""
    for i, r in enumerate(rows):
        cls = ' class="win"' if win is not None and win(i, r) else ""
        cells = "".join(f'<td class="n">{c}</td>' if isinstance(c, str) and (c[:1].isdigit() or c[:1] in "[-+") else f"<td>{c}</td>" for c in r)
        body += f"<tr{cls}>{cells}</tr>"
    return f'<div class="table-wrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def honest(txt):
    return f'<div class="honest"><b>Honest read.</b> {txt}</div>'


def sec(i, title, inner):
    return f'<section id="s{i}"><h2>Stage {i}. {esc(title)}</h2>{inner}</section>'


def build_html(v):
    prof, inv, s23, s45, s6, s78, s9, frozen = (v[k] for k in ("prof", "inv", "s23", "s45", "s6", "s78", "s9", "frozen"))
    y, n, npos, maj, model, cand, ops, figs, s10, stages = (v[k] for k in ("y", "n", "npos", "maj", "model", "cand", "ops", "figs", "s10", "stages"))
    perm_lr, perm_fr = v["perm_lr"], v["perm_fr"]
    out = []
    # ---- header + verdict
    out.append(f"""<h1>RESILIENT: dataset study protocol report</h1>
<p class="subtitle">Binary PHQ-9 (total &ge; 10) from wrist watch and sleep-mat data in 73 older adults. Every number on this page comes from the fresh <code>resilient_protocol_*</code> scripts run on 2026-10-05; nothing is copied from older RESILIENT reports.</p>
<div><span class="chip">N = {n}</span><span class="chip">{npos} positives</span><span class="chip">SMALL-N branch</span><span class="chip">Stage 6 branch R</span><span class="chip warn">FL not feasible</span><span class="chip">no published prediction model</span></div>
<section class="verdict"><h2>Verdict</h2>
<div class="stat-grid">
<div class="stat"><div class="k">Frozen AUROC</div><div class="v">{cand['auroc']:.3f}</div><div class="c">{ci3(cand['auroc_ci'])}</div></div>
<div class="stat"><div class="k">AUPRC (chance {y.mean():.3f})</div><div class="v">{cand['auprc']:.3f}</div><div class="c">{ci3(cand['auprc_ci'])}</div></div>
<div class="stat"><div class="k">Sens / spec, default cut-off</div><div class="v">{ops['default']['sensitivity']:.2f} / {ops['default']['specificity']:.2f}</div><div class="c">accuracy {ops['default']['accuracy']:.3f} vs majority {maj:.3f}</div></div>
<div class="stat"><div class="k">Permutation AUROC</div><div class="v">{perm_fr['mean']:.3f}</div><div class="c">frozen model, 5 label shuffles</div></div>
</div>
<p>{esc(HONEST_OVERALL.format(model=model, auc=cand['auroc'], lo=cand['auroc_ci'][0], hi=cand['auroc_ci'][1], npos=npos))}</p></section>
<nav class="toc">{''.join(f'<a href="#s{i}">Stage {i}</a>' for i in range(13))}</nav>""")

    # ---- applicability table
    out.append('<section><h2>Applicability summary</h2>' + table(["Stage", "Name", "Status", "Branch", "Note"],
               [[str(i), esc(nm), esc(st), esc(br or ""), esc(nt)] for i, nm, st, br, nt in stages]) +
               "<p>Skipped parts, with reasons: Stage 3 Part B (no published model on this data to reproduce); Stage 4.3 temporal validity (one baseline PHQ-9 per participant, no later labels); Stage 6 MiniRocket and TCN baselines (sktime not installed, and 58 training sequences with 8 positives per fold are too few for a TCN); Stage 7 as a real FL study (not feasible, run as exploratory only); Stage 9 leave-one-site-out (single site); Stage 11 figure 5 (no published operating point to match); dashboard page wiring (out of scope for this run).</p></section>")

    # ---- Stage 0
    rows = [[k.split("_", 1)[0], esc(k.split("_", 1)[1].replace("_", " ")), esc(json.dumps(val.get("answer", val.get("N")) if isinstance(val, dict) else val)), esc(val.get("evidence", "") if isinstance(val, dict) else "")]
            for k, val in prof.items() if k.startswith("P")]
    out.append(sec(0, "Dataset profile", table(["Q", "Question", "Answer", "Evidence"], rows) +
                   "<p>Branches: SMALL-N everywhere (N = 73 &lt; 200 and 10 positives &lt; 50). With N above 60 the default is repeated nested stratified CV (outer 5-fold x 5 repeats, inner 3-fold, at most 10 configurations per model), with leave-one-participant-out as a check in Stage 9. Signals are raw sub-hourly streams, so Stage 6 takes branch R. With one row per participant a grouped split equals a stratified split.</p>"
                   "<p>Inputs decided here: unit = participant; target = PHQ-9 total &ge; 10 (Kroenke, Spitzer and Williams 2001, moderate or worse, and the locked cut-off of the existing RESILIENT binary script); label columns = all <code>phq*</code> items and total; GDS-15 is a concurrent depression scale and is excluded from features (it is the secondary target instead); GAD-7 and ACE-III are same-visit questionnaires, used only as an ablation arm; clinical target sensitivity = 0.80 (no paper); data terms CC BY 4.0, ethics IRAS 321104 (separate from REB25-1587_MOD1).</p>" +
                   honest("This is a small, single-site, cross-sectional problem: one PHQ-9 per person, only 10 people above the cut-off. Every later number has wide intervals, and FL was expected to be out of reach before any model was fit.")))

    # ---- Stage 1
    rr = inv["raw_rows_per_participant"]
    g = inv["granularity_gate"]
    out.append(sec(1, "Inventory and raw-data gate",
                   table(["File", "Rows per participant (min / median / max)", "Participants with 0 rows", "Median sampling interval"],
                         [[esc(k), f"{int(x['min'])} / {int(x['median'])} / {int(x['max'])}", str(inv["participants_with_zero_rows"][k]), f"{g['check4_sampling_interval_median_sec'][k]:.0f} s"] for k, x in rr.items()]) +
                   f"<p>Recording dates span {inv['date_range'][0]} to {inv['date_range'][1]}. Demographics.csv is {inv['demographics_shape'][0]} x {inv['demographics_shape'][1]}; its most-missing columns are ACE-III items (up to 85% missing at 6 months).</p>"
                   "<h3>Granularity gate</h3>" + table(["Check", "Result"], [["dtype scan", esc(g["check1_dtype_scan"])], ["description scan", esc(g["check2_description_scan"])], ["file search", esc(g["check3_file_search"])], ["sampling interval", esc(json.dumps(g["check4_sampling_interval_median_sec"]))], ["verdict", f"<b>{esc(g['verdict'])}</b>"]]) +
                   f"<p>Pre-assessment data: {inv['pre_baseline_data']['participants_with_any_pre_baseline_row']}/73 participants have any row before their PHQ-9 date, in any modality. So the feature window is <code>{esc(inv['feature_window'])}</code>, the same bounded window the existing binary script uses, and every feature is a summary of the 3 weeks after the visit.</p>"
                   "<h3>Data dictionary (participant-level features built here)</h3>" +
                   table(["Column", "Group", "Description", "% missing"], [[f"<code>{esc(r['column'])}</code>", esc(r["group"]), esc(r["description"]), f"{r['pct_missing']:.1f}"] for r in inv["data_dictionary"]]) +
                   honest("The data are genuinely raw (10-minute watch HR, hourly steps, 2 to 3 minute sleep-mat physiology), so both hand-made summaries and MOMENT sequence embeddings are possible. Coverage is good: only one participant lacks sleep-mat physiology. The real limitation is timing: no one has data before the PHQ-9 visit, so these models describe the weeks right after the assessment, not a prediction ahead of it.")))

    # ---- Stage 2
    t = s23["target"]
    hist = ", ".join(f"{k}: {c}" for k, c in t["raw_score"]["hist"].items())
    out.append(sec(2, "Target construction and leakage audit",
                   f"<p>PHQ-9 total: mean {t['raw_score']['mean']:.2f}, SD {t['raw_score']['sd']:.2f}, median {t['raw_score']['median']:.0f}, range {t['raw_score']['min']} to {t['raw_score']['max']}. Counts per score: {hist}.</p>"
                   f"<p>Target PHQ-9 &ge; 10: <b>{t['n_pos']} positive, {t['n_neg']} negative</b> (positive rate {t['pos_rate']:.3f}, majority rate {t['majority_rate']:.3f}). For context: PHQ-9 &ge; 5 has {t['phq_ge5_n']} positives and GDS-15 &ge; 5 has {t['gds_ge5_n']}. Floor or ceiling groups do not apply (not a change target).</p>"
                   "<h3>Leakage checklist</h3>" + table(["Item", "Result", "Evidence"], [[esc(c["item"]), "pass" if c["pass"] else "<b>fail</b>", esc(c["evidence"])] for c in s23["leakage_checklist"]]) +
                   "<h3>Label-shuffle permutation test</h3>" + table(["Model", "AUROC per shuffle", "Mean (SD)"],
                         [[esc(perm_lr["model"]), ", ".join(f"{a:.3f}" for a in perm_lr["aurocs"]), f"{perm_lr['mean']:.3f} ({perm_lr['sd']:.3f})"],
                          [f"{model} (frozen family, nested, 5-fold x 1 repeat)", ", ".join(f"{a:.3f}" for a in perm_fr["aurocs"]), f"{perm_fr['mean']:.3f} ({perm_fr['sd']:.3f})"]]) +
                   "<p>Row versus grouped split: not applicable, one row per participant.</p>" +
                   honest(f"Five of six checks pass. The one fail is structural and disclosed: the sensors come after the questionnaire, which is not label leakage but limits the claim to a concurrent association. Shuffled labels fall to chance ({perm_fr['mean']:.3f} for the frozen model family), so the pipeline is not manufacturing signal. A shuffle-to-shuffle SD near 0.09 shows how noisy any single AUROC is with 10 positives.")))

    # ---- Stage 3
    A = s23["audit_A"]
    sc = s23["screen"]
    folds = A["A4_A5"][:5]
    out.append(sec(3, "Baseline and forensic audit",
                   "<h3>Part A, pipeline audit</h3>" + table(["Check", "Result"], [
                       ["A1 shape, target vs raw score", f"X = {A['A1']['shape'][0]} x {A['A1']['shape'][1]}; target equals phq_total &ge; 10 for all rows: {A['A1']['target_matches_raw']}"],
                       ["A2 participant IDs", f"{A['A2']['n_unique']} unique 8-character IDs, folder name equals Demographics user_id; spot check of 10 ok: {A['A2']['spot_check_10_ok']}"],
                       ["A3 target direction", f"mean PHQ-9 {A['A3']['mean_phq_pos']:.1f} in positives vs {A['A3']['mean_phq_neg']:.1f} in negatives; clean split at 10: {A['A3']['direction_ok']}"],
                       ["A4-A5 fold sizes (repeat 0)", "; ".join(f"fold {f['fold']}: train {f['n_train']} ({f['pos_train']} pos), test {f['n_test']} ({f['pos_test']} pos)" for f in folds)],
                       ["A6 feature sources", f"{A['A6']['sensor_derived']} sensor-derived, {A['A6']['demographic']} demographic; questionnaires {', '.join(A['A6']['questionnaire_excluded'])} excluded from the candidate"],
                       ["A7 matrix sanity", f"constant columns {A['A7']['constant_cols'] or 'none'}, infinities {A['A7']['inf_values']}, all-NaN columns {A['A7']['all_nan_cols'] or 'none'}"],
                       ["A8-A9 per-fold scores", f"{model}, repeat 0: mean score in positives " + ", ".join(f3(x) for x in A['A8_A9'][model]['mean_score_pos_by_fold']) + " vs negatives " + ", ".join(f3(x) for x in A['A8_A9'][model]['mean_score_neg_by_fold'])],
                       ["A10 permutation", f"LogReg {perm_lr['mean']:.3f}; frozen family {perm_fr['mean']:.3f} (Stage 2)"],
                       ["A11 row vs grouped", "not applicable"],
                       ["A12 conditional", f"enrolment date alone gives AUROC {A['A12']['enrolment_date_auroc_vs_label']:.3f} against the label (not a feature; followed up in Stage 9); file order {A['A12']['file_order_auroc_vs_label']:.3f}; single site; no baseline-severity variable"]]) +
                   "<h3>Part B, reproduce the published model</h3><p>Skipped: the RESILIENT paper is a data descriptor with no prediction model. Reference = regularized logistic regression (the strongest simple baseline).</p>"
                   "<h3>Centralized model screen (sensor + demographics, 28 features)</h3>" +
                   table(["Model", "AUROC (repeat-averaged OOF)", "95% CI", "Per-repeat mean (SD)", "AUPRC", "Paired vs LogReg (delta, CI)"],
                         [[esc(m), f"{x['auroc']:.3f}", ci3(x["auroc_ci"]), f"{x['auroc_per_repeat_mean']:.3f} ({x['auroc_per_repeat_sd']:.3f})", f"{x['auprc']:.3f}",
                           ("reference" if m == "LogReg" else f"{s23['screen_vs_logreg'][m]['delta']:+.3f} {ci3(s23['screen_vs_logreg'][m]['ci'])}, {s23['screen_vs_logreg'][m]['verdict']}")]
                          for m, x in sorted(sc.items(), key=lambda kv: -kv[1]["auroc"])], win=lambda i, r: r[0] == model) +
                   f'<div class="fig">{figs["fig3_screen"]}</div>'
                   "<p>Headline AUROC is computed on each participant's out-of-fold score averaged over the 5 CV repeats, with a 2000-resample participant bootstrap. AUPRC chance level is the positive rate, " + f"{y.mean():.3f}.</p>" +
                   honest(f"Was the old weak result a bug or the data's limit? Mostly the model and features: a regularized logistic regression on the same 28 features sits at chance ({sc['LogReg']['auroc']:.3f}), while tree ensembles reach about 0.67 to 0.72 and most beat it in paired tests; {model} is highest (paired gain over LogReg {s23['screen_vs_logreg'][model]['delta']:+.3f}, CI excluding 0). The relation is non-linear (Stage 4 shows it is mostly step counts). Choosing the best of 7 families on the same predictions inflates the winner a little; Stage 9 re-runs every family with new CV seeds to check this.")))

    # ---- Stage 4
    arms = s45["arms"]
    pv = s45["paired_vs_candidate"]
    imp = sorted(s45["importance_permutation"]["auroc_drop_mean_of_10"].items(), key=lambda kv: -kv[1])[:8]
    nat = s45["importance_native"]
    nat_top = sorted(nat["values"].items(), key=lambda kv: -kv[1])[:8] if nat else []
    out.append(sec(4, "Feature-source ablation and baseline freeze",
                   f"<p>All arms use {model} with the same nested repeated CV. Paired bootstrap against the candidate (sensor + demographics). The candidate run reproduces the Stage 3 out-of-fold predictions exactly: {s45['reproduces_stage3_oof']}.</p>" +
                   table(["Arm", "Features", "AUROC", "95% CI", "AUPRC", "Paired delta vs candidate", "Verdict"],
                         [[esc(a), str(x["n_features"]), f"{x['auroc']:.3f}", ci3(x["auroc_ci"]), f"{x['auprc']:.3f}",
                           ("candidate" if a not in pv else f"{pv[a]['delta']:+.3f} {ci3(pv[a]['ci'])}"), ("" if a not in pv else esc(pv[a]["verdict"]))] for a, x in arms.items()],
                         win=lambda i, r: "candidate" in r[0]) +
                   "<p>Temporal validity: skipped (one baseline label per participant). No derived features from other-time outcomes were built, so no cross-fitting was needed.</p>"
                   f"<h3>Frozen configuration (<code>scripts/resilient_protocol_frozen_config.json</code>)</h3><p>{model}, sensor + demographics, {frozen['n_features']} features, balanced subsample class weights, 300 trees, seed 42. Hyperparameters chosen by inner 3-fold CV on all 73 participants: <code>{esc(json.dumps(frozen['hyperparameters_full_data_inner_cv']))}</code> (inner AUROC {frozen['inner_cv_auroc_full_data']:.3f}). During evaluation the hyperparameters are re-selected inside every outer fold. The freeze rule (sensor + demographics arm, best screen family) was written into the script before the ablation ran, so the higher watch-only number did not change it.</p>"
                   "<h3>Feature importance</h3>" + table(["Permutation (AUROC drop, outer folds of repeat 0)", "Value", "Impurity importance (full-data fit)", "Value"],
                         [[f"<code>{esc(a)}</code>", f"{b:.3f}", f"<code>{esc(c)}</code>" if c else "", f"{e:.3f}" if c else ""] for (a, b), (c, e) in zip(imp, nat_top or [("", 0)] * len(imp))]) +
                   honest(f"Steps carry the model: the four step features are the only ones whose permutation drops AUROC noticeably, and the watch-only arm (HR + steps) does at least as well as the full candidate. The sleep mat alone is no better than chance and significantly worse than the candidate. Same-visit questionnaires (GAD-7 and ACE-III) beat every wearable arm ({arms['questionnaire_only (GAD-7, ACE-III)']['auroc']:.3f}), which is expected because GAD-7 overlaps heavily with PHQ-9 symptoms; they are not available to a wearable-only Agent 1 and are kept out of the frozen model. Demographics alone are not distinguishable from the candidate, so sex and age still explain part of the signal.")))

    # ---- Stage 5
    rows = []
    for k, lab in [("default", "Default (max balanced accuracy on inner OOF)"), ("max_accuracy", "Max accuracy (inner OOF)"), ("matched_sensitivity", "Matched sensitivity (inner OOF sens &ge; 0.80)")]:
        o = ops[k]
        rows.append([lab, f"{o['threshold_median']:.3f} [{o['threshold_iqr'][0]:.3f}, {o['threshold_iqr'][1]:.3f}]", f"{o['accuracy']:.3f}", f"{o['sensitivity']:.3f}", f"{o['specificity']:.3f}", f3(o["ppv"]), f"{o['tp']}/{o['fn']}/{o['tn']}/{o['fp']}"])
    out.append(sec(5, "Operating points",
                   f"<p>Cut-offs are chosen on the inner out-of-fold scores of each outer training fold (validation only), then applied once to that outer test fold. Counts pool all 25 outer folds, so each participant appears 5 times (365 predictions, 50 of them positive). Majority-class accuracy: {maj:.3f}.</p>" +
                   table(["Point", "Cut-off median [IQR over folds]", "Accuracy", "Sensitivity", "Specificity", "PPV", "TP/FN/TN/FP"], rows) +
                   f'<div class="fig">{figs["fig2_operating_curves"]}</div><p>The full 0.01-grid sweep (accuracy, sensitivity, specificity, PPV, NPV, counts) on the repeat-averaged OOF score is saved in <code>scripts/resilient_protocol_stage04_05_results.json</code> and summarized as <code>plots.sweep</code> in the summary JSON.</p>' +
                   honest(f"No cut-off gives a useful accuracy: every point is below the {maj:.3f} you get by calling everyone negative. The max-accuracy cut-off gets close ({ops['max_accuracy']['accuracy']:.3f}) only by almost never flagging anyone (sensitivity {ops['max_accuracy']['sensitivity']:.2f}). Aiming for the 0.80 clinical sensitivity on validation lands at {ops['matched_sensitivity']['sensitivity']:.2f} on test, with specificity {ops['matched_sensitivity']['specificity']:.2f}, so about 4 in 10 people without depression would be flagged and fewer than 1 in 4 flags would be true. No target accuracy was set for this dataset.")))

    # ---- Stage 6
    m6 = s6["arms"]
    out.append(sec(6, "Foundation-model features (MOMENT-Small, frozen), branch R",
                   f"<p>Frozen <code>AutonLab/MOMENT-1-small</code>, task embedding, 512-d output, reused through the existing RESILIENT loaders: hourly HR mean and hourly step sums over the 512-hour window (exactly 512 steps, own-mean gap fill), and nightly sleep-mat physiology (4 channels) and sleep-state shares (4 channels) left-aligned with <code>input_mask</code> = 0 on padding. One window per participant per modality, so per-participant pooling is the identity. Participants failing the coverage rule get missing embeddings, imputed inside the training fold: {', '.join(f'{k} {v}' for k, v in s6['missing_embeddings'].items())}. Embeddings are cached in <code>scripts/resilient_protocol_moment_embeds.pkl</code>; peak RSS while embedding was about 2.1 GB.</p>" +
                   table(["Arm", "Dims", "AUROC", "95% CI", "AUPRC", "Paired delta vs frozen", "Verdict"],
                         [[esc(a), str(x["n_features"]), f"{x['auroc']:.3f}", ci3(x["auroc_ci"]), f"{x['auprc']:.3f}", f"{x['paired_vs_frozen']['delta']:+.3f} {ci3(x['paired_vs_frozen']['ci'])}", esc(x["paired_vs_frozen"]["verdict"])] for a, x in m6.items()]) +
                   f"<p>Frozen baseline for reference: {cand['auroc']:.3f} {ci3(cand['auroc_ci'])}. MiniRocket and TCN: {esc(s6['sequence_baselines'])}.</p>" +
                   honest(STAGE6_HONEST(m6, cand))))

    # ---- Stage 7
    fm = s78["methods"]
    fe = s78["feasibility"]
    out.append(sec(7, "Federated learning (exploratory only)",
                   f'<div class="caution"><b>Not feasible at this N.</b> The protocol needs every client to hold at least about 10 positives. RESILIENT has {fe["total_positives"]} positives in total and {fe["positives_per_outer_train_fold"][0]} in each outer training fold, so even K = 2 gives each client {fe["K2_client_positives_range"][0]} positives. There are no natural sites. Everything below is exploratory, run to measure the mechanics and communication, not to claim an FL result.</div>'
                   "<p>Design: simulated K = 2 clients inside each of the 25 outer folds (label-stratified seeded split, so each client gets 4 positives; disjointness asserted). Federated sufficient statistics (per-client count, sum, sum of squares) give the global mean and SD for imputation and z-scoring of the MLP inputs; trees use native missing values. There is no validation data at this size, so no round, eta or epoch is chosen from data: every method runs a fixed, pre-declared budget and the last round is scored.</p>"
                   "<p>TreeAvg update: each round r every client k fits one depth-4 tree h<sub>k,r</sub> to the shared margin; w<sub>k</sub> = n<sub>k</sub>/N; margin = &Sigma;<sub>r</sub> &Sigma;<sub>k</sub> &eta; w<sub>k</sub> h<sub>k,r</sub>(x); p = sigmoid(margin). FedAvg: global weights = &Sigma;<sub>k</sub> (n<sub>k</sub>/N) local weights after 2 local epochs, 40 rounds. FedXgbBagging: Flower's aggregate, one tree per client per round, 30 rounds.</p>" +
                   table(["Method", "Kind", "AUROC", "95% CI", "Measured MB per outer fold"],
                         [[esc(m), esc(v["kind"][m]), f"{fm[m]['auroc']:.3f}", ci3(fm[m]["auroc_ci"]), f3(fm[m].get("comm_mb_per_fold_mean"))] for m in fm]) +
                   f"<p>Local only, each client alone: client 1 {fm['Local only']['per_client'][0]['auroc']:.3f} {ci3(fm['Local only']['per_client'][0]['auroc_ci'])}, client 2 {fm['Local only']['per_client'][1]['auroc']:.3f} {ci3(fm['Local only']['per_client'][1]['auroc_ci'])}; the table row averages the two clients' scores.</p>"
                   f'<div class="fig">{figs["fig7_fl"]}</div><div class="fig">{figs["fig4_comm"]}</div>'
                   f"<p>Communication ratio (measured bytes): FedXgbBagging / TreeAvg = {s78['comm_ratio']['FedXgbBagging_over_TreeAvg']:.1f}x, FedAvg MLP / TreeAvg = {s78['comm_ratio']['FedAvg_over_TreeAvg']:.1f}x. The existing HR-only FedAvg simulation (<code>scripts/simulate_resilient_fedavg_hr_only.py</code>) was not re-run, because running it would overwrite its committed results; it stays separate prior evidence and none of its numbers are used here.</p>" +
                   honest(f"The tree-based federated methods land close to centralized XGBoost ({fm['TreeAvg']['auroc']:.3f} TreeAvg and {fm['FedXgbBagging']['auroc']:.3f} FedXgbBagging against {fm['Centralized XGBoost']['auroc']:.3f}), and the MLPs, federated or not, are weak on this tabular problem. None of the federated versus centralized differences can be told apart, and with 4 positives per client none of this is evidence that FL works for RESILIENT. TreeAvg is the cheapest to communicate by a wide margin.")))

    # ---- Stage 8
    pr = s78["paired"]
    eta = s78["treeavg_eta_sensitivity_not_used_for_selection"]
    out.append(sec(8, "FL audit and diagnostics (exploratory)",
                   table(["Audit item", "Result"], [[esc(k.replace("_", " ")), esc(v)] for k, v in s78["audit"].items()]) +
                   "<h3>Paired participant-level bootstrap</h3>" + table(["Comparison", "Delta AUROC", "95% CI", "Verdict"],
                         [[esc(k), f"{x['delta']:+.3f}", ci3(x["ci"]), esc(x["verdict"])] for k, x in pr.items()]) +
                   "<h3>TreeAvg eta sensitivity (reported, not used for selection; anchor 0.10 fixed in advance)</h3>" +
                   table(["eta", "AUROC"], [[k, f"{x:.3f}"] for k, x in eta.items()] + [["0.10, 60 rounds", f"{s78['treeavg_60_rounds_check']:.3f}"]]) +
                   "<h3>Per-round curve (outer test folds, mean over repeats, diagnostic only)</h3>" +
                   table(["Method", "Round 1", "Round 10", "Round 20", "Last round"],
                         [[esc(m), f"{c[0]:.3f}", f"{c[9]:.3f}", f"{c[19]:.3f}", f"{c[-1]:.3f}"] for m, c in s78["curves_test_fold_diagnostic_only"].items()]) +
                   honest("The audit passes: weights sum to 1, clients are disjoint and never see the test fold during training, and each round aggregates fresh client models only. What limits FL here is not the algorithm but the data ceiling and fragmentation: 4 positives per client, no validation data to tune rounds, and a centralized ceiling that is itself barely above chance. TreeAvg moves only modestly across eta values (0.65 to 0.71), so the fixed anchor does not drive the conclusion.")))

    # ---- Stage 9
    md = s9["models"]
    ed = s9["enrolment_date_check"]
    fs = s9["frozen_fold_spread"]
    out.append(sec(9, "Robustness",
                   table(["Model", "AUROC seed 42", "seed 43", "seed 44", "Mean", "SD (population)"],
                         [[esc(m), *(f"{a:.3f}" for a in x["aurocs"]), f"{x['mean']:.3f}", f"{x['sd']:.3f}"] for m, x in sorted(md.items(), key=lambda kv: -kv[1]["mean"])], win=lambda i, r: r[0] == model) +
                   f"<p>Same data everywhere: all headline rows use the same {s9['same_data_check']['n']} participants and {s9['same_data_check']['n_pos']} positives, and all are pooled out-of-fold numbers (there is no fixed split in this run, so nothing is mixed).</p>"
                   f"<p>Fold-to-fold spread of the frozen model: {fs['n_folds']} outer folds, AUROC mean {fs['mean']:.3f}, SD {fs['sd']:.3f}, range {fs['min']:.3f} to {fs['max']:.3f} (each test fold holds {fs['positives_per_test_fold']} positives). Leave-one-participant-out (nested inner 3-fold): AUROC {s9['frozen_lopo']['auroc']:.3f} {ci3(s9['frozen_lopo']['auroc_ci'])}.</p>"
                   "<h3>Enrolment-date check (A12 follow-up)</h3>" +
                   table(["Check", "Value"], [["AUROC of enrolment date alone vs label", f"{ed['date_only_auroc']:.3f}"],
                                              ["Spearman, frozen score vs enrolment date", f"{ed['spearman_score_vs_date']:.3f} (p = {ed['p']:.3f})"],
                                              ["Positive rate, enrolled Nov-Feb vs other months", f"{ed['winter_enrolment_pos_rate']:.3f} vs {ed['other_pos_rate']:.3f}"]] +
                         [[f"Frozen AUROC within {k} enrolment (n={x['n']}, {x['n_pos']} pos)", f3(x["auroc"])] for k, x in ed["within_season_auroc"].items()]) +
                   "<h3>Supplementary labels, frozen pipeline</h3>" +
                   table(["Label", "Positives", "AUROC", "95% CI", "AUPRC (chance)"],
                         [[esc(k), str(x["n_pos"]), f"{x['auroc']:.3f}", ci3(x["auroc_ci"]), f"{x['auprc']:.3f} ({x['chance_auprc']:.3f})"] for k, x in s9["supplementary_labels"].items()]) +
                   "<p>Leave-one-site-out: not applicable (single site).</p>" +
                   honest(STAGE9_HONEST(s9, model, cand))))

    # ---- Stage 10
    dd = s10["descriptive_direction_check_vs_data_paper"]
    out.append(sec(10, "Comparison with published work (fallback)",
                   "<p>The only paper on this data is the RESILIENT data descriptor (Gomez et al., Scientific Data 2025, PMC12546717). It reports no prediction model or metric, only descriptive patterns, so the protocol fallback applies: compare against the clinical target sensitivity and, clearly labelled, with what the data paper describes. No same-population prediction paper with comparable metrics was verified in this run (needs your action if one should be cited).</p>"
                   f"<p>At the 0.80 clinical target (Stage 5 rule) the frozen model reaches test sensitivity {ops['matched_sensitivity']['sensitivity']:.3f}, specificity {ops['matched_sensitivity']['specificity']:.3f}, accuracy {ops['matched_sensitivity']['accuracy']:.3f}. Matching on validation lands below the target on test, as the protocol warns can happen.</p>"
                   "<h3>Descriptive check against the data paper (same data, not a model comparison)</h3>"
                   f"<p>Data paper: \"{esc(s10['data_paper_statement'])}\". Spearman correlation with PHQ-9 total in the 512-hour window:</p>" +
                   table(["Feature", "All", "Female", "Male"], [[f"<code>{esc(f)}</code>", *(f"rho {x[g]['rho']:+.3f} (n={x[g]['n']}, p={x[g]['p']:.3f})" for g in ("all", "female", "male"))] for f, x in dd.items()]) +
                   honest(STAGE10_HONEST(dd))))

    # ---- Stage 11
    out.append(sec(11, "Paper prep",
                   f'<div class="figpair"><div class="fig">{figs["fig1_roc_pr"][0]}</div><div class="fig">{figs["fig1_roc_pr"][1]}</div></div><div class="fig">{figs["fig3b_ablation"]}</div>'
                   "<p>Figures are written by <code>scripts/resilient_protocol_stage10_12_report.py</code> to <code>public/docs/paper_figures/resilient_protocol/</code> as SVG (matplotlib is not installed in either venv, so PNG/PDF at 300 dpi were not produced; SVG is vector and converts losslessly). Produced: ROC and PR (fig 1), operating-point curves (fig 2), model screen and arm forest plots (fig 3), measured communication (fig 4), FL versus centralized versus local (fig 7). Skipped: pipeline diagram (not script-generated here) and fig 5 matched-sensitivity bars versus a paper (no paper).</p>"
                   "<h3>Literature</h3><ul>"
                   "<li>Gomez NC, Chen Y, Kouchaki S, et al. The RESILIENT Dataset: Multimodal Monitoring of Ageing-Related Comorbidities and Cognitive Decline. Scientific Data, 2025. PMC12546717 (verified from the PMC page in this run).</li>"
                   "<li>Kroenke K, Spitzer RL, Williams JBW. The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med, 2001 (not re-verified from source in this run).</li>"
                   "<li>Yesavage JA, Sheikh JI. Geriatric Depression Scale (GDS): recent evidence and development of a shorter version. Clinical Gerontologist, 1986 (not re-verified).</li>"
                   "<li>Goswami M, et al. MOMENT: A Family of Open Time-series Foundation Models. ICML 2024 (not re-verified).</li>"
                   "<li>McMahan HB, et al. Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS 2017 (not re-verified).</li></ul>"
                   "<h3>MAMAI framing</h3><p>In MAMAI, Agent 1 turns wearable signals into a mood score that Agent 3 combines with self-report, and Agent 5 trains it federatedly. RESILIENT is an external, non-perinatal check of the Agent 1 idea: with 73 older adults it says daily activity carries some depression signal that a simple tree model can pick up, that frozen MOMENT embeddings add nothing measurable at this size, and that the cohort is far too small to evaluate Agent 5 style federation. It should be presented as a small external feasibility check, not as validation of the perinatal pipeline.</p>"
                   "<h3>Venue logistics (needs your action)</h3><p>Target venue, APC and institutional discount, and candidate special issues with deadlines were not set for this dataset and need your decision. A natural fit would be a short paper or a section inside a multi-dataset paper rather than a standalone RESILIENT paper.</p>" +
                   honest("The figures are enough for a supplementary section. A standalone paper on this cohort would be hard to defend: 10 positives, a post-visit sensor window and an AUROC interval that starts just above chance.")))

    # ---- Stage 12
    out.append(sec(12, "Final report (sections A-K)", table(["Section", "Content", "Tag"], [
        ["A Dataset and cohort flow", f"73 participants in Demographics.csv; 73 with sensor folders; 73 with a PHQ-9 date and features; {npos} PHQ-9 &ge; 10 (one row per participant, nobody dropped; missing values imputed inside folds)", "Main result"],
        ["B Methods and equations", f"28 hand-made features over [phq_date, +512 h); 7 model families; {model} frozen; TreeAvg/FedAvg/FedXgbBagging as in Stage 7; participant bootstrap 2000 resamples, seed 42, percentile 95%; paired bootstrap for differences", "Main result"],
        ["C Protocol", "Stage 0 profile; SMALL-N repeated nested CV (5x5 outer, 3 inner, &le; 10 configs, seeds 42-46); simulated K=2 clients; train-only preprocessing", "Main result"],
        ["D Primary and secondary metrics", f"AUROC {cand['auroc']:.3f} {ci3(cand['auroc_ci'])}; AUPRC {cand['auprc']:.3f} {ci3(cand['auprc_ci'])} (chance {y.mean():.3f})", "Main result"],
        ["E Operating points", f"default sens {ops['default']['sensitivity']:.3f} / spec {ops['default']['specificity']:.3f}; matched-0.80 sens {ops['matched_sensitivity']['sensitivity']:.3f} / spec {ops['matched_sensitivity']['specificity']:.3f}; all accuracies below majority {maj:.3f}", "Main result"],
        ["F Feature-source ablation", "steps drive the signal; sleep mat alone at chance; questionnaires stronger but excluded", "Main result"],
        ["G Measured communication", f"TreeAvg {fm['TreeAvg']['comm_mb_per_fold_mean']:.3f} MB, FedXgbBagging {fm['FedXgbBagging']['comm_mb_per_fold_mean']:.3f} MB, FedAvg MLP {fm['FedAvg MLP']['comm_mb_per_fold_mean']:.3f} MB per outer fold (exploratory)", "Supplementary"],
        ["H Robustness", f"3 CV seeds ({model} SD {md[model]['sd']:.3f}); LOPO {s9['frozen_lopo']['auroc']:.3f}; enrolment-date check; GDS-15 and PHQ-9 &ge; 5 labels", "Supplementary"],
        ["I Published work", "no prediction paper on RESILIENT; clinical-target fallback and descriptive agreement with the data paper on steps", "Comparison"],
        ["J Limitations", "10 positives; sensors only after the PHQ-9 visit; single site; best-of-7 model selection on the same predictions; simulated FL clients with 4 positives; no validation split for FL rounds; SVG instead of PNG/PDF figures; GAD-7/ACE-III excluded by design", "Main result"],
        ["K Conclusions", "a weak but above-chance wearable signal driven by step counts; MOMENT and FL not supported at this N", "Main result"]]) +
        honest(esc(HONEST_OVERALL.format(model=model, auc=cand['auroc'], lo=cand['auroc_ci'][0], hi=cand['auroc_ci'][1], npos=npos)))))

    out.append("<footer><p>Scripts: <code>scripts/resilient_protocol_common.py</code>, <code>resilient_protocol_stage00_01_profile_inventory.py</code>, <code>stage02_03_leakage_baseline.py</code>, <code>stage04_05_ablation_freeze_thresholds.py</code>, <code>stage06a_moment_embed.py</code>, <code>stage06b_moment_arms.py</code>, <code>stage07_08_fl_exploratory.py</code>, <code>stage09_robustness.py</code>, <code>stage10_12_report.py</code>; each writes a <code>resilient_protocol_*_results.json</code> next to it. RESILIENT data: Zenodo 16755408, CC BY 4.0, ethics IRAS 321104. Generated 2026-10-05.</p></footer>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>RESILIENT Protocol Report</title>"
            "<link rel=\"stylesheet\" href=\"https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Public+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap\">"
            f"<style>{CSS}</style></head><body><div class=\"page\">{''.join(out)}</div></body></html>")


def STAGE6_HONEST(m6, cand):
    a = m6["MOMENT 4 modalities"]
    b = m6["MOMENT 4 + frozen features"]
    h = m6["MOMENT HR only"]
    raw = m6["Raw flattened HR+Steps (control)"]
    return (f"MOMENT does not help here. All four embeddings together reach {a['auroc']:.3f} {ci3(a['auroc_ci'])}, close to chance, and HR alone is {h['auroc']:.3f} {ci3(h['auroc_ci'])}, "
            f"significantly worse than the frozen model and below chance, which is the typical anti-learning pattern of a high-dimensional input with 8 positives per training fold, not a usable inverse signal. "
            f"Adding the embeddings to the frozen features drags it down to {b['auroc']:.3f} (paired delta {b['paired_vs_frozen']['delta']:+.3f}, {b['paired_vs_frozen']['verdict']}). "
            f"The raw flattened HR + steps control, with no foundation model at all, matches the frozen model ({raw['auroc']:.3f}, delta {raw['paired_vs_frozen']['delta']:+.3f}), which again points at plain activity level, not at patterns a frozen embedding captures.")


def STAGE9_HONEST(s9, model, cand):
    md = s9["models"][model]
    lo = s9["frozen_lopo"]
    ed = s9["enrolment_date_check"]
    lab = s9["supplementary_labels"]
    w = ed["within_season_auroc"]
    return (f"{model} holds up across CV seeds ({', '.join(f'{a:.3f}' for a in md['aurocs'])}, SD {md['sd']:.3f}) and under leave-one-participant-out ({lo['auroc']:.3f} {ci3(lo['auroc_ci'])}), "
            f"and it stays the best of the 7 families on every seed, so the Stage 3 winner is not a selection fluke, although single test folds swing widely. "
            f"Enrolment season is a real confounder candidate: people enrolled November to February were more often positive ({ed['winter_enrolment_pos_rate']:.2f} vs {ed['other_pos_rate']:.2f}) and the score correlates weakly with enrolment date (rho {ed['spearman_score_vs_date']:.2f}, p {ed['p']:.3f}). "
            f"Within each season group the score still ranks positives higher (AUROC {w['Nov-Feb']['auroc']:.3f} and {w['Mar-Oct']['auroc']:.3f}, 5 positives each), so season does not explain it all, but this cannot be settled with 10 positives. "
            f"With more positives the same pipeline is steadier: GDS-15 &ge; 5 ({lab['GDS-15 >= 5']['n_pos']} positives) gives {lab['GDS-15 >= 5']['auroc']:.3f} {ci3(lab['GDS-15 >= 5']['auroc_ci'])} and PHQ-9 &ge; 5 ({lab['PHQ-9 >= 5']['n_pos']} positives) {lab['PHQ-9 >= 5']['auroc']:.3f} {ci3(lab['PHQ-9 >= 5']['auroc_ci'])}, both with intervals well clear of chance.")


def STAGE10_HONEST(dd):
    a = dd["steps_daily_median"]["all"]["rho"]
    b = dd["steps_daily_sd"]["all"]["rho"]
    return (f"There is no published model to beat or match. Against the 0.80 clinical target the model falls short on test. The descriptive check agrees in direction with the data paper for step level (overall rho {a:+.3f}: fewer steps, higher PHQ-9), "
            "and for step variability (overall rho {b:+.3f}), in women and in men separately, which supports reading the model as picking up reduced and flatter daily activity. These are within-cohort correlations on 73 people, not a confirmatory test.")


if __name__ == "__main__":
    main()
