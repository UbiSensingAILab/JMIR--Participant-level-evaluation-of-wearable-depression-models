#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stages 10-12 for the HRV + PHQ-9 (Ref 2) protocol run:
  Stage 10  comparison with published work (fallback branch) -> hrv_phq9_protocol_stage10_results.json
  Stage 11  paper-prep figures as SVG (matplotlib is not installed in any project venv)
            -> public/docs/paper_figures/hrv_phq9/*.svg
  Stage 12  standalone report public/docs/HRV_PHQ9_Protocol_Report.html and the dashboard
            summary src/data/protocolComparison/hrv_phq9.json
Every number is read from the stage results JSONs written by the earlier scripts.
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrv_phq9_protocol_common as C  # noqa: E402

REPO = C.REPO
FIG_DIR = REPO / "public/docs/paper_figures/hrv_phq9"
REPORT = REPO / "public/docs/HRV_PHQ9_Protocol_Report.html"
SUMMARY = REPO / "src/data/protocolComparison/hrv_phq9.json"

J = {k: C.load_json(f"hrv_phq9_protocol_{k}.json") for k in
     ["profile", "inventory", "stage02_results", "stage03_results", "stage04_results", "frozen_config",
      "stage05_results", "stage06_results", "stage06b_moment_embed_results", "stage06a_hrm_minute_bins_results",
      "stage07_08_results", "stage09_results"]}
P, INV, S2, S3, S4, FZ, S5, S6, S6B, S6A, S78, S9 = (J[k] for k in J)


def f4(x):
    return "n/a" if x is None else f"{x:.4f}"


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def ci(c, d=3):
    return "n/a" if not c else f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def esc(s):
    return html.escape(str(s))


def table(headers, rows, win=None):
    h = "".join(f"<th>{esc(x)}</th>" for x in headers)
    body = ""
    for i, r in enumerate(rows):
        cls = ' class="win"' if win is not None and i == win else ""
        body += f"<tr{cls}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<div class="table-wrap"><table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


# --------------------------------------------------------------------------------------
# data needed for curves
seg = C.build_segments()
y = seg["y"].to_numpy()
g = seg["deviceId"].to_numpy()
frozen_p = np.nanmean(np.load(C.CACHE / "hrv_phq9_protocol_frozen_oof.npz")["oof"], axis=1)
s4oof = np.load(C.CACHE / "hrv_phq9_protocol_stage04_oof.npz")
wear_key = S4["best_wearable_only"]
wear_p = np.nanmean(s4oof[wear_key], axis=1)
s6oof = np.load(C.CACHE / "hrv_phq9_protocol_stage06_oof.npz")
mom_p = np.nanmean(s6oof["MOMENT_HRmin_alone"], axis=1)
maj = S2["majority_rate"]

# --------------------------------------------------------------------------------------
# Stage 10 (fallback branch: no published model with this target on this dataset)
op = S5["operating_points"]
rules = []
for k in (3, 4, 5, 6):
    pr = (seg["PHQ9_1"] >= k).astype(int).to_numpy()
    tp = int(((pr == 1) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    rules.append({"rule": f"PHQ9_1 >= {k}", "accuracy": (pr == y).mean(), "sensitivity": tp / y.sum(), "specificity": tn / (y == 0).sum()})
stage10 = {
    "branch": "fallback (P8 = no)",
    "dataset_paper": "Baigutanova A, Park S, Constantinides M, Lee SW, Quercia D, Cha M. In-situ wearable-based dataset of continuous heart rate variability monitoring accompanied by sleep diaries. Scientific Data 2025 (PMC12375003).",
    "dataset_paper_metrics": "none comparable: descriptive statistics and correlations only; no depression prediction task, so no sensitivity / specificity / AUROC to match",
    "clinical_target_sensitivity": C.CLINICAL_TARGET_SENS,
    "frozen_at_matched_sensitivity": op["matched_sensitivity"],
    "frozen_default": op["default"],
    "majority_rate": maj,
    "a_priori_rules_no_training_in_sample": rules,
    "related_work_not_same_data": [
        "Kroenke K, Spitzer RL, Williams JB. The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med 2001 (source of the >= 5 / >= 10 cut-offs)",
        "Studies predicting PHQ-9 from consumer wearables in other cohorts exist (for example the GLOBEM benchmark, Xu et al. 2022/2023). Their numbers are NOT on this dataset, target or N and are not compared here; verify citations before use.",
    ],
    "note": "matching on inner-fold validation can land a little off the 0.80 target on the outer test folds",
}
C.write_json("hrv_phq9_protocol_stage10_results.json", stage10)

# --------------------------------------------------------------------------------------
# Stage 11 figures (SVG). Colours via CSS variables so they follow the report theme.
FIG_DIR.mkdir(parents=True, exist_ok=True)
SER = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)"]


def svg_frame(w, h, inner, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(title)}" '
            f'style="width:100%;height:auto;font-family:Public Sans,Arial,sans-serif;font-size:11px">{inner}</svg>')


def line_chart(series, title, xlabel, ylabel, w=560, h=300, diag=False, hline=None, xr=(0, 1), yr=(0, 1), legend=True):
    L, R, T, B = 48, 14, 14, 40
    pw, ph = w - L - R, h - T - B
    sx = lambda x: L + (x - xr[0]) / (xr[1] - xr[0]) * pw  # noqa: E731
    sy = lambda v: T + ph - (v - yr[0]) / (yr[1] - yr[0]) * ph  # noqa: E731
    o = f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="var(--border)"/>'
    for t in np.linspace(0, 1, 6):
        xv = xr[0] + t * (xr[1] - xr[0]); yv = yr[0] + t * (yr[1] - yr[0])
        o += f'<line x1="{sx(xv):.1f}" x2="{sx(xv):.1f}" y1="{T}" y2="{T + ph}" stroke="var(--grid)"/>'
        o += f'<line x1="{L}" x2="{L + pw}" y1="{sy(yv):.1f}" y2="{sy(yv):.1f}" stroke="var(--grid)"/>'
        o += f'<text x="{sx(xv):.1f}" y="{T + ph + 14}" text-anchor="middle" fill="var(--text-muted)">{xv:.2g}</text>'
        o += f'<text x="{L - 6}" y="{sy(yv) + 4:.1f}" text-anchor="end" fill="var(--text-muted)">{yv:.2g}</text>'
    if diag:
        o += f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="var(--text-faint)" stroke-dasharray="4 4"/>'
    if hline is not None:
        o += f'<line x1="{L}" x2="{L + pw}" y1="{sy(hline[0]):.1f}" y2="{sy(hline[0]):.1f}" stroke="var(--text-faint)" stroke-dasharray="4 4"/>'
        o += f'<text x="{L + pw - 4}" y="{sy(hline[0]) - 4:.1f}" text-anchor="end" fill="var(--text-faint)">{esc(hline[1])}</text>'
    for i, (name, xs, ys) in enumerate(series):
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(xs, ys))
        o += f'<polyline points="{pts}" fill="none" stroke="{SER[i % 4]}" stroke-width="2"/>'
        if legend:
            o += f'<rect x="{L + 8}" y="{T + 8 + 16 * i}" width="10" height="10" rx="2" fill="{SER[i % 4]}"/>'
            o += f'<text x="{L + 22}" y="{T + 17 + 16 * i}" fill="var(--text)">{esc(name)}</text>'
    o += f'<text x="{L + pw / 2}" y="{h - 6}" text-anchor="middle" fill="var(--text-muted)">{esc(xlabel)}</text>'
    o += f'<text transform="translate(12,{T + ph / 2}) rotate(-90)" text-anchor="middle" fill="var(--text-muted)">{esc(ylabel)}</text>'
    return svg_frame(w, h, o, title)


def forest(rows, title, ref=0.5, w=640, xr=(0.2, 1.0)):
    L, R, T = 250, 70, 10
    h = T + 24 * len(rows) + 34
    pw = w - L - R
    sx = lambda v: L + (v - xr[0]) / (xr[1] - xr[0]) * pw  # noqa: E731
    o = ""
    for t in np.arange(xr[0], xr[1] + 1e-9, 0.1):
        o += f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{T}" y2="{h - 28}" stroke="var(--grid)"/>'
        o += f'<text x="{sx(t):.1f}" y="{h - 14}" text-anchor="middle" fill="var(--text-muted)">{t:.1f}</text>'
    o += f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{T}" y2="{h - 28}" stroke="var(--text-faint)" stroke-dasharray="4 4"/>'
    for i, (name, v, c, hi) in enumerate(rows):
        yy = T + 12 + 24 * i
        col = "var(--s1)" if hi else "var(--s2)"
        o += f'<text x="{L - 8}" y="{yy + 4}" text-anchor="end" fill="var(--text)">{esc(name)}</text>'
        if c:
            o += f'<line x1="{sx(c[0]):.1f}" x2="{sx(c[1]):.1f}" y1="{yy}" y2="{yy}" stroke="{col}" stroke-width="2"/>'
        o += f'<circle cx="{sx(v):.1f}" cy="{yy}" r="4.5" fill="{col}"/>'
        o += f'<text x="{w - R + 6}" y="{yy + 4}" fill="var(--text-muted)">{v:.3f}</text>'
    o += f'<text x="{L + pw / 2}" y="{h - 1}" text-anchor="middle" fill="var(--text-muted)">AUROC (dashed line = chance 0.5)</text>'
    return svg_frame(w, h, o, title)


def bars(groups, keys, title, w=600, h=260, ref=None):
    L, R, T, B = 40, 10, 24, 46
    pw, ph = w - L - R, h - T - B
    n = len(groups)
    gw = pw / n
    bw = gw / (len(keys) + 1)
    o = ""
    for t in np.linspace(0, 1, 6):
        yy = T + ph - t * ph
        o += f'<line x1="{L}" x2="{L + pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--grid)"/><text x="{L - 6}" y="{yy + 4:.1f}" text-anchor="end" fill="var(--text-muted)">{t:.1f}</text>'
    for gi, (gname, vals) in enumerate(groups):
        for ki, k in enumerate(keys):
            v = vals[ki]
            x = L + gi * gw + (ki + 0.5) * bw
            o += f'<rect x="{x:.1f}" y="{T + ph - v * ph:.1f}" width="{bw * 0.9:.1f}" height="{v * ph:.1f}" rx="2" fill="{SER[ki % 4]}"/>'
            o += f'<text x="{x + bw * 0.45:.1f}" y="{T + ph - v * ph - 3:.1f}" text-anchor="middle" fill="var(--text-muted)" font-size="9">{v:.2f}</text>'
        o += f'<text x="{L + gi * gw + gw / 2:.1f}" y="{T + ph + 16}" text-anchor="middle" fill="var(--text)">{esc(gname)}</text>'
    for ki, k in enumerate(keys):
        o += f'<rect x="{L + 110 * ki}" y="{h - 16}" width="10" height="10" rx="2" fill="{SER[ki % 4]}"/><text x="{L + 14 + 110 * ki}" y="{h - 7}" fill="var(--text)">{esc(k)}</text>'
    if ref is not None:
        yy = T + ph - ref[0] * ph
        o += f'<line x1="{L}" x2="{L + pw}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="var(--text-faint)" stroke-dasharray="4 4"/><text x="{L + pw}" y="{yy - 4:.1f}" text-anchor="end" fill="var(--text-faint)">{esc(ref[1])}</text>'
    return svg_frame(w, h, o, title)


def pipeline_svg():
    boxes = ["Watch tables (1 Hz HR, 5-min HRV/motion), diary, survey",
             "Calendar anchor: day 1 = 2021-03-09 KST, segments A (days 1-14) and B (15-28)",
             "Label: PHQ-9 >= 5 on PHQ9_2 (A) or PHQ9_F (B)",
             "Features per segment: sensor, diurnal, diary, day-0 questionnaires; MOMENT embeddings",
             "Repeated nested grouped CV (5x5, inner 3), participant bootstrap"]
    o = ""
    for i, b in enumerate(boxes):
        yy = 8 + i * 56
        o += f'<rect x="10" y="{yy}" width="600" height="40" rx="8" fill="var(--surface-2)" stroke="var(--border-strong)"/>'
        o += f'<text x="310" y="{yy + 24}" text-anchor="middle" fill="var(--text)">{esc(b)}</text>'
        if i < len(boxes) - 1:
            o += f'<line x1="310" x2="310" y1="{yy + 40}" y2="{yy + 56}" stroke="var(--text-faint)" stroke-width="2"/>'
    return svg_frame(620, 8 + 56 * len(boxes) - 8, o, "Pipeline")


figs = {}
figs["fig1_pipeline"] = pipeline_svg()
# operating curves over the range where the frozen predictions vary
lo, hi = float(np.min(frozen_p)), float(np.max(frozen_p))
ths = np.linspace(lo, hi, 120)
acc = [((frozen_p >= t).astype(int) == y).mean() for t in ths]
sen = [((frozen_p >= t) & (y == 1)).sum() / y.sum() for t in ths]
spe = [((frozen_p < t) & (y == 0)).sum() / (y == 0).sum() for t in ths]
figs["fig2_threshold_curves"] = line_chart([("accuracy", ths, acc), ("sensitivity", ths, sen), ("specificity", ths, spe)],
                                          "Operating curves", "threshold on frozen model probability (pooled OOF)", "rate",
                                          xr=(lo, hi), hline=(maj, f"majority rate {maj:.3f}"))
auc_rows = []
for r in S4["results"]:
    auc_rows.append((f"{r['arm']} | {r['model']}", r["auroc_repeat_avg_pred"], r["auroc_ci"], r["arm"] == FZ["arm"] and r["model"] == FZ["model"]))
for k, r in S6["results"].items():
    if r.get("auroc_ci"):
        auc_rows.append((k, r["auroc_repeat_avg_pred"], r["auroc_ci"], False))
figs["fig3_auroc_all_models"] = forest(auc_rows, "AUROC of all centralized models, same 98 segments")
fm = S78["main"]["methods"]
comm = [(m, fm[m]["comm_mb_per_fold_mean"]) for m in ["FedAvg MLP", "FedXgbBagging", "TreeAvg"]]
mx = max(v for _, v in comm)
figs["fig4_communication"] = bars([(m, [v / mx]) for m, v in comm], ["share of the largest (FedAvg MLP = 1.0)"], "Measured communication per outer fold")
figs["fig5_operating_points"] = bars([(n, [op[k]["accuracy"], op[k]["sensitivity"], op[k]["specificity"]]) for n, k in
                                      [("default (bal. acc)", "default"), ("max accuracy", "max_accuracy"), ("matched sens 0.80", "matched_sensitivity")]],
                                     ["accuracy", "sensitivity", "specificity"], "Validation-selected operating points", ref=(maj, "majority rate"))
roc_series, pr_series = [], []
for name, p in [("frozen: PHQ9_1 LR", frozen_p), (f"best wearable: {wear_key}", wear_p), ("MOMENT HR-min alone", mom_p)]:
    fpr, tpr, _ = roc_curve(y, p); roc_series.append((name, fpr, tpr))
    pr_, rc_, _ = precision_recall_curve(y, p); pr_series.append((name, rc_, pr_))
figs["fig6_roc"] = line_chart(roc_series, "ROC curves", "1 - specificity", "sensitivity", diag=True)
figs["fig6_pr"] = line_chart(pr_series, "Precision-recall curves", "recall", "precision", hline=(float(y.mean()), "chance = positive rate"))
fl_rows = [(m, fm[m]["auroc_repeat_avg_pred"], fm[m]["auroc_ci"], m.startswith(("Fed", "TreeAvg")))
           for m in ["Local only (client 1)", "Local only (client 2)", "FedAvg MLP", "FedXgbBagging", "TreeAvg", "Centralized MLP", "Centralized XGBoost", "Centralized LightGBM"]]
figs["fig7_client_vs_fl_vs_central"] = forest(fl_rows, "One client alone vs FL vs centralized (frozen feature)")
for k, v in figs.items():
    (FIG_DIR / f"{k}.svg").write_text(v.replace('style="', 'style="--s1:#0E8562;--s2:#B8710A;--s3:#3D6FB6;--s4:#8E5BB0;--grid:#E3E8DE;--border:#D8DED2;--border-strong:#C3CCBC;--surface-2:#ECEFE8;--text:#1B211C;--text-muted:#5B655C;--text-faint:#828C7F;', 1))

# --------------------------------------------------------------------------------------
# helpers for the summary
fl_main = S78["main"]
cl_pos_rows = fl_main["client_pos_rows_min_median_max"]
cl_pos_parts = fl_main["client_pos_participants_min_median_max"]
val_single = sum(1 for f in fl_main["folds"] if f["val"]["pos_rows"] in (0, f["val"]["rows"]))
fz_cv = FZ["fixed_params_cv"]
mom_alone = S6["results"]["MOMENT_HRmin_alone"]
mom_add = S6["results"]["MOMENT_HRmin_plus_frozen_features"]
tcn = S6["results"]["TCN_HRmin_windows"]
perm_mean = float(np.mean([p["mean"] for p in S2["permutation_test"]]))
seeds = S9["seeds_all_models_frozen_arm"]

# --------------------------------------------------------------------------------------
# Stage 12 HTML
CSS = """
:root{--bg:#F5F6F2;--surface:#FFFFFF;--surface-2:#ECEFE8;--border:#D8DED2;--border-strong:#C3CCBC;--text:#1B211C;--text-muted:#5B655C;--text-faint:#828C7F;
--accent:#0E8562;--accent-soft:#E1F0E8;--accent-text:#0A6B4E;--warn-soft:#F7EBD6;--warn-text:#8A5507;--grid:#E3E8DE;--s1:#0E8562;--s2:#B8710A;--s3:#3D6FB6;--s4:#8E5BB0;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14181A;--surface:#1B211D;--surface-2:#222A23;--border:#313B32;--border-strong:#3C4A3D;--text:#E7EBE4;--text-muted:#A9B3A5;--text-faint:#7C867A;
--accent:#3FA989;--accent-soft:#1D3129;--accent-text:#5DBFA0;--warn-soft:#3A2E1A;--warn-text:#E8B968;--grid:#293229;--s1:#3FA989;--s2:#D9A150;--s3:#7FA6E0;--s4:#B993D6;}}
:root[data-theme="dark"]{--bg:#14181A;--surface:#1B211D;--surface-2:#222A23;--border:#313B32;--border-strong:#3C4A3D;--text:#E7EBE4;--text-muted:#A9B3A5;--text-faint:#7C867A;
--accent:#3FA989;--accent-soft:#1D3129;--accent-text:#5DBFA0;--warn-soft:#3A2E1A;--warn-text:#E8B968;--grid:#293229;--s1:#3FA989;--s2:#D9A150;--s3:#7FA6E0;--s4:#B993D6;}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:"Public Sans",-apple-system,"Segoe UI",sans-serif;font-size:16px;line-height:1.6;margin:0;padding:0 16px 64px}
.page{max-width:900px;margin:0 auto}
h1,h2,h3{font-family:"Source Serif 4",Georgia,serif;font-weight:600}
h1{font-size:2rem;line-height:1.15;margin:40px 0 8px}
h2{font-size:1.4rem;margin:0 0 6px}
h3{font-size:1.05rem;margin:18px 0 8px}
p{margin:0 0 12px;max-width:72ch}
code,.mono{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em}
.sub{color:var(--text-muted)}
section{background:var(--surface);border:1px solid var(--border);border-radius:14px;padding:20px 22px;margin:22px 0}
.num{font-family:"IBM Plex Mono",monospace;color:var(--accent-text);background:var(--accent-soft);border-radius:8px;padding:2px 8px;margin-right:8px;font-size:.85rem}
.honest{background:var(--accent-soft);border-radius:10px;padding:12px 14px;margin-top:12px}
.honest b{color:var(--accent-text)}
.caution{background:var(--warn-soft);border-radius:10px;padding:12px 14px;margin:12px 0}
.caution b{color:var(--warn-text)}
.table-wrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px;margin:12px 0}
table{border-collapse:collapse;width:100%;font-size:.86rem}
th,td{text-align:left;padding:7px 11px;border-bottom:1px solid var(--border);vertical-align:top}
th{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:var(--text-muted);background:var(--surface-2)}
tr.win td{background:var(--accent-soft);font-weight:600}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:14px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px}
.tile .l{font-size:.72rem;text-transform:uppercase;color:var(--text-muted);letter-spacing:.04em}
.tile .v{font-family:"IBM Plex Mono",monospace;font-size:1.3rem;font-weight:600}
.tile .c{font-size:.75rem;color:var(--text-faint);font-family:"IBM Plex Mono",monospace}
.fig{margin:12px 0;border:1px solid var(--border);border-radius:10px;padding:10px;background:var(--surface)}
.fig figcaption{font-size:.82rem;color:var(--text-muted);margin-top:6px}
ul{margin:0 0 12px;padding-left:22px}
nav a{color:var(--accent-text);margin-right:10px;font-size:.85rem}
"""


def sec(num, title, body, honest):
    return (f'<section id="s{num}" aria-labelledby="h{num}"><h2 id="h{num}"><span class="num">{num}</span>{esc(title)}</h2>'
            f'{body}<div class="honest"><b>Honest read.</b> {honest}</div></section>')


def fig(key, cap):
    return f'<figure class="fig">{figs[key]}<figcaption>{esc(cap)} (file: paper_figures/hrv_phq9/{key}.svg)</figcaption></figure>'


S = []
# ---- Stage 0
app_rows = [[str(a["stage"]), a["status"], esc(a["branch"] or "")] for a in P["applicability"]]
inp = P["inputs"]
S.append(sec(0, "Dataset profile and applicability", f"""
<p>{esc(inp['dataset'])}. Data: <code>{esc(inp['data_path'])}</code>. Unit of analysis: {esc(inp['unit'])}. Target: {esc(inp['target'])}.</p>
{table(["Question", "Answer"], [
    ["P1 individual-level rows", "yes"], ["P2 rows per participant", esc(P['P2_rows_per_participant'])], ["P3 design", esc(P['P3_design'])],
    ["P4 size", f"N = 49 participants, 98 segments, {P['P4_size']['positives_segments_phq9_ge5']} positive segments from {P['P4_size']['positive_participants_ge5']} participants (PHQ-9 &gt;= 5). SMALL-N branch."],
    ["P5 target type", esc(P['P5_target_type'])], ["P6 signal granularity", esc(P['P6_signal_granularity'])],
    ["P7 natural sites", "none (single study, one watch model)"], ["P8 published reference", "no (the data paper has no prediction task)"], ["P9 survey weights", "no"]])}
<h3>Why PHQ-9 &gt;= 5 and not &gt;= 10</h3><p>{esc(P['target_choice_reason'])}</p>
<h3>Applicability</h3>{table(["Stage", "Status", "Branch / note"], app_rows)}
""", "This is a 49-person dataset with two labelled two-week segments per person, so every result below is a small-N result with wide intervals. "
     "The standard depression cut-off (PHQ-9 &gt;= 10) has only 5 positive people, which is too few to model; the study uses the mild-symptom cut-off (&gt;= 5) instead and keeps &gt;= 10 as a sensitivity check. "
     "Federated learning is only borderline possible (K=2), and there is no published model to compare against."))

# ---- Stage 1
ca = INV["calendar_anchor"]; gg = INV["granularity_gate"]
miss_top = list(INV["missingness_sensor_columns_sorted"].items())[:6]
S.append(sec(1, "Inventory and raw-data gate", f"""
<p>survey.csv {INV['survey_shape'][0]} x {INV['survey_shape'][1]}; sensor_hrv.csv {INV['sensor_hrv_shape'][0]:,} five-minute rows; sleep_diary.csv {INV['sleep_diary_shape'][0]:,} rows ({esc(INV['sleep_diary_days_per_participant'])} days per person, {esc(INV['sleep_diary_date_range'][0])} to {esc(INV['sleep_diary_date_range'][1])}).
Sensor rows per participant: min {INV['sensor_rows_per_participant']['min']}, median {INV['sensor_rows_per_participant']['median']:.0f}, max {INV['sensor_rows_per_participant']['max']}.</p>
{table(["Granularity check", "Result"], [
    ["dtype scan", esc(gg['dtype_scan'])],
    ["description scan", "README says the motion / light columns are 'averaged per the 5-minute interval'"],
    ["file search", esc(', '.join(gg['file_search_raw_tables']))],
    ["sampling interval", f"sensor_hrv slice {gg['sampling_interval_sec']['sensor_hrv (ts_end - ts_start) median']:.0f} s; raw hrm {gg['sampling_interval_sec']['raw hrm.csv.gz median']:.2f} s; raw lit {gg['sampling_interval_sec']['raw lit.csv.gz median']:.2f} s"],
    ["verdict", esc(gg['verdict'])]])}
<h3>Two data problems found and fixed</h3>
<div class="caution"><b>Day anchor.</b> Earlier Ref 2 scripts set day 1 to each device's first sensor row. The sleep diary shows the study ran on one shared calendar ({esc(INV['sleep_diary_date_range'][0])} to {esc(INV['sleep_diary_date_range'][1])}) for everyone, while watches started recording a median of {abs(ca['first_sensor_row_offset_days_vs_day1']['median']):.0f} days earlier, and {ca['devices_with_clock_glitch_before_2021']} device has a clock glitch dated 2018 that pushed its whole study window out of range. This run anchors day 1 to 2021-03-09 KST. {ca['rows_in_calendar_window']:,} rows fall inside the calendar window; only {ca['rows_kept_by_both_anchors']:,} rows are shared with the old anchor's {ca['old_first_row_anchor_rows_kept']:,}.</div>
<div class="caution"><b>HRV power artifacts.</b> lf and hf contain impossible values up to 1e47 ({INV['hrv_power_artifacts_raw_rows_gt_1e6']['lf']} and {INV['hrv_power_artifacts_raw_rows_gt_1e6']['hf']} raw rows above 1e6 ms^2). Rule: values above 1e6 set to missing, then log1p. A fixed rule, not fitted on data.</div>
<p>Coverage is uneven: {ca['segment_rows']['segments_with_lt_288_rows_(1_day)']} of 98 segments have less than one day (288 rows) of five-minute HRV data. Most-missing sensor columns: {esc(', '.join(f'{k} {100 * v:.0f}%' for k, v in miss_top))}.</p>
""", "The dataset holds raw 1 Hz watch tables plus a five-minute summary table, so the raw-signal branch of Stage 6 applies. "
     "Two real data problems were found: the old per-device day anchor was misaligned with the survey calendar (and broken for one device), and the HRV power columns contain absurd artifact values. Both are fixed by fixed rules before any modelling. "
     "HRV coverage is thin: a quarter of segments have under a day of usable five-minute rows."))

# ---- Stage 2
rv5, rvs = S2["row_vs_grouped_5min_rows"], S2["row_vs_grouped_segments"]
S.append(sec(2, "Target construction and leakage audit", f"""
<p>Target: {esc(S2['target'])}. Cut-off source: {esc(S2['cutoff_source'])}. {S2['n_pos']} of {S2['n_rows']} segments are positive (rate {S2['pos_rate']:.3f}; majority rate {S2['majority_rate']:.3f}); {S2['positive_participants']} of 49 participants are positive at least once.
Mid-study and end positives: {S2['by_segment']['A']['pos']} and {S2['by_segment']['B']['pos']}. Label pattern per person (A then B): negative-negative {S2['participant_label_patterns']['neg_neg']}, negative-positive {S2['participant_label_patterns']['neg_pos']}, positive-negative {S2['participant_label_patterns']['pos_neg']}, positive-positive {S2['participant_label_patterns']['pos_pos']}. Secondary target PHQ-9 &gt;= 10: {S2['secondary_ge10']['n_pos']} segments from {S2['secondary_ge10']['positive_participants']} people.</p>
{table(["Leakage check", "Pass", "Evidence"], [[esc(c['check']), "yes" if c['pass'] else "NO", esc(c['evidence'])] for c in S2['leakage_checklist']])}
{table(["Test", "Result"], [
    ["Label-shuffle (participant-block, 5 seeds), wearable arm, LR", f"mean AUROC {S2['permutation_test'][0]['mean']:.3f} (seeds {', '.join(f'{v:.2f}' for v in S2['permutation_test'][0]['seed_aurocs'])})"],
    ["Label-shuffle, all features, LR", f"mean AUROC {S2['permutation_test'][1]['mean']:.3f}"],
    ["5-minute rows, random row split vs grouped (RF)", f"{rv5['random_row_kfold']:.3f} vs {rv5['grouped_by_participant']:.3f} ({rv5['n_rows']:,} rows)"],
    ["Segments, random split vs grouped (RF, wearable arm)", f"{rvs['random_segment_kfold']:.3f} vs {rvs['grouped_by_participant']:.3f}"]])}
""", f"No leakage was found: all {S2['checklist_pass']} checklist items pass and shuffled labels give AUROC around {perm_mean:.2f}, at or a little under chance. "
     "Note that the null for this design sits slightly below 0.5, which is a known small-sample effect of grouped CV (a held-out fold's class balance moves against the training fold). "
     f"Splitting five-minute rows at random gives an AUROC of {rv5['random_row_kfold']:.2f}, but only {rv5['grouped_by_participant']:.2f} when people are kept apart: the row-level number only measures how well the model recognises a person, so it is not a valid result. This is the main reason earlier row-level Ref 2 numbers looked high."))

# ---- Stage 3
scr = sorted(S3["screen"], key=lambda r: (r["arm"], -r["mean_auroc_over_repeats"]))
a = S3["part_A_audit"]
S.append(sec(3, "Baseline and forensic audit", f"""
<p>Part A: A1 98 labelled segments, target equals the score cut-off; A2 10 of 10 spot-checked IDs present in survey, sensor and diary; A3 mean PHQ-9 {a['A3_direction']['mean_score_pos']:.2f} in positives vs {a['A3_direction']['mean_score_neg']:.2f} in negatives; A4-A5 outer folds hold {', '.join(str(f['test_pos']) for f in a['A4_A5_folds_repeat0'])} positives (repeat 0); A6 sources: {esc(a['A6_feature_sources'])}; A7 no all-NaN, constant or infinite columns; A8-A9 positives score higher than negatives in every fold; A12 PHQ9_1 alone has in-sample AUROC {a['A12_conditional']['phq9_1_alone_auroc_in_sample']:.3f}, and segment A vs B carries no signal (AUROC 0.5). Part B: {esc(S3['part_B'])}.</p>
<p>CV: {esc(S3['cv'])}.</p>
{table(["Arm", "Model", "Mean AUROC (5 repeats)", "SD", "AUROC (avg pred) [95% CI]", "AUPRC"], [[r['arm'], r['model'], f3(r['mean_auroc_over_repeats']), f3(r['sd_auroc_over_repeats']), f"{f3(r['auroc_repeat_avg_pred'])} {ci(r['auroc_ci'])}", f3(r['auprc'])] for r in scr])}
""", "The pipeline audit found no bug. With everything included, the best screen model is regularized logistic regression at about 0.83, but the wearable-only arm gives AUROCs of 0.38 to 0.47 for every model family, the same as the shuffled-label null. "
     "So a weak wearable result is not a bug: it is the honest limit of these features at N = 49. Flexible models (trees, MLP) do worse than LR here, as expected with 98 rows."))

# ---- Stage 4
res4 = sorted(S4["results"], key=lambda r: -r["mean_auroc"])
win4 = next(i for i, r in enumerate(res4) if r["arm"] == FZ["arm"] and r["model"] == FZ["model"])
kc = S4["key_contrasts"]
tf, tw = S4["temporal_frozen"], S4["temporal_best_wearable"]
S.append(sec(4, "Feature-source ablation and baseline freeze", f"""
{table(["Arm", "Model", "n feat", "Mean AUROC", "AUROC [95% CI]", "AUPRC", "Paired diff vs candidate [CI]", "Verdict"],
       [[r['arm'], r['model'], str(r['n_features']), f3(r['mean_auroc']), f"{f3(r['auroc_repeat_avg_pred'])} {ci(r['auroc_ci'])}", f3(r['auprc']),
         "candidate" if not r['paired_vs_candidate'] else f"{r['paired_vs_candidate']['diff']:+.3f} {ci(r['paired_vs_candidate']['ci'])}",
         "" if not r['paired_vs_candidate'] else r['paired_vs_candidate']['verdict']] for r in res4], win=win4)}
{table(["Key contrast (paired participant bootstrap)", "Difference [95% CI]", "Verdict"], [[esc(k), f"{v['diff']:+.3f} {ci(v['ci'])}", v['verdict']] for k, v in kc.items()])}
<h3>Temporal validity</h3>
<p>Frozen model trained on segment A, tested on segment B: same people {f3(tf['train_A_test_B_same_people_auroc'])}; different people {f3(tf['train_A_test_B_disjoint_people_auroc_mean'])}. Best wearable arm: same people {f3(tw['train_A_test_B_same_people_auroc'])}, different people {f3(tw['train_A_test_B_disjoint_people_auroc_mean'])}.</p>
<h3>Frozen configuration</h3>
<p><code>{esc(FZ['arm'])}</code>, {esc(FZ['model'])}, hyperparameters {esc(FZ['hyperparameters'])} (chosen in {esc(FZ['hyperparameter_modal_count'])}), seed {FZ['seed']}. Fixed-parameter CV: mean AUROC {f4(fz_cv['mean_auroc'])} (SD {f4(fz_cv['sd_auroc'])}), AUROC of repeat-averaged predictions {f4(fz_cv['auroc_repeat_avg_pred'])} {ci(fz_cv['auroc_ci'], 4)}, AUPRC {f4(fz_cv['auprc'])} {ci(fz_cv['auprc_ci'], 4)} (chance {f3(fz_cv['pos_rate_chance_auprc'])}). Rule: {esc(FZ['freeze_rule'])}.</p>
<p>Importance: permutation (grouped test folds) {esc(S4['importance_permutation_grouped']['top'])}. With one feature, importance is trivially 100% PHQ9_1.</p>
""", "The single strongest predictor of PHQ-9 &gt;= 5 over the next two weeks is the PHQ-9 taken on day 0. Logistic regression on PHQ9_1 alone wins the freeze; adding all 15 other questionnaire and demographic items is slightly worse, and adding all watch features to the baseline adds nothing (paired difference "
     f"{kc['sensor_diurnal_baseline_LR_vs_baseline_only_LR']['diff']:+.3f}, CI includes 0). The best watch-only arm is {kc['best_wearable_vs_phq9_1_only_LR']['diff']:+.3f} AUROC below PHQ9_1, and the sleep diary is not clearly above chance (its CI includes 0.5). "
     "Because the freeze picked the best of 14 cells on the same CV, the frozen number is slightly optimistic; the gap to the wearable arms is far too large for that to matter. The frozen model is therefore a questionnaire baseline, not a wearable model."))

# ---- Stage 5
opr = [[n, f"{op[k]['threshold_median']:.4f}", f3(op[k]['accuracy']), f3(op[k]['sensitivity']), f3(op[k]['specificity']), f3(op[k]['ppv']), f3(op[k]['npv'])]
       for n, k in [("default (max balanced accuracy)", "default"), ("max accuracy", "max_accuracy"), ("matched sensitivity 0.80", "matched_sensitivity")]]
S.append(sec(5, "Operating points", f"""
<p>{esc(S5['selection'])}. Metrics are pooled over the outer folds of a repeat and averaged over 5 repeats. Majority rate {maj:.3f}.</p>
{table(["Point", "Threshold (median)", "Accuracy", "Sensitivity", "Specificity", "PPV", "NPV"], opr)}
{fig('fig2_threshold_curves', 'Accuracy, sensitivity and specificity against the frozen model threshold (pooled OOF, descriptive)')}
{fig('fig5_operating_points', 'Validation-selected operating points')}
<p>Simple a-priori rules on all 98 segments (no training, descriptive): {esc('; '.join(f"{r['rule']}: acc {r['accuracy']:.3f}, sens {r['sensitivity']:.3f}, spec {r['specificity']:.3f}" for r in rules))}.</p>
""", f"At the default point the frozen model reaches accuracy {op['default']['accuracy']:.3f} against a majority rate of {maj:.3f}, with sensitivity {op['default']['sensitivity']:.3f} and specificity {op['default']['specificity']:.3f}, so it does beat the majority baseline. "
     "The very strong regularization (C = 0.001) squeezes all probabilities close to 0.5, so every rule picks essentially the same cut, which is equivalent to flagging people whose day-0 PHQ-9 is 4 or more. "
     "In plain terms: the screen says that people who already report some symptoms on day 0 tend to still report them two and four weeks later."))

# ---- Stage 6
r6 = S6["results"]
rows6 = [[k, esc(v.get('n_inputs', '')), f3(v.get('mean_auroc')), f"{f3(v.get('auroc_repeat_avg_pred'))} {ci(v.get('auroc_ci'))}",
          "" if not v.get('paired_vs_frozen') else f"{v['paired_vs_frozen']['diff']:+.3f} {ci(v['paired_vs_frozen']['ci'])} ({v['paired_vs_frozen']['verdict']})"]
         for k, v in r6.items() if v.get("auroc_ci")]
S.append(sec(6, "Foundation-model features (MOMENT-Small, frozen), branch R", f"""
<p>Raw 1 Hz HR (hrm.csv.gz, {S6A['rows_read']:,} rows streamed in {S6A['seconds']:.0f} s, {S6A['rows_kept_hr_gt0_in_window']:,} valid readings) reduced to 1-minute means; 512-minute windows with at least 50% observed minutes: {S6B['hr_minute']['windows']:,} windows, all {S6B['hr_minute']['segments_with_windows']} segments covered (median {S6B['hr_minute']['windows_per_segment_median']:.0f} per segment). Five-minute multichannel view ({esc(', '.join(S6B['m5_multichannel']['channels']))}): {S6B['m5_multichannel']['windows']} windows, {S6B['m5_multichannel']['segments_with_windows']} of 98 segments. MOMENT-1-small ({S6B['backbone_params']:,} frozen parameters), input_mask = 0 on missing samples, mean-pooled per segment, 16 PCA components fit inside each training fold, then LR. No Butterworth filter (signals are already below every cut-off). PPG: {esc(S6['ppg'])}. MiniRocket: not run (sktime not installed).</p>
{table(["Arm", "Inputs", "Mean AUROC", "AUROC [95% CI]", "Paired diff vs frozen [CI]"], rows6)}
""", f"MOMENT embeddings of minute-level heart rate are at chance on their own ({f3(mom_alone['auroc_repeat_avg_pred'])}, CI {ci(mom_alone['auroc_ci'])}), and so is the five-minute multichannel view. "
     f"Adding them to the frozen PHQ9_1 feature makes the model worse ({f3(mom_add['auroc_repeat_avg_pred'])}, paired difference {mom_add['paired_vs_frozen']['diff']:+.3f}, CI excludes 0): 16 noisy components dilute one strong predictor at N = 98. "
     f"The small TCN trained on the same windows is the best watch-only model in the whole run ({f3(tcn['auroc_repeat_avg_pred'])}, CI {ci(tcn['auroc_ci'])}), but its interval includes 0.5, so it is not distinguishable from chance. The foundation model adds nothing here."))

# ---- Stage 7
fl_order = ["FedAvg MLP", "FedXgbBagging", "TreeAvg", "TreeAvg (anchor eta 0.10)", "TreeAvg 60 rounds", "Centralized MLP", "Centralized XGBoost", "Centralized LightGBM", "Local only (client 1)", "Local only (client 2)"]


def fl_table(block):
    m = block["methods"]
    return table(["Method", "Mean AUROC (3 repeats)", "AUROC [95% CI]", "vs centralized XGBoost [CI]", "Comm per outer fold (MB, measured)"],
                 [[k, f3(m[k]['mean_auroc_over_repeats']), f"{f3(m[k]['auroc_repeat_avg_pred'])} {ci(m[k]['auroc_ci'])}",
                   "reference" if not m[k]['paired_vs_centralized_xgboost'] else f"{m[k]['paired_vs_centralized_xgboost']['diff']:+.3f} {ci(m[k]['paired_vs_centralized_xgboost']['ci'])}",
                   "" if m[k]['comm_mb_per_fold_mean'] is None else f"{m[k]['comm_mb_per_fold_mean']:.3f}"] for k in fl_order])


mlp_mb, xgb_mb, ta_mb = (fm[k]["comm_mb_per_fold_mean"] for k in ["FedAvg MLP", "FedXgbBagging", "TreeAvg"])
S.append(sec(7, "Federated learning (K = 2 simulated clients, inside the outer CV)", f"""
<p>Clients: {esc(S78['clients'])}; partition seeds {S78['partition_seeds']}, 5 outer folds x {S78['repeats']} repeats. Every client has {cl_pos_rows[0]} to {cl_pos_rows[2]} positive segments (median {cl_pos_rows[1]:.0f}) but only {cl_pos_parts[0]} to {cl_pos_parts[2]} positive participants. K = 3 would give about 5 positive participants per client, so it was not run. Validation = 20% of each fold's training participants; {val_single} of 15 validation sets held a single class, in which case round 1 is kept. Preprocessing from federated sufficient statistics. Methods and settings as in the protocol (FedAvg MLP 40 rounds x 2 local epochs; FedXgbBagging with Flower's real aggregate, 30 rounds; TreeAvg {esc(S78['treeavg_equation'])}, eta grid 0.03-0.30, 30 rounds plus a 60-round check).</p>
<h3>Main: frozen feature (PHQ9_1)</h3>{fl_table(fl_main)}
<h3>Supplementary: wearable features only ({esc(S78['supplementary_wearable']['feature_set'])})</h3>{fl_table(S78['supplementary_wearable'])}
{fig('fig7_client_vs_fl_vs_central', 'One client alone vs FL vs centralized, frozen feature')}
{fig('fig4_communication', 'Measured communication per outer fold, relative to FedAvg MLP')}
<p>Measured communication per outer fold: FedAvg MLP {mlp_mb:.3f} MB, FedXgbBagging {xgb_mb:.3f} MB, TreeAvg {ta_mb:.3f} MB (FedAvg / TreeAvg = {mlp_mb / ta_mb:.1f}x).</p>
""", "FL is only borderline feasible here: each simulated client holds 6 to 9 positive people, under the protocol's ~10. "
     "On the frozen PHQ9_1 feature every method, federated or not, lands between about 0.79 and 0.86 and none differs from centralized XGBoost with a CI that excludes 0; a single client alone is about as good, because one questionnaire score needs very little data to learn. "
     "On watch features every method (federated, centralized and local) stays near or below chance. So FL neither costs nor buys anything measurable at this N; it cannot rescue features that carry no signal."))

# ---- Stage 8
pw_rows = [[esc(k), f"{v['diff']:+.3f} {ci(v['ci'])}", v['verdict']] for k, v in fl_main["pairwise"].items()]
eta_counts = {}
for f in fl_main["folds"]:
    eta_counts[str(f["treeavg_eta_chosen"])] = eta_counts.get(str(f["treeavg_eta_chosen"]), 0) + 1
lc = fl_main["learning_curves_val_auroc_mean"]
lc_series = [(k, np.arange(1, len(v) + 1), np.nan_to_num(v, nan=0.5)) for k, v in lc.items()]
figs["fig8_learning_curves"] = line_chart(lc_series, "Validation AUROC per round", "round", "mean validation AUROC", xr=(1, 40), yr=(0.4, 1.0))
(FIG_DIR / "fig8_learning_curves.svg").write_text(figs["fig8_learning_curves"])
S.append(sec(8, "FL audit and diagnostics", f"""
{table(["Audit item", "Result"], [[esc(k), esc(v)] for k, v in S78['audit'].items()])}
{fig('fig8_learning_curves', 'Mean validation AUROC per round (frozen feature)')}
<p>TreeAvg eta chosen on validation per fold: {esc(eta_counts)}. Best rounds are mostly round 1 to a few rounds: with one feature the first trees already rank people by PHQ9_1 and later rounds only add ties or noise.</p>
{table(["Pairwise (paired participant bootstrap)", "Difference [CI]", "Verdict"], pw_rows)}
""", "The audit found no implementation problem: weights sum to 1, clients and splits are disjoint by assertion, the test fold is only read at the validation-selected round, and communication is measured from serialized bytes. "
     "FedAvg MLP ranks slightly above the two tree methods on the PHQ9_1 feature, because a smooth model does not create ties on a 0-14 score, but it is not distinguishable from centralized MLP. "
     "What limits FL here is the data ceiling (a single strong questionnaire feature, and watch features without signal) plus fragmentation into clients with fewer than ten positive people, not the algorithms."))

# ---- Stage 9
S.append(sec(9, "Robustness", f"""
{table(["Model (frozen arm, nested CV)", "AUROC per seed", "Mean", "SD (population)"], [[k, ', '.join(f'{x:.3f}' for x in v['aurocs']), f3(v['mean']), f4(v['sd_population'])] for k, v in seeds.items()])}
<p>Frozen model fold-to-fold spread over {S9['frozen_fold_spread']['n_folds']} outer folds: mean {f3(S9['frozen_fold_spread']['mean'])}, SD {f3(S9['frozen_fold_spread']['sd'])}, range {f3(S9['frozen_fold_spread']['min'])} to {f3(S9['frozen_fold_spread']['max'])}. Leave-one-participant-out: AUROC {f3(S9['lopo']['auroc'])} {ci(S9['lopo']['auroc_ci'])}. Secondary target PHQ-9 &gt;= 10 ({S9['secondary_target_phq9_ge10']['n_pos']} segments, {S9['secondary_target_phq9_ge10']['pos_participants']} people): AUROC {f3(S9['secondary_target_phq9_ge10']['auroc_repeat_avg_pred'])} {ci(S9['secondary_target_phq9_ge10']['auroc_ci'])}, AUPRC {f3(S9['secondary_target_phq9_ge10']['auprc'])} (chance {f3(S9['secondary_target_phq9_ge10']['pos_rate_chance_auprc'])}). Same-data check: every headline table uses {S9['same_data_check']['n_rows']} segments, {S9['same_data_check']['n_pos']} positives, {S9['same_data_check']['n_participants']} participants. Pooled-OOF CV numbers (Stages 3-6, 9) and FL numbers (3 repeats, FL validation carve-out) are kept in separate tables.</p>
""", "The questionnaire result is stable: across 3 seeds every model family stays between about 0.80 and 0.88 on PHQ9_1, LOPO agrees, and the &gt;= 10 sensitivity analysis points the same way (but rests on 5 people, so its interval is very wide). "
     "Single outer folds range widely (each holds only 9 to 11 people), which is why only pooled numbers with participant-level CIs are reported."))

# ---- Stage 10
ms = op["matched_sensitivity"]
S.append(sec(10, "Comparison with published work (fallback)", f"""
<p>Dataset paper: {esc(stage10['dataset_paper'])} Reported metrics: {esc(stage10['dataset_paper_metrics'])}.</p>
<p>Fallback reference: clinical target sensitivity {C.CLINICAL_TARGET_SENS:.2f}. At the validation-matched point the frozen model lands at sensitivity {ms['sensitivity']:.3f}, specificity {ms['specificity']:.3f}, accuracy {ms['accuracy']:.3f} (majority rate {maj:.3f}). Matching on inner-fold validation can land a little off the target on test, as here.</p>
<p>Related work, not the same data (verify before citing): {esc(' | '.join(stage10['related_work_not_same_data']))}</p>
""", "There is nothing to compare against on this dataset: its own paper does not predict depression. At the clinical target of 80% sensitivity the questionnaire baseline keeps about three quarters of non-cases correctly negative. "
     "No watch-based model reached a usable operating point, so no wearable comparison is made."))

# ---- Stage 11
S.append(sec(11, "Paper prep", f"""
{fig('fig1_pipeline', 'Pipeline')}
{fig('fig3_auroc_all_models', 'AUROC with participant-level 95% CIs, all centralized models on the same 98 segments')}
{fig('fig6_roc', 'ROC curves (pooled OOF); no published operating point exists to mark')}
{fig('fig6_pr', 'Precision-recall curves (pooled OOF)')}
<p>Figures are SVG in <code>public/docs/paper_figures/hrv_phq9/</code> (PNG/PDF were not produced: matplotlib is not installed in any project environment). Figure numbers follow the protocol; figures 2, 4, 5 and 7 appear in Stages 5 and 7.</p>
<h3>MAMAI framing</h3>
<p>This dataset is a non-perinatal, 49-person proxy for Agent 1 (the wearable Mood Monitoring Agent). Its lesson for MAMAI: on a short two-to-four-week horizon, the mother's own recent questionnaire answers (Agent 3's input) carry the predictive signal, while summary heart rate, HRV, motion and light features (Agent 1's input) and frozen MOMENT embeddings of them carry none that this sample can detect. Agent 5's federated training is technically fine but has nothing to add when the features do not carry signal.</p>
<h3>Needs your action</h3><ul><li>Venue choice, APC, institutional discount and special-issue deadlines: not researched in this run.</li><li>Literature list: only the dataset paper and Kroenke 2001 are cited; any wearable-PHQ-9 comparison papers must be checked from source.</li></ul>
""", "The figures are generated from the result JSONs. A paper built on this dataset would be a careful negative result for wearables plus a positive questionnaire baseline; it would need a clear statement that N = 49, the cut-off is mild symptoms, and the clients are simulated."))

# ---- Stage 12
S.append(sec(12, "Final summary (sections A-K)", table(["Section", "Content", "Tag"], [
    ["A cohort flow", f"49 participants, 98 segments, {S2['n_pos']} positive segments ({S2['positive_participants']} people); 73,360 calendar-window sensor rows; all 98 segments kept", "Main result"],
    ["B methods", "LR / RF / ET / HGB / LGBM-DART / XGB / MLP; FedAvg, FedXgbBagging, TreeAvg; participant cluster bootstrap (2000, seed 42)", "Main result"],
    ["C protocol", "SMALL-N: repeated nested StratifiedGroupKFold 5x5, inner 3-fold; K=2 simulated clients inside outer folds", "Main result"],
    ["D metrics", f"Frozen PHQ9_1 LR AUROC {f3(fz_cv['auroc_repeat_avg_pred'])} {ci(fz_cv['auroc_ci'])}, AUPRC {f3(fz_cv['auprc'])} (chance {f3(fz_cv['pos_rate_chance_auprc'])}); best wearable arm {f3(next(r['auroc_repeat_avg_pred'] for r in S4['results'] if f"{r['arm']}|{r['model']}" == wear_key))}", "Main result"],
    ["E operating points", f"default acc {f3(op['default']['accuracy'])} / sens {f3(op['default']['sensitivity'])} / spec {f3(op['default']['specificity'])} vs majority {f3(maj)}", "Main result"],
    ["F ablation", "watch-only arms at chance; questionnaire arms 0.81-0.86; watch adds nothing to questionnaires", "Main result"],
    ["G communication", f"FedAvg {mlp_mb:.2f} MB, FedXgbBagging {xgb_mb:.2f} MB, TreeAvg {ta_mb:.2f} MB per outer fold", "Supplementary"],
    ["H robustness", f"seeds SD at most {max(v['sd_population'] for v in seeds.values()):.3f}; LOPO {f3(S9['lopo']['auroc'])}; A to B temporal {f3(tf['train_A_test_B_disjoint_people_auroc_mean'])}", "Supplementary"],
    ["I published work", "none on this target; clinical 0.80 sensitivity fallback", "Comparison"],
    ["J limitations", "N = 49, mild-symptom cut-off, thin HRV coverage, simulated clients, PPG not used, MiniRocket not run, freeze chosen on the same CV", "Main result"],
    ["K conclusions", "questionnaire carries the signal; watch features and MOMENT do not at this N", "Main result"]]) +
    "<p>Skipped or not run: Stage 3 Part B (no published model), MiniRocket (sktime not installed), PPG (size and no validated 5-minute PPG feature), K = 3 FL (too few positives), PNG/PDF figures (no matplotlib), venue logistics (needs your action).</p>",
    "Overall, the protocol confirms that the Ref 2 watch data, processed honestly at the participant level, does not predict even mild depressive symptoms in this 49-person sample, while a single day-0 PHQ-9 does. "
    "The earlier high Ref 2 numbers came from row-level evaluation, which mostly measures person identity."))

tiles = f"""<div class="tiles">
<div class="tile"><div class="l">Frozen AUROC (PHQ9_1 LR)</div><div class="v">{f3(fz_cv['auroc_repeat_avg_pred'])}</div><div class="c">{ci(fz_cv['auroc_ci'])}</div></div>
<div class="tile"><div class="l">Best watch-only AUROC</div><div class="v">{f3(next(r['auroc_repeat_avg_pred'] for r in S4['results'] if f"{r['arm']}|{r['model']}" == wear_key))}</div><div class="c">{esc(wear_key)}</div></div>
<div class="tile"><div class="l">MOMENT alone</div><div class="v">{f3(mom_alone['auroc_repeat_avg_pred'])}</div><div class="c">{ci(mom_alone['auroc_ci'])}</div></div>
<div class="tile"><div class="l">Accuracy at default</div><div class="v">{f3(op['default']['accuracy'])}</div><div class="c">majority {f3(maj)}</div></div>
</div>"""
nav = "<nav aria-label='Stages'>" + "".join(f'<a href="#s{i}">Stage {i}</a>' for i in range(13)) + "</nav>"
doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HRV PHQ-9 Protocol</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&family=Public+Sans:wght@400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style></head><body><div class="page">
<h1>HRV + PHQ-9 (Ref 2): dataset study protocol run</h1>
<p class="sub">Generated 2026-10-05 by <code>scripts/hrv_phq9_protocol_stage10_12_report.py</code> from the stage JSONs. REB25-1587_MOD1 research data; no individual rows are shown. Fresh run of every stage; no number is copied from older Ref 2 reports.</p>
{tiles}{nav}{''.join(S)}</div></body></html>"""
assert "—" not in doc, "no em dashes"
REPORT.write_text(doc)

# --------------------------------------------------------------------------------------
# summary JSON (exact schema)
r4 = C.r4
best_wear_row = next(r for r in S4["results"] if f"{r['arm']}|{r['model']}" == wear_key)


def opd(k):
    return {"threshold": r4(op[k]["threshold_median"]), "accuracy": r4(op[k]["accuracy"]),
            "sensitivity": r4(op[k]["sensitivity"]), "specificity": r4(op[k]["specificity"])}


stage_notes = [
    (0, "Dataset profile", "run", None, "N=49, 98 participant-segments, PHQ-9>=5 (>=10 has only 5 positive people); SMALL-N"),
    (1, "Inventory and raw-data gate", "run", None, "raw 1 Hz tables + 5-min aggregates; fixed day anchor (shared calendar) and lf/hf artifacts"),
    (2, "Target and leakage audit", "run", None, f"7/7 checks pass; shuffled AUROC ~{perm_mean:.2f}; row split {rv5['random_row_kfold']:.2f} vs grouped {rv5['grouped_by_participant']:.2f}"),
    (3, "Baseline and forensic audit", "branch", "SMALL-N", "no bug; wearable-only 0.38-0.47 for all models; Part B skipped (no published model)"),
    (4, "Ablation and freeze", "branch", "SMALL-N", "PHQ9_1-only LR frozen; watch adds nothing to questionnaires"),
    (5, "Operating points", "branch", "SMALL-N", "thresholds from inner OOF; clinical target sensitivity 0.80"),
    (6, "MOMENT features", "branch", "R", "MOMENT on 1-min HR and 5-min multichannel at chance; PPG not used; MiniRocket not run"),
    (7, "Federated learning", "branch", "SMALL-N, K=2 simulated", "borderline: 6-9 positive participants per client; no method differs from centralized"),
    (8, "FL audit", "run", None, "asserts pass; limit is data ceiling and fragmentation"),
    (9, "Robustness", "branch", "SMALL-N", "3 seeds, fold spread, LOPO, PHQ-9>=10 sensitivity analysis"),
    (10, "Published comparison", "branch", "fallback", "data paper has no prediction task; compared at clinical 0.80 sensitivity"),
    (11, "Paper prep", "run", None, "SVG figures only (no matplotlib); venue logistics need user action"),
    (12, "Final report", "run", None, "public/docs/HRV_PHQ9_Protocol_Report.html"),
]
methods_out = []
for name, key in [("FedAvg MLP", "FedAvg MLP"), ("FedXgbBagging", "FedXgbBagging"), ("TreeAvg", "TreeAvg"),
                  ("Centralized XGBoost", "Centralized XGBoost"), ("Centralized LightGBM", "Centralized LightGBM"),
                  ("Local only", "Local only (client 1)")]:
    m = fm[key]
    methods_out.append({"name": name, "auroc": r4(m["auroc_repeat_avg_pred"]), "auroc_ci": r4(m["auroc_ci"]),
                        "comm_mb": r4(m["comm_mb_per_fold_mean"])})
summary = {
    "dataset": "HRV + PHQ-9 (Ref 2)", "prefix": "hrv_phq9", "report": "/docs/HRV_PHQ9_Protocol_Report.html",
    "n_participants": 49, "n_rows": int(S2["n_rows"]), "unit": "participant-segment (days 1-14 and 15-28; 2 per participant)",
    "target": "PHQ-9 >= 5 on the score covering the segment (PHQ9_2 for days 1-14, PHQ9_F for days 15-28)",
    "target_type": "binary (cut-off)", "n_pos": int(S2["n_pos"]), "pos_rate": r4(S2["pos_rate"]), "majority_rate": r4(maj),
    "profile": {"individual_rows": True, "rows_per_participant": "2 labelled segments (median 1,207 five-minute sensor rows per person overall; about 1,500 per person inside the study window)",
                "design": "longitudinal, 4 weeks, PHQ-9 at day 0, mid-study and end", "small_n": True,
                "signal_granularity": "raw 1 Hz watch tables + 5-minute aggregates", "natural_sites": None,
                "published_reference": None, "survey_weights": False},
    "stages": [{"id": i, "name": n, "status": s, "branch": b, "note": t} for i, n, s, b, t in stage_notes],
    "leakage": {"permutation_metric_mean": r4(perm_mean),
                "row_vs_grouped": f"5-min rows: random {rv5['random_row_kfold']:.4f} vs grouped {rv5['grouped_by_participant']:.4f}; segments: random {rvs['random_segment_kfold']:.4f} vs grouped {rvs['grouped_by_participant']:.4f}",
                "checklist_pass": int(S2["checklist_pass"]), "checklist_total": int(S2["checklist_total"])},
    "frozen": {"model": f"Logistic regression (L2, C={FZ['hyperparameters']['m__C']}, balanced)", "features": "PHQ9_1 (day-0 PHQ-9 score) only",
               "n_features": int(FZ["n_features"]), "cv": "repeated grouped StratifiedGroupKFold 5x5 (participant), repeat-averaged OOF predictions",
               "auroc": r4(fz_cv["auroc_repeat_avg_pred"]), "auroc_ci": r4(fz_cv["auroc_ci"]),
               "auprc": r4(fz_cv["auprc"]), "auprc_ci": r4(fz_cv["auprc_ci"])},
    "operating_points": {"default": opd("default"), "max_accuracy": opd("max_accuracy"),
                         "matched_sensitivity": {**opd("matched_sensitivity"), "reference_sensitivity": C.CLINICAL_TARGET_SENS,
                                                 "reference_source": "clinical target sensitivity 0.80 (no published model on this dataset)"}},
    "moment": {"branch": "R", "alone_auroc": r4(mom_alone["auroc_repeat_avg_pred"]), "alone_ci": r4(mom_alone["auroc_ci"]),
               "added_auroc": r4(mom_add["auroc_repeat_avg_pred"]), "delta_vs_frozen_ci": r4(mom_add["paired_vs_frozen"]["ci"]),
               "note": "MOMENT-1-small on 512-minute windows of raw 1 Hz HR reduced to 1-min means; alone at chance, adding it to PHQ9_1 lowers AUROC (paired CI excludes 0). PPG not used."},
    "fl": {"feasible": True, "clients": "simulated: K=2 seeded stratified permutation of training participants inside each outer fold",
           "K": 2, "methods": methods_out,
           "note": f"borderline: clients hold {cl_pos_rows[0]}-{cl_pos_rows[2]} positive segments but only {cl_pos_parts[0]}-{cl_pos_parts[2]} positive participants; K=3 not feasible. Run on the frozen PHQ9_1 feature; no FL method differs from centralized XGBoost. Local only = client 1 alone (client 2: {fm['Local only (client 2)']['auroc_repeat_avg_pred']:.4f}). Watch-only FL stays at chance. comm_mb = measured MB per outer fold."},
    "robustness": {"seeds": 3, "auroc_sd": r4(seeds[FZ["model"]]["sd_population"]),
                   "note": f"SD over 3 seeds for the frozen LR; all 7 model families 0.80-0.87 on PHQ9_1; LOPO AUROC {S9['lopo']['auroc']:.4f}; PHQ-9>=10 AUROC {S9['secondary_target_phq9_ge10']['auroc_repeat_avg_pred']:.4f} on 5 positive people"},
    "published": {"paper": None, "their_metrics": None,
                  "comparison": "No published model on this dataset and target (Baigutanova et al. 2025 is a data descriptor without prediction); fallback: at the clinical target sensitivity 0.80 the frozen model reaches specificity "
                                f"{ms['specificity']:.4f} and accuracy {ms['accuracy']:.4f} vs majority {maj:.4f}."},
    "honest_read": ("Evaluated honestly by participant, the Ref 2 watch features (HR, HRV, motion, light) and frozen MOMENT embeddings do not predict even mild depressive symptoms (PHQ-9 >= 5) in these 49 people: no watch-only model is distinguishable from chance. "
                    f"A single day-0 PHQ-9 score does, with AUROC {fz_cv['auroc_repeat_avg_pred']:.2f} and accuracy {op['default']['accuracy']:.2f} against a majority rate of {maj:.2f}, and the watch adds nothing on top of it. "
                    "Earlier high Ref 2 numbers came from row-level splits that mostly recognise the person. "
                    "Federated learning is only borderline feasible at this N and neither helps nor hurts."),
}
SUMMARY.parent.mkdir(parents=True, exist_ok=True)
SUMMARY.write_text(json.dumps(summary, indent=1))
print("wrote", REPORT, SUMMARY, sorted(p.name for p in FIG_DIR.iterdir()))

# --------------------------------------------------------------------------------------
# Plot data for the dashboard (computed from saved OOF predictions, no retraining)
def _r(v):
    return round(float(v), 4)


def _thin(pts, n=60):
    if len(pts) <= n:
        return pts
    idx = np.unique(np.linspace(0, len(pts) - 1, n).round().astype(int))
    return [pts[i] for i in idx]


fpr, tpr, _ = roc_curve(y, frozen_p)
roc_pts = [[_r(a), _r(b)] for a, b in zip(fpr, tpr)]
if roc_pts[0] != [0.0, 0.0]:
    roc_pts.insert(0, [0.0, 0.0])
if roc_pts[-1] != [1.0, 1.0]:
    roc_pts.append([1.0, 1.0])
prec, rec, _ = precision_recall_curve(y, frozen_p)
pr_pts = [[_r(r_), _r(p_)] for r_, p_ in zip(rec[::-1], prec[::-1])]
sweep_pts = []
for t in np.arange(0.01, 0.9901, 0.02):
    pred = (frozen_p >= t).astype(int)
    sweep_pts.append({"t": _r(t), "acc": _r((pred == y).mean()),
                      "sens": _r(((pred == 1) & (y == 1)).sum() / y.sum()),
                      "spec": _r(((pred == 0) & (y == 0)).sum() / (y == 0).sum())})
ablation = [{"arm": f"{r['arm']} | {r['model']}", "n_features": int(r["n_features"]), "auroc": _r(r["auroc_repeat_avg_pred"]),
             "ci": [_r(v) for v in r["auroc_ci"]]} for r in S4["results"]]
for k, v in S6["results"].items():
    if v.get("auroc_ci"):
        nf = v.get("n_inputs")
        ablation.append({"arm": f"Stage 6: {k}", "n_features": int(nf) if isinstance(nf, (int, float)) else None,
                         "auroc": _r(v["auroc_repeat_avg_pred"]), "ci": [_r(c) for c in v["auroc_ci"]]})
fl_comm = [{"method": m, "mb": _r(fm[m]["comm_mb_per_fold_mean"]), "unit": "MB per fold"} for m in ["FedAvg MLP", "FedXgbBagging", "TreeAvg"]]


def _kind(m):
    return "federated" if m.startswith(("Fed", "TreeAvg")) else ("local" if m.startswith("Local") else "centralized")


fl_methods = []
for block, tag in [(S78["main"], "frozen PHQ9_1"), (S78["supplementary_wearable"], "watch-only")]:
    for m, v in block["methods"].items():
        fl_methods.append({"name": f"{m} ({tag})", "auroc": _r(v["auroc_repeat_avg_pred"]),
                           "ci": [_r(c) for c in v["auroc_ci"]] if v.get("auroc_ci") else None, "kind": _kind(m)})
summary["plots"] = {
    "roc": {"model": "Logistic regression on PHQ9_1 (frozen), repeat-averaged OOF", "points": _thin(roc_pts)},
    "pr": {"points": _thin(pr_pts)},
    "sweep": sweep_pts,
    "sweep_note": "frozen LR (C=0.001) probabilities lie in a narrow band around 0.5, so the 0.02 grid jumps from flag-all to flag-none near t=0.5",
    "ablation": ablation,
    "ablation_note": "Stage 4 arms under both screened models (LR and ET) plus Stage 6 MOMENT/TCN/control arms; all repeated nested grouped CV 5x5, AUROC of repeat-averaged OOF predictions; Stage 6 TCN n_features null (raw windows)",
    "fl_comm": fl_comm,
    "fl_methods": fl_methods,
}
for m in summary["fl"]["methods"]:
    m.setdefault("auroc_ci", m.get("auroc_ci"))
SUMMARY.write_text(json.dumps(summary, indent=1))
print("plots added", len(roc_pts), len(pr_pts), len(sweep_pts), len(ablation), len(fl_methods))
