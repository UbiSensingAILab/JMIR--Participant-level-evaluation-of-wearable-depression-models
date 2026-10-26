#!/usr/bin/env python3
"""Figures 1-4 for the v9 manuscript: Times New Roman, one 300-dpi PNG each, in ~/Desktop/JMIR_submission/v9_figures.

Plotting only. Figure 2 and 4 data come from the saved CSVs; Figure 3 from T9_calibration_deciles.csv
(written by scripts/revision/final_fig3_deciles.py, which asserts the refit reproduces T9_calibration.csv)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
OUTD = Path.home() / "Desktop/JMIR_submission/v9_figures"
REV = Path(__file__).resolve().parents[3] / "results" / "revision"
OUTD.mkdir(exist_ok=True)


def save(fig, name, width_in=None, **kw):
    """Save at 300 dpi; if width_in is given, scale the tight bounding box to that width at 300 dpi."""
    dpi = 300
    if width_in:
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
        dpi = 300 * width_in / bb.width
    fig.savefig(OUTD / name, dpi=dpi, facecolor="white", **kw)
    plt.close(fig)
    im = Image.open(OUTD / name)
    im.load()
    im.save(OUTD / name, dpi=(300, 300))


# ------------------------------------------------------------------ Figure 1: workflow
INK, MUTED, LINE = "#1f2933", "#52606d", "#9aa5b1"
ACC, ACC_BG, DS_BG, OUT_BG, OUT = "#2f5d8a", "#e8f0f8", "#f4f5f7", "#eaf5ee", "#2e7d4f"
fig, ax = plt.subplots(figsize=(12, 6.9))
ax.set_xlim(0, 120)
ax.set_ylim(4, 73)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=lw))


def num(x, y, n, c=ACC):
    ax.add_patch(Circle((x, y), 1.55, fc=c, ec="none", zorder=5))
    ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=11, fontweight="bold", zorder=6)


def step(x, y, w, h, n, title, body, fc=ACC_BG, ec=ACC, c=ACC):
    box(x, y, w, h, fc, ec)
    num(x + 2.3, y + h - 2.3, n, c)
    ax.text(x + 4.6, y + h - 2.3, title, ha="left", va="center", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(x + 1.6, y + h - 5.0, body, ha="left", va="top", fontsize=9.6, color=MUTED, linespacing=1.35)


def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, color=LINE, lw=1.4, shrinkA=0, shrinkB=0))


ax.text(2, 71.0, "Public wearable depression data sets", fontsize=12, fontweight="bold", color=INK)
ds = [("PSYCHE-D", "4036 participants, 10,866 participant-quarters", "Fitbit sleep and steps summaries", "Outcome: PHQ-9 category worsens"),
      ("HRV and PHQ-9", "49 participants, 98 two-week segments", "Galaxy Watch: 1 Hz HR, HRV, accelerometer", "Outcome: PHQ-9 ≥5"),
      ("RESILIENT", "73 older adults", "Withings watch HR, steps; sleep sensor", "Outcome: PHQ-9 ≥10")]
for i, (t, a, b, c) in enumerate(ds):
    x = 2 + i * 39.3
    box(x, 57.5, 36.5, 11.0, DS_BG, LINE)
    ax.text(x + 1.5, 66.3, t, fontsize=11.5, fontweight="bold", color=INK)
    ax.text(x + 1.5, 63.6, a + "\n" + b + "\n" + c, fontsize=9.6, color=MUTED, va="top", linespacing=1.35)

arrow(60, 57.0, 60, 54.4)
ax.text(2, 52.6, "Standardized 9-step protocol, participant-level evaluation", fontsize=12, fontweight="bold", color=INK)
w2 = 27.2
core = [(1, "Profile and inventory", "Data set profile, small-N branch,\nraw-data checks"),
        (2, "Leakage audit", "Record vs participant split,\nlabel-shuffle test, checklist"),
        (3, "Baseline model screen", "LR, RF, ET, HGB, LightGBM,\nXGBoost, MLP; class weights"),
        (4, "Timing-aware ablation", "Baseline PHQ-9, screener, lifestyle\nand medication, Fitbit; freeze\nprotocol model")]
for i, (n, t, b) in enumerate(core):
    x = 2 + i * (w2 + 2.4)
    step(x, 38.8, w2, 11.2, n, t, b)
    if i < 3:
        arrow(x + w2 + 0.5, 44.4, x + w2 + 2.1, 44.4)

arrow(60, 38.3, 60, 36.9)
ax.text(2, 35.1, "Extensions tested on the protocol model", fontsize=12, fontweight="bold", color=INK)
w3 = 37.0
ext = [(5, "Operating points", "Default, maximum-accuracy, and matched-\nsensitivity thresholds; calibration"),
       (6, "Foundation model", "Frozen MOMENT-1-small embeddings,\nalone and added to protocol inputs"),
       (7, "Federated learning", "FedAvg, FedXgbBagging, TreeAvg;\naccuracy and bytes communicated")]
for i, (n, t, b) in enumerate(ext):
    step(2 + i * (w3 + 2.2), 22.5, w3, 10.0, n, t, b)

arrow(60, 22.0, 60, 19.4)
step(2, 8.5, 56, 10.0, 8, "Robustness", "3 seeds, leave-one-participant-out (small N),\ntemporal splits; 2000 participant-level bootstrap CIs")
step(62, 8.5, 56, 10.0, 9, "Comparison with published work", "Same-sensitivity comparison with the published\nPSYCHE-D model; comparison across 3 data sets",
     fc=OUT_BG, ec=OUT, c=OUT)
save(fig, "Figure1.png", width_in=7.5, bbox_inches="tight")

# ------------------------------------------------------------------ Figure 2: timing-aware ablation forest plot
LABELS = ["Screener added to baseline PHQ-9", "Fitbit added to P+S", "Lifestyle and medication added to P+S", "Fitbit added to P+S+L",
          "Protocol model vs baseline PHQ-9", "Protocol model without medication-change items",
          "Protocol model without 6 treatment-change items"]
pr = pd.read_csv(REV / "R2_R3_R4_paired_differences.csv")
pr = pr[pr.task == "R2"].reset_index(drop=True)
assert len(pr) == len(LABELS) and pr.comparison[0].startswith("(P+S) - P") and "6 treatment" in pr.comparison[6]
rows = [dict(label=LABELS[k], delta=r.delta, ci=json.loads(r.ci)) for k, r in pr.iterrows()]
fig, ax = plt.subplots(figsize=(7.0, 3.4))
for i, r in enumerate(rows[::-1]):
    ax.errorbar(r["delta"], i, xerr=[[r["delta"] - r["ci"][0]], [r["ci"][1] - r["delta"]]], fmt="o", color="#534AB7", capsize=3)
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([r["label"] for r in rows[::-1]], fontsize=9.5)
ax.set_xlabel("Paired difference in AUROC (95% CI, participant-level bootstrap)", fontsize=10)
ax.set_title("PSYCHE-D timing-aware ablation", fontsize=11)
ax.tick_params(axis="x", labelsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
save(fig, "Figure2.png")

# ------------------------------------------------------------------ Figure 3: calibration
dec = pd.read_csv(REV / "T9_calibration_deciles.csv")
PANELS = {"Frozen (P+S+L+W, 152)": "Protocol model (152 inputs)", "R3 forecast": "Forecasting model",
          "Development-selected (P+S+L, 41)": "Development-selected model (41 inputs)"}
VER = {"uncalibrated": ("Uncalibrated", "o"), "Platt (validation)": ("Platt scaling", "s")}
fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.1), sharey=True)
for ax, (key, title) in zip(axes, PANELS.items()):
    for ver, (vlab, mk) in VER.items():
        d = dec[(dec.model == key) & (dec.version == ver)].sort_values("decile")
        assert len(d) == 10
        c = d.iloc[0]
        ax.plot(d.mean_predicted, d.observed_rate, marker=mk, ms=3.5, lw=1.1,
                label=f"{vlab}\nCITL {c.citl:.2f}, slope {c.slope:.2f}, Brier {c.brier:.3f}".replace("-", "\u2212"))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{title}\ntest n=1631", fontsize=9.5)
    ax.set_xlabel("Mean predicted probability (decile)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.6, loc="upper left", frameon=True, framealpha=1, edgecolor="none")
axes[0].set_ylabel("Observed worsening rate", fontsize=8.5)
fig.tight_layout()
save(fig, "Figure3.png")

# ------------------------------------------------------------------ Figure 4: federated communication curves
rdf = pd.read_csv(REV / "T5_per_round_communication_seedvary.csv")
colors = {"FedAvg": "#2a6fdb", "FedXgbBagging": "#d9822b", "TreeAvg": "#1a9e5c"}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, (sname, title) in zip(axes, [("P+S+L 41", "Development-selected inputs (P+S+L, 41)"),
                                     ("frozen 152", "Protocol model inputs (P+S+L+W, 152)")]):
    for m, col in colors.items():
        for s in [1, 2, 3]:
            g = rdf[(rdf.input_set == sname) & (rdf.partition == "IID") & (rdf.method == m) & (rdf.partition_seed == s)].sort_values("round")
            ax.plot(g.cum_total_mb, g.val_auroc, color=col, lw=1.4 if s == 1 else 0.7, alpha=1 if s == 1 else 0.45,
                    label=m if s == 1 else None, marker="o" if s == 1 else None, ms=2)
    ax.axvline(2.16, color="0.4", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Cumulative communication, MB (log scale)", fontsize=9)
    ax.tick_params(labelsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Validation AUROC", fontsize=9)
axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
fig.text(0.01, 0.01, "Bold: partition seed 1; faint: seeds 2 and 3. 5 IID clients. Dashed line: 2.16 MB.", fontsize=7.5, color="0.35")
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "Figure4.png")

for i in range(1, 5):
    im = Image.open(OUTD / f"Figure{i}.png")
    print(f"Figure{i}.png", im.size, im.info.get("dpi"), f"{im.size[0] / 300:.2f} x {im.size[1] / 300:.2f} in")
