#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408) — a different institution's REC/IRAS-
# approved protocol (IRAS 321104), independent of this project's own
# REB25-1587_MOD1; do not conflate the two ethics approvals.
"""RESILIENT multimodal, participant-level, BINARY PHQ-9 (>=10) classification.

This is a from-scratch redesign, not an edit of
scripts/train_resilient_multisource_phq9_gds15.py, which is left completely
untouched and still serves as the 4-class HR+tabular baseline. This script
exists specifically to fix the methodological gaps a two-round chat audit
found in that script (documented in public/docs/RESILIENT_Pipeline_Audit.html
sections 1 and 17): no temporal alignment to the PHQ-9 assessment date, HR
used alone, an aggressive external per-window z-score stacked on top of
MOMENT's own RevIN, and a 4-class model whose binary metric was a post-hoc
softmax collapse rather than a natively-trained binary objective.

------------------------------------------------------------------------
WHAT THE REAL DATA SHOWS (measured directly against the raw RESILIENT
files before any of the design choices below were made — see the dated
investigation this script's authoring session ran; every number here is
reproducible by rerunning the same pandas queries against
research_datasets/old/resilient/):

  - PHQ-9 (`phq_date` in Demographics.csv) is a BASELINE/enrollment
    assessment, not something given partway through monitoring. Across all
    73 participants, the median `phq_date - earliest_recorded_timestamp`
    is 0 days for HR, Sleep_physio, and Sleep_state, and 12-43 of 73
    participants (depending on modality) have `phq_date` STRICTLY BEFORE
    their first recorded timestamp (by up to 20 days). There is almost no
    usable pre-assessment data for most of this cohort.
  - The median `latest_recorded_timestamp - phq_date` is 135-182 days
    (mean), i.e. the overwhelming majority of every participant's
    recording happens AFTER the assessment, not before.
  - Native cadence per modality (measured directly on real timestamps):
    ScanWatch_HR.csv ~600s median gap (continuous, all day); ScanWatch_
    Steps.csv ~3600s median gap (continuous, all day, i.e. natively
    hourly); Sleep_physio.csv/Sleep_state.csv ~60-180s median gap but
    ONLY during real sleep sessions (nocturnal, not all-day).
  - At a candidate window of [phq_date, phq_date + 512 hours) — chosen
    below — real coverage was checked directly: 0/73 participants have
    zero HR rows in-window, 0/73 have zero Steps rows, 0/73 have zero
    Sleep_state segments, and only the single already-known zero-row
    One Sleep_physio participant has none.

WHAT THIS IMPLIES FOR THE DESIGN BELOW (RB = "RESILIENT Binary", this
script's own pre-registered decisions, fixed before any accuracy/AUROC
number is measured, same standing discipline as this project's RM1-RM12/
RH1-RH8 precedents):

RB1 — Temporal window. Day 0 = `phq_date`. The window is
[phq_date, phq_date + 512 hours) — a BOUNDED NEAR-TERM POST-BASELINE
window, not the participant's entire remaining recording (which the RM1
design used, and which can span up to ~200 more days of possible clinical
drift after the assessment). "Before the assessment" was ruled out because
the data above shows there is essentially no usable pre-assessment
recording for most participants. 512 hours = 21.33 days was chosen
because it is an EXACT, zero-padding fit to MOMENT's fixed 512-timestep
context length for the two continuous/all-day channels (HR, Steps), and
is the same order of magnitude as PHQ-9's own 2-week recall window,
versus the old design's up-to-204-day drift risk. This is a real,
disclosed methodological choice, not a proof that 21.3 days is the
"correct" clinical window — it is the most defensible bounded window this
dataset's actual coverage supports.

RB2 — One window per participant, by construction. Unlike RM1's many
non-overlapping windows per participant (mean-pooled afterwards), this
design produces exactly ONE window per participant (the single post-
baseline period above), so no cross-window pooling strategy is needed —
the "simple, defensible aggregation strategy" this script's brief asked
for is: there is nothing to aggregate, by design.

RB3 — Four real per-participant modalities, each represented in the
native shape appropriate to its own real coverage pattern (never forced
onto one grid):
  - HR (ScanWatch_HR.csv): 1 channel, hourly mean bpm, T=512 real hours,
    input_mask all-ones (missing hours are density-filled, see RB5, not
    treated as a shorter real sequence).
  - Steps (ScanWatch_Steps.csv): 1 channel, hourly sum, T=512, same
    treatment as HR.
  - Sleep_physio (Heart Rate during sleep, Respiration Rate, Snoring,
    SDNN_1): 4 channels, ONE VALUE PER REAL NIGHT (mean per sub-signal,
    except Snoring which is summed), because this signal is structurally
    nocturnal-only, not all-day. Real nights (~21-30 typically) are
    left-aligned at positions 0..n_real-1 of a 512-length array; the
    remaining positions are zero-filled AND marked invalid via
    input_mask=0 — MOMENT's own documented mechanism for a real sequence
    shorter than its 512 context length. This is genuinely a much
    smaller real sequence than HR/Steps get, a real limitation of forcing
    a nocturnal-only signal through a 512-context transformer, and is
    reported honestly rather than hidden.
  - Sleep_state (REM/deep/light/wakeup): 4 CONTINUOUS proportion-of-night
    channels (fraction of that night's total scored duration spent in
    each state), not an arbitrary ordinal encoding of a categorical
    variable. Same per-real-night/padded-to-512/input_mask treatment as
    Sleep_physio.
  - Sex / Age group (Demographics.csv): unchanged, static tabular
    features appended to the final representation (never time series).

RB4 — Missing-bin filling vs. sequence-length padding are two DIFFERENT
operations, not confused with each other: an hourly bin with no real HR/
Steps reading (a real device non-wear gap inside an otherwise-covered
512-hour window) is filled with that channel's OWN real-bin mean, the
same "genuinely required, documented" densification RM1 already used —
this is not a normalization step, it is filling a data GAP so a dense
transformer input exists at all. A participant whose channel has fewer
real hours/nights than the threshold below is NOT density-filled — their
whole channel-group embedding is left as NaN and imputed per-fold (RB7),
exactly like RM3's zero-qualifying-window handling.

RB4-correction — the SAME "own real-value mean, never 0.0" fill applies
WITHIN a real night when only one of {Sleep_physio, Sleep_state} recorded
that night (see `build_nightly_group()`), not just to HR/Steps hourly
gaps. This was found missing on a rigorous re-check after this script's
first full run: the shared union-based `input_mask` marked such a night
"valid" while the other source's 4 channels silently held a literal 0.0
— an implausible physiological value passed through as if it were real.
Measured before the fix: 268/1503 (17.8%) of real-night positions across
37/73 participants had this gap. Fixed the same way RB4 already handles
every other missing-bin case, then the full 73-participant run was
repeated — see the Results section below for the corrected numbers.

RB5 — Coverage thresholds for treating a whole channel-group as unusable
for a participant (all-NaN, RB7), rather than density-filling it:
MIN_COVERAGE=0.5 of real hourly bins for HR and Steps (same threshold RM1
already used); at least 1 real night for the Sleep_physio/Sleep_state
group (there is no meaningful "50% of nights" threshold when the real
sequence itself is only ~21-30 long to begin with).

RB6 — Normalization. The RM1 per-window z-score (subtract the window's
own mean, divide by its own std) is REMOVED ENTIRELY, per this redesign's
explicit instruction. MOMENT's own internal RevIN (momentfm/models/
layers/revin.py, invoked inside MOMENTPipeline.embed()) runs unmodified
and is the only normalization the model receives. The RB4 mean-fill for
missing hourly bins is NOT a normalization step — it changes zero real
values into a plausible density-filled value so a dense 512-length input
exists, it does not rescale or re-center anything a real observation
provided. No other new normalization was found to be needed for any
modality here; if one had been, it would be named and justified in this
docstring, not silently added.

RB7 — Missing whole-channel-group imputation. A participant whose HR/
Steps/Sleep-group coverage (RB5) is below threshold contributes an
all-NaN block for that MOMENT-derived embedding (never a NaN-containing
tensor fed into MOMENT itself). SimpleImputer(strategy="mean",
keep_empty_features=True) fills these, fit on the TRAINING FOLD ONLY
inside each CV split — never globally before the split — identical
discipline to RM7.

RB8 — Binary target and training objective. `phq_total >= 10` is the
target, trained as a GENUINE BINARY classifier (`nn.BCEWithLogitsLoss`,
one logit), never a 4-class softmax collapsed afterward. Class imbalance
is handled via `pos_weight = n_neg_train / n_pos_train`, computed from
the TRAINING FOLD's own realized label counts only, inside the same fold
loop that fits the imputer/scaler (RB7) — never from the global 73-
participant label distribution. No focal loss or other imbalance-specific
loss is used, by explicit instruction, so this redesign's architectural/
data changes can be isolated from a loss-function change.

RB9 — Evaluation. Participant-level `StratifiedKFold(n_splits=5,
shuffle=True, random_state=42)` — identical convention to RM11/RM-family
scripts, one row per participant, so no participant can appear in both
train and test within a fold. Every fitted step (SimpleImputer,
MinMaxScaler, pos_weight) is fit on the training fold only. Reported,
pooled out-of-fold: accuracy, sensitivity, specificity, precision (via
the existing, unmodified `confusion_export.confusion_counts`), balanced
accuracy, AUROC, AUPRC, the raw confusion matrix, and n_pos/n_neg — plus
bootstrap 95% CIs (2000 resamples of participants, matching this
project's existing bootstrap-CI convention) for AUROC and AUPRC given how
few positive participants exist.

RB10 — Three configurations compared under the identical RB1-RB9
protocol (temporal window, binary target, no external z-score,
evaluation), so the only thing that differs between them is which
modalities reach the model and how:
  (a) HR-only  — a clean baseline using the SAME new temporal alignment/
      binary target/normalization/evaluation as (b)/(c), NOT the old
      RM1-RM12 script (which differs on every one of those axes and so
      cannot serve as a fair baseline for this comparison). The old
      script is left untouched and still exists as the separate 4-class
      result.
  (b) Multichannel MOMENT — modalities grouped by shared native temporal
      grid (RB3), not one artificial supertensor: {HR, Steps} jointly as
      one [2, 512] MOMENT call (real, all-ones mask); {Sleep_physio x4,
      Sleep_state x4} jointly as one [8, 512] MOMENT call (heavily
      masked, ~21-30 real positions). The two resulting 512-D embeddings
      are concatenated with the 2 tabular features.
  (c) Separate MOMENT per modality — 4 single-source MOMENT calls (HR,
      Steps, Sleep_physio, Sleep_state), each producing its own 512-D
      embedding via MOMENT's own channel+patch mean-pooling, concatenated
      to 2048-D plus the 2 tabular features. This is the data-justified
      granularity (4 real source files), not one channel at a time.

Both (b) and (c) reuse the exact same underlying per-participant channel
arrays and masks built once in `build_channel_tensors()` below — the two
configurations differ only in which MOMENT calls group which channels
together, so the comparison isolates the "joint multichannel vs. separate-
then-concatenate" architectural question from any data-preparation
difference.

RB11 — Opt-in comparison arm (`--external-zscore`), added after this
script's initial publication. This does NOT change the default path
described above (RB6 remains accurate: with the flag absent, no external
per-window normalization is applied, exactly as before). When the flag IS
passed, the RM1-style external per-window z-score
(scripts/train_resilient_multisource_phq9_gds15.py, ~lines 471-473:
`z = (filled - filled.mean()) / max(filled.std(), 1e-6)`) is re-applied to
each per-participant channel array immediately before it reaches MOMENT,
reusing that exact formula and 1e-6 std floor rather than a new variant.
For HR/Steps (already fully density-filled across all 512 hourly bins by
RB4, no zero-padding) the statistics are computed over the whole 512-array.
For the Sleep_physio/Sleep_state nightly channels (RB3), which ARE
zero-padded beyond `n_real_nights`, the mean/std are computed over, and the
transform applied to, ONLY the real (`input_mask==1`) positions of that
channel — folding the zero-padded tail into the statistics would not be
the RM1 treatment and would corrupt the comparison. The padded positions
and the `input_mask` array itself are left untouched either way. Only the
`separate` per-modality configuration (RB10c) is run under this flag, so
the resulting row differs from row B in exactly this one variable; seed,
fold assignment, epoch count and every other RB1-RB9 choice are identical
to the default path. This is an opt-in comparison arm, not a change to
the default behavior described above.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from mood_monitoring_agent.model import MomentRiskPipeline  # noqa: E402
from mood_monitoring_agent.training.train import embed_all_windows  # noqa: E402
from confusion_export import DECISION_THRESHOLD, confusion_counts  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-registered constants (RB1, RB3, RB5, RB8, RB9)
# ---------------------------------------------------------------------------

RAW_DIR = (
    _REPO_ROOT / "research_datasets" / "old" / "resilient"
    / "Sleepmat_Watch_Data" / "Sleepmat_Watch_Data"
)
DEMOGRAPHICS_CSV = _REPO_ROOT / "research_datasets" / "old" / "resilient" / "Demographics.csv"
REPORT_PATH = _REPO_ROOT / "scripts" / "train_resilient_multimodal_binary_phq9_report.md"

EXCLUDE_ENTRIES = {"__MACOSX", ".DS_Store"}
TRACER_SAMPLE_SIZE = 15

WINDOW_HOURS = 512               # RB1 — exact MOMENT context length, no padding for hourly channels
SECONDS_PER_HOUR = 3600.0
MIN_COVERAGE = 0.5               # RB5 — HR/Steps hourly-bin coverage floor
MIN_REAL_NIGHTS = 1              # RB5 — Sleep_physio/Sleep_state floor
D_MODEL = 512                    # MOMENT-1-small embedding dim
N_SPLITS = 5
RANDOM_STATE = 42
PHQ9_POSITIVE_THRESHOLD = 10

HIDDEN_DIM = 16
DROPOUT_P = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
MAX_EPOCHS = 200

SEX_MAP = {"Female": 1, "Male": 0}
AGE_GROUP_MAP = {"[72, 75]": 0, "[76, 87]": 1, "[88, 99]": 2}
SLEEP_STATES = ("REM", "deep", "light", "wakeup")

N_BOOTSTRAP = 2000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def get_participant_uids() -> list[str]:
    """Real 8-char-UID participant directories only, __MACOSX/.DS_Store excluded."""
    uids = sorted(
        p.name for p in RAW_DIR.iterdir()
        if p.is_dir() and p.name not in EXCLUDE_ENTRIES and len(p.name) == 8
    )
    return uids


def load_demographics() -> pd.DataFrame:
    """Sex / Age group / phq_total / phq_date, one row per participant.

    phq_date is stored as a plain DD/MM/YYYY string (SOURCE.md) — parsed
    with dayfirst=True and UTC-localized so it can be diffed directly
    against the UTC ISO-8601 timestamps in the raw per-participant files.
    """
    df = pd.read_csv(DEMOGRAPHICS_CSV)
    df["phq_date"] = pd.to_datetime(df["phq_date"], dayfirst=True, utc=True, errors="coerce")
    return df.set_index("user_id")[["Sex", "Age group", "phq_total", "phq_date"]]


def load_raw_frames(uid: str) -> dict[str, pd.DataFrame | None]:
    """Loads the 4 real per-participant RESILIENT tables, or None if a file
    is absent/empty (the known zero-row Sleep_physio participant)."""
    d = RAW_DIR / uid
    out: dict[str, pd.DataFrame | None] = {}

    hr_path = d / "ScanWatch_HR.csv"
    hr = pd.read_csv(hr_path) if hr_path.exists() else pd.DataFrame()
    if len(hr) > 0:
        hr["ts"] = pd.to_datetime(hr["HR Timestamp"], utc=True, errors="coerce")
        hr = hr.dropna(subset=["ts"])
    out["hr"] = hr if len(hr) > 0 else None

    steps_path = d / "ScanWatch_Steps.csv"
    steps = pd.read_csv(steps_path) if steps_path.exists() else pd.DataFrame()
    if len(steps) > 0:
        steps["ts"] = pd.to_datetime(steps["Steps Timestamp"], utc=True, errors="coerce")
        steps = steps.dropna(subset=["ts"])
    out["steps"] = steps if len(steps) > 0 else None

    physio_path = d / "Sleep_physio.csv"
    physio = pd.read_csv(physio_path) if physio_path.exists() else pd.DataFrame()
    if len(physio) > 0:
        physio["ts"] = pd.to_datetime(physio["Timestamp"], utc=True, errors="coerce")
        physio = physio.dropna(subset=["ts"])
    out["sleep_physio"] = physio if len(physio) > 0 else None

    state_path = d / "Sleep_state.csv"
    state = pd.read_csv(state_path) if state_path.exists() else pd.DataFrame()
    if len(state) > 0:
        state["start"] = pd.to_datetime(state["Start time"], utc=True, errors="coerce")
        state["end"] = pd.to_datetime(state["End time"], utc=True, errors="coerce")
        state = state.dropna(subset=["start", "end"])
    out["sleep_state"] = state if len(state) > 0 else None

    return out


# ---------------------------------------------------------------------------
# RB3/RB4/RB5 — per-participant channel construction
# ---------------------------------------------------------------------------

def build_hourly_channel(
    df: pd.DataFrame | None, ts_col: str, value_col: str, t0: pd.Timestamp, agg: str
) -> tuple[np.ndarray, float]:
    """Bins df[value_col] into WINDOW_HOURS hourly bins starting at t0.

    Returns (values[512] float32, coverage) where coverage is the fraction
    of the 512 hourly bins with >=1 real observation. Missing bins are
    filled with the channel's OWN real-bin mean (RB4) — never a global or
    cross-participant value. If coverage is 0.0 the returned array is a
    zero array and MUST be discarded by the caller (RB5/RB7), never fed to
    MOMENT.
    """
    values = np.zeros(WINDOW_HOURS, dtype=np.float64)
    real_mask = np.zeros(WINDOW_HOURS, dtype=bool)
    if df is not None and len(df) > 0:
        rel_hours = (df[ts_col] - t0).dt.total_seconds() / SECONDS_PER_HOUR
        in_window = (rel_hours >= 0) & (rel_hours < WINDOW_HOURS)
        sub = df.loc[in_window].copy()
        if len(sub) > 0:
            sub["bin"] = np.floor(rel_hours[in_window]).astype(int).clip(0, WINDOW_HOURS - 1)
            if agg == "mean":
                grouped = sub.groupby("bin")[value_col].mean()
            else:
                grouped = sub.groupby("bin")[value_col].sum()
            values[grouped.index.to_numpy()] = grouped.to_numpy()
            real_mask[grouped.index.to_numpy()] = True

    coverage = float(real_mask.mean())
    if coverage > 0.0:
        fill_value = float(values[real_mask].mean())
        values[~real_mask] = fill_value
    return values.astype(np.float32), coverage


def build_nightly_group(
    physio_df: pd.DataFrame | None,
    state_df: pd.DataFrame | None,
    t0: pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Builds the 8-channel {Sleep_physio x4, Sleep_state x4} nightly-cadence
    array, padded to WINDOW_HOURS positions (RB3), plus its shared
    input_mask (1 for the n_real_nights left-aligned real positions, 0
    elsewhere). Returns (channels[8, 512] float32, mask[512] float32,
    n_real_nights).

    Grouping key: the UTC calendar date of each real timestamp. This is a
    documented simplification (a session crossing local midnight can split
    across two UTC dates) rather than a per-sleep-session inference from
    Sleep_state's own interval boundaries — acceptable because a night's
    physiology summary is not sensitive to a single boundary date split,
    and the dataset carries no timezone beyond "+UTC" to do better.
    """
    channels = np.zeros((8, WINDOW_HOURS), dtype=np.float32)
    mask = np.zeros(WINDOW_HOURS, dtype=np.float32)

    physio_by_date: dict = {}
    if physio_df is not None and len(physio_df) > 0:
        rel_hours = (physio_df["ts"] - t0).dt.total_seconds() / SECONDS_PER_HOUR
        in_window = (rel_hours >= 0) & (rel_hours < WINDOW_HOURS)
        sub = physio_df.loc[in_window].copy()
        if len(sub) > 0:
            sub["date"] = sub["ts"].dt.date
            physio_by_date = {
                d: (
                    float(g["Heart Rate"].mean()),
                    float(g["Respiration Rate"].mean()),
                    float(g["Snoring"].sum()),
                    float(g["SDNN_1"].mean()),
                )
                for d, g in sub.groupby("date")
            }

    state_by_date: dict = {}
    if state_df is not None and len(state_df) > 0:
        rel_hours = (state_df["start"] - t0).dt.total_seconds() / SECONDS_PER_HOUR
        in_window = (rel_hours >= 0) & (rel_hours < WINDOW_HOURS)
        sub = state_df.loc[in_window].copy()
        if len(sub) > 0:
            sub["date"] = sub["start"].dt.date
            sub["dur"] = (sub["end"] - sub["start"]).dt.total_seconds().clip(lower=0.0)
            for d, g in sub.groupby("date"):
                total = float(g["dur"].sum())
                if total <= 0:
                    continue
                props = tuple(
                    float(g.loc[g["Sleep state"] == s, "dur"].sum()) / total
                    for s in SLEEP_STATES
                )
                state_by_date[d] = props

    real_dates = sorted(set(physio_by_date) | set(state_by_date))
    real_dates = real_dates[:WINDOW_HOURS]  # structurally never truncates in practice

    # RB4-consistent fix: a real night where only ONE of {physio, state} has
    # a reading is a missing-BIN gap within an otherwise-real night, not a
    # shorter-real-sequence padding case — the same distinction RB4 already
    # draws for HR/Steps hourly gaps. Filling the other source's 4 channels
    # with THAT SOURCE's own real-value mean (never 0.0, never a global/
    # cross-participant value) keeps every value fed to MOMENT on a masked
    # "valid" night genuinely representative, instead of a literal 0.0
    # (implausible for HR/Respiration/SDNN, and meaningless for a state
    # proportion) silently passing through the shared union mask as if it
    # were a real observation. Bug found and fixed during a rigorous
    # re-check of this script: 268/1503 (17.8%) of real-night positions
    # across 37/73 participants had this exact false-zero-as-valid gap.
    physio_mean = (
        np.mean(list(physio_by_date.values()), axis=0) if physio_by_date else np.zeros(4)
    )
    state_mean = (
        np.mean(list(state_by_date.values()), axis=0) if state_by_date else np.zeros(4)
    )
    for i, d in enumerate(real_dates):
        channels[0:4, i] = physio_by_date.get(d, physio_mean)
        channels[4:8, i] = state_by_date.get(d, state_mean)
        mask[i] = 1.0

    return channels, mask, len(real_dates)


# ---------------------------------------------------------------------------
# RB11 — opt-in external per-window z-score (default off, see docstring)
# ---------------------------------------------------------------------------

def apply_external_zscore(values: np.ndarray, real_mask: np.ndarray | None = None) -> np.ndarray:
    """RM1-style per-window z-score, exact formula/floor reused from
    scripts/train_resilient_multisource_phq9_gds15.py (~lines 471-473):
    z = (x - mean) / max(std, 1e-6).

    When `real_mask` is None (HR/Steps — RB4 already density-fills all 512
    bins, no zero-padding), statistics are computed over the whole array.
    When `real_mask` is given (Sleep_physio/Sleep_state nightly channels,
    RB3 — zero-padded beyond n_real_nights), statistics are computed over,
    and the transform applied to, ONLY the True positions; padded positions
    are returned unchanged so the input_mask MOMENT sees stays meaningful.
    """
    values = values.astype(np.float64)
    if real_mask is None:
        mean = float(values.mean())
        std = max(float(values.std()), 1e-6)
        return ((values - mean) / std).astype(np.float32)

    out = values.copy()
    real = values[real_mask]
    if real.size == 0:
        return out.astype(np.float32)
    mean = float(real.mean())
    std = max(float(real.std()), 1e-6)
    out[real_mask] = (real - mean) / std
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Assembly: one row of raw channel arrays per participant
# ---------------------------------------------------------------------------

def build_channel_tensors(uids: list[str], demo: pd.DataFrame, external_zscore: bool = False) -> dict:
    """Builds every raw per-participant channel array + mask needed by all
    3 configurations, keyed by uid. Nothing here calls MOMENT — this is
    pure data assembly so the 6 downstream MOMENT calls (see
    embed_all_groups) can each batch across every participant at once.

    `external_zscore` (RB11, default off) re-applies the RM1-style external
    per-window z-score to each channel array immediately after RB4 gap
    filling and before it is handed to MOMENT. Default False reproduces
    today's report identically.
    """
    hr_arr, steps_arr = {}, {}
    hr_cov, steps_cov = {}, {}
    night_channels, night_mask, night_n_real = {}, {}, {}
    tabular = {}

    for uid in uids:
        if uid not in demo.index or pd.isna(demo.loc[uid, "phq_date"]):
            continue
        t0 = demo.loc[uid, "phq_date"]
        frames = load_raw_frames(uid)

        hr_vals, cov = build_hourly_channel(frames["hr"], "ts", "Heart Rate", t0, "mean")
        if external_zscore:
            hr_vals = apply_external_zscore(hr_vals)
        hr_arr[uid], hr_cov[uid] = hr_vals, cov

        steps_vals, cov = build_hourly_channel(frames["steps"], "ts", "Steps", t0, "sum")
        if external_zscore:
            steps_vals = apply_external_zscore(steps_vals)
        steps_arr[uid], steps_cov[uid] = steps_vals, cov

        ch, mk, n_real = build_nightly_group(frames["sleep_physio"], frames["sleep_state"], t0)
        if external_zscore:
            real_mask_bool = mk.astype(bool)
            for i in range(ch.shape[0]):
                ch[i] = apply_external_zscore(ch[i], real_mask=real_mask_bool)
        night_channels[uid], night_mask[uid], night_n_real[uid] = ch, mk, n_real

        sex = SEX_MAP.get(demo.loc[uid, "Sex"], np.nan)
        age = AGE_GROUP_MAP.get(demo.loc[uid, "Age group"], np.nan)
        tabular[uid] = (float(sex), float(age))

    return {
        "hr_arr": hr_arr, "hr_cov": hr_cov,
        "steps_arr": steps_arr, "steps_cov": steps_cov,
        "night_channels": night_channels, "night_mask": night_mask, "night_n_real": night_n_real,
        "tabular": tabular,
    }


def embed_group(
    pipeline: MomentRiskPipeline,
    uids: list[str],
    channel_lists: list[dict],
    masks: dict | None,
    coverage_ok: dict[str, bool],
) -> dict[str, np.ndarray]:
    """Runs one MOMENT call for a [n_ok, C, 512] batch (C = len(channel_lists)),
    only for participants passing coverage_ok, and scatters the resulting
    512-D embeddings back into a uid -> array dict with NaN for every
    participant that failed coverage (RB7) — never fed to MOMENT.
    """
    ok_uids = [u for u in uids if coverage_ok.get(u, False)]
    result = {u: np.full(D_MODEL, np.nan, dtype=np.float64) for u in uids}
    if not ok_uids:
        return result

    stacked = np.stack(
        [np.stack([ch[u] for ch in channel_lists], axis=0) for u in ok_uids], axis=0
    ).astype(np.float32)
    windows = torch.from_numpy(stacked)
    input_masks = None
    if masks is not None:
        input_masks = torch.from_numpy(
            np.stack([masks[u] for u in ok_uids], axis=0).astype(np.float32)
        )
    embeddings = embed_all_windows(pipeline, windows, input_masks=input_masks).cpu().numpy()
    for i, u in enumerate(ok_uids):
        result[u] = embeddings[i].astype(np.float64)
    return result


def embed_all_groups(pipeline: MomentRiskPipeline, uids: list[str], ct: dict) -> dict:
    """The 6 distinct MOMENT calls every one of the 3 configurations is
    built from (see RB10) — each channel-group/mask pair is embedded
    exactly once and reused across configs that need it."""
    hr_ok = {u: ct["hr_cov"].get(u, 0.0) >= MIN_COVERAGE for u in uids}
    steps_ok = {u: ct["steps_cov"].get(u, 0.0) >= MIN_COVERAGE for u in uids}
    hr_steps_ok = {u: hr_ok[u] and steps_ok[u] for u in uids}
    night_ok = {u: ct["night_n_real"].get(u, 0) >= MIN_REAL_NIGHTS for u in uids}

    all_ones_mask = {u: np.ones(WINDOW_HOURS, dtype=np.float32) for u in uids}

    print("[embed] HR alone ...")
    hr_emb = embed_group(pipeline, uids, [ct["hr_arr"]], all_ones_mask, hr_ok)
    print("[embed] Steps alone ...")
    steps_emb = embed_group(pipeline, uids, [ct["steps_arr"]], all_ones_mask, steps_ok)
    print("[embed] {HR, Steps} joint (config A) ...")
    hrsteps_emb = embed_group(pipeline, uids, [ct["hr_arr"], ct["steps_arr"]], all_ones_mask, hr_steps_ok)

    night_ch_split = {i: {u: ct["night_channels"][u][i] for u in uids if u in ct["night_channels"]} for i in range(8)}
    print("[embed] Sleep_physio alone (4ch) ...")
    physio_emb = embed_group(pipeline, uids, [night_ch_split[i] for i in range(4)], ct["night_mask"], night_ok)
    print("[embed] Sleep_state alone (4ch) ...")
    state_emb = embed_group(pipeline, uids, [night_ch_split[i] for i in range(4, 8)], ct["night_mask"], night_ok)
    print("[embed] {Sleep_physio, Sleep_state} joint (config A) ...")
    sleep_group_emb = embed_group(pipeline, uids, [night_ch_split[i] for i in range(8)], ct["night_mask"], night_ok)

    return {
        "hr": hr_emb, "steps": steps_emb, "hr_steps": hrsteps_emb,
        "sleep_physio": physio_emb, "sleep_state": state_emb, "sleep_group": sleep_group_emb,
        "coverage": {"hr_ok": hr_ok, "steps_ok": steps_ok, "night_ok": night_ok},
    }


# ---------------------------------------------------------------------------
# Feature matrix assembly per configuration (RB10)
# ---------------------------------------------------------------------------

def assemble_feature_matrix(uids: list[str], embeds: dict, ct: dict, config: str) -> tuple[np.ndarray, list[str]]:
    tabular = np.array([ct["tabular"].get(u, (np.nan, np.nan)) for u in uids], dtype=np.float64)

    if config == "hr_only":
        hr = np.stack([embeds["hr"][u] for u in uids], axis=0)
        X = np.concatenate([hr, tabular], axis=1)
        names = [f"hr_{i}" for i in range(D_MODEL)] + ["sex", "age_group"]
    elif config == "multichannel":
        hs = np.stack([embeds["hr_steps"][u] for u in uids], axis=0)
        sg = np.stack([embeds["sleep_group"][u] for u in uids], axis=0)
        X = np.concatenate([hs, sg, tabular], axis=1)
        names = [f"hrsteps_{i}" for i in range(D_MODEL)] + [f"sleepgroup_{i}" for i in range(D_MODEL)] + ["sex", "age_group"]
    elif config == "separate":
        hr = np.stack([embeds["hr"][u] for u in uids], axis=0)
        st = np.stack([embeds["steps"][u] for u in uids], axis=0)
        ph = np.stack([embeds["sleep_physio"][u] for u in uids], axis=0)
        sl = np.stack([embeds["sleep_state"][u] for u in uids], axis=0)
        X = np.concatenate([hr, st, ph, sl, tabular], axis=1)
        names = (
            [f"hr_{i}" for i in range(D_MODEL)] + [f"steps_{i}" for i in range(D_MODEL)]
            + [f"physio_{i}" for i in range(D_MODEL)] + [f"state_{i}" for i in range(D_MODEL)]
            + ["sex", "age_group"]
        )
    else:
        raise ValueError(config)
    return X, names


# ---------------------------------------------------------------------------
# Classifier (RB8)
# ---------------------------------------------------------------------------

def build_classifier(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM),
        nn.ReLU(),
        nn.Dropout(DROPOUT_P),
        nn.Linear(HIDDEN_DIM, 1),
    )


def run_cv(X: np.ndarray, y: np.ndarray, uids: list[str], seed_offset: int = 0, repeat_idx: int = 0) -> dict:
    """RB9 — participant-level StratifiedKFold(5), every fitted step
    train-fold-only, genuine binary BCEWithLogitsLoss training.

    `repeat_idx=0` (the default) reproduces the original single-split
    behavior byte-for-byte — both the StratifiedKFold split itself and
    every fold's torch init depend on `repeat_idx` only as an additive
    offset that is 0 at the default, so existing callers are unaffected.
    RB12 (repeated CV, see `run_cv_repeated`) varies `repeat_idx` to get
    an independent fold assignment + model init per repeat, to quantify
    how much of the single-split AUROC/AUPRC is fold-assignment noise
    rather than signal (Section 19/20 motivation: n_pos=10 makes any one
    5-fold split's OOF metric noisy)."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE + repeat_idx)
    n = len(uids)
    oof_proba = np.full(n, np.nan, dtype=np.float64)
    fold_of = np.full(n, -1, dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        imputer = SimpleImputer(strategy="mean", keep_empty_features=True)
        X_train = imputer.fit_transform(X[train_idx])
        X_test = imputer.transform(X[test_idx])

        scaler = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        y_train = y[train_idx]
        n_pos_train = int((y_train == 1).sum())
        n_neg_train = int((y_train == 0).sum())
        pos_weight = torch.tensor([n_neg_train / max(n_pos_train, 1)], dtype=torch.float32)

        torch.manual_seed(RANDOM_STATE + seed_offset + fold_idx + repeat_idx * 1000)
        model = build_classifier(X.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        model.train()
        for _ in range(MAX_EPOCHS):
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_proba = torch.sigmoid(test_logits).squeeze(1).numpy()

        oof_proba[test_idx] = test_proba
        fold_of[test_idx] = fold_idx

    assert not np.isnan(oof_proba).any(), "every participant must receive exactly one OOF prediction"
    return {"oof_proba": oof_proba, "fold_of": fold_of, "y": y, "uids": uids}


# ---------------------------------------------------------------------------
# Metrics (RB9)
# ---------------------------------------------------------------------------

def bootstrap_ci(y_true: np.ndarray, proba: np.ndarray, metric_fn, n_boot: int = N_BOOTSTRAP, seed: int = RANDOM_STATE) -> tuple[float, float, int]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, pb = y_true[idx], proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        values.append(metric_fn(yt, pb))
    if not values:
        return float("nan"), float("nan"), 0
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi), len(values)


def compute_metrics(result: dict) -> dict:
    y = result["y"]
    proba = result["oof_proba"]
    counts = confusion_counts(y, proba)
    balanced_acc = balanced_accuracy_score(y, (proba >= DECISION_THRESHOLD).astype(int))
    auroc = float(roc_auc_score(y, proba)) if len(np.unique(y)) >= 2 else float("nan")
    auprc = float(average_precision_score(y, proba)) if len(np.unique(y)) >= 2 else float("nan")
    auroc_lo, auroc_hi, auroc_draws = bootstrap_ci(y, proba, roc_auc_score)
    auprc_lo, auprc_hi, auprc_draws = bootstrap_ci(y, proba, average_precision_score)
    return {
        **counts,
        "balanced_accuracy": float(balanced_acc),
        "auroc": auroc, "auroc_ci": (auroc_lo, auroc_hi), "auroc_draws": auroc_draws,
        "auprc": auprc, "auprc_ci": (auprc_lo, auprc_hi), "auprc_draws": auprc_draws,
        "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum()),
    }


def _fmt_ci(lo: float, hi: float) -> str:
    if np.isnan(lo) or np.isnan(hi):
        return "n/a"
    return f"[{lo:.4f}, {hi:.4f}]"


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())


def run_cv_repeated(X: np.ndarray, y: np.ndarray, uids: list[str], seed_offset: int, n_repeats: int) -> dict:
    """RB12 — repeats `run_cv` with `n_repeats` independent StratifiedKFold
    splits (different fold assignment + model init per repeat, same X/y),
    then reports mean +/- std across repeats for every metric. Addresses
    the Section 19/20 "only 10 positives -> one 5-fold split is noisy"
    finding directly: a single split's AUROC/AUPRC is one draw from a wide
    distribution, not a stable estimate."""
    per_repeat = [compute_metrics(run_cv(X, y, uids, seed_offset=seed_offset, repeat_idx=r)) for r in range(n_repeats)]
    return {
        "n": len(y),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "n_repeats": n_repeats,
        "accuracy": _mean_std([m["accuracy"] for m in per_repeat]),
        "balanced_accuracy": _mean_std([m["balanced_accuracy"] for m in per_repeat]),
        "sensitivity": _mean_std([m["sensitivity"] for m in per_repeat]),
        "specificity": _mean_std([m["specificity"] for m in per_repeat]),
        "auroc": _mean_std([m["auroc"] for m in per_repeat]),
        "auprc": _mean_std([m["auprc"] for m in per_repeat]),
    }


def build_repeated_report(repeated_metrics_by_config: dict) -> str:
    lines = []
    lines.append("## Repeated stratified CV (RB12 — mean +/- std across independent 5-fold splits)\n")
    lines.append(
        "Same X/y and classifier as the single-split table above; only the "
        "StratifiedKFold split (and each fold's model init) is re-drawn per "
        "repeat. Quantifies fold-assignment noise directly rather than relying "
        "solely on the asymptotic bootstrap CI, given only ~10 positive cases.\n"
    )
    lines.append("| Config | Repeats | N | n_pos/n_neg | Accuracy | Balanced acc. | Sensitivity | Specificity | AUROC | AUPRC |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for cfg, m in repeated_metrics_by_config.items():
        acc_m, acc_s = m["accuracy"]
        bal_m, bal_s = m["balanced_accuracy"]
        sens_m, sens_s = m["sensitivity"]
        spec_m, spec_s = m["specificity"]
        auroc_m, auroc_s = m["auroc"]
        auprc_m, auprc_s = m["auprc"]
        lines.append(
            f"| {cfg} | {m['n_repeats']} | {m['n']} | {m['n_pos']}/{m['n_neg']} | "
            f"{acc_m:.4f} +/- {acc_s:.4f} | {bal_m:.4f} +/- {bal_s:.4f} | "
            f"{sens_m:.4f} +/- {sens_s:.4f} | {spec_m:.4f} +/- {spec_s:.4f} | "
            f"{auroc_m:.4f} +/- {auroc_s:.4f} | {auprc_m:.4f} +/- {auprc_s:.4f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(shapes: dict, coverage: dict, metrics_by_config: dict) -> str:
    lines = []
    lines.append("# RESILIENT Multimodal Binary PHQ-9 (>=10) Classification\n")
    lines.append(
        "Real measured run of `scripts/train_resilient_multimodal_binary_phq9.py`. "
        "The existing `scripts/train_resilient_multisource_phq9_gds15.py` (4-class, "
        "whole-recording HR+tabular) is untouched and unaffected by this file.\n"
    )
    lines.append("## Pre-registered design (RB1-RB10)\n")
    lines.append(
        "See this script's own module docstring for the full reasoning "
        "(temporal window choice, per-modality representation, normalization "
        "removal, binary training, evaluation). Not restated here to avoid drift "
        "between two copies of the same text.\n"
    )
    lines.append("## Coverage (RB5/RB7)\n")
    n = len(coverage["hr_ok"])
    lines.append(f"- HR coverage >= {MIN_COVERAGE}: {sum(coverage['hr_ok'].values())}/{n}")
    lines.append(f"- Steps coverage >= {MIN_COVERAGE}: {sum(coverage['steps_ok'].values())}/{n}")
    lines.append(f"- Sleep group with >= {MIN_REAL_NIGHTS} real night: {sum(coverage['night_ok'].values())}/{n}\n")

    lines.append("## Exact tensor shapes\n")
    for cfg, shp in shapes.items():
        lines.append(f"**{cfg}**")
        for k, v in shp.items():
            lines.append(f"- {k}: `{v}`")
        lines.append("")

    lines.append("## Results (pooled out-of-fold, 5-fold participant-level StratifiedKFold)\n")
    lines.append("| Config | N | n_pos/n_neg | Accuracy | Sensitivity | Specificity | Balanced acc. | AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | TP/FP/TN/FN |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cfg, m in metrics_by_config.items():
        lines.append(
            f"| {cfg} | {m['n']} | {m['n_pos']}/{m['n_neg']} | {m['accuracy']:.4f} | "
            f"{m['sensitivity']:.4f} | {m['specificity']:.4f} | {m['balanced_accuracy']:.4f} | "
            f"{m['auroc']:.4f} | {_fmt_ci(*m['auroc_ci'])} | {m['auprc']:.4f} | {_fmt_ci(*m['auprc_ci'])} | "
            f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']} |"
        )
    lines.append("")
    lines.append(
        "Bootstrap CIs use 2000 resamples of participants; a resample lacking both "
        "classes contributes no value to that metric's distribution (draws used is "
        "reported in the run log, not this table)."
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracer", action="store_true", help=f"Run on a {TRACER_SAMPLE_SIZE}-participant subset, write no report.")
    parser.add_argument(
        "--external-zscore", action="store_true",
        help="RB11 opt-in comparison arm: re-apply the RM1-style external per-window "
             "z-score before MOMENT. Default off — omitting this flag reproduces the "
             "published report identically. Only the 'separate' config is run when set.",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="RB12 opt-in: also run each config through N independent StratifiedKFold "
             "splits and report mean +/- std (default 1 = off, reproduces the published "
             "report identically). Does not change the single-split table already written.",
    )
    parser.add_argument(
        "--positive-threshold", type=int, default=PHQ9_POSITIVE_THRESHOLD,
        help=f"RB13 opt-in comparison arm: relabel positive as phq_total >= this value "
             f"instead of the locked convention's {PHQ9_POSITIVE_THRESHOLD} (default — "
             "omitting this flag reproduces the published report identically). "
             "phq_total>=5 is PHQ-9's own standard 'mild-or-above' clinical band, not an "
             "invented cutoff. Never overwrites the default-threshold report; always "
             "writes a threshold-suffixed report file instead.",
    )
    args = parser.parse_args()

    demo = load_demographics()
    uids = get_participant_uids()
    if args.tracer:
        uids = uids[:TRACER_SAMPLE_SIZE]
    print(f"[main] n_uids={len(uids)} tracer={args.tracer} external_zscore={args.external_zscore}")

    print("[main] Building per-participant channel arrays (RB3/RB4/RB5) ...")
    ct = build_channel_tensors(uids, demo, external_zscore=args.external_zscore)
    uids = [u for u in uids if u in ct["tabular"]]  # drop any with no phq_date at all (none expected)
    print(f"[main] n_uids with real phq_date={len(uids)}")

    print("[main] Loading real frozen MOMENT-Small backbone (MomentRiskPipeline.from_pretrained()) ...")
    pipeline_model = MomentRiskPipeline.from_pretrained()
    trainable = sum(p.numel() for p in pipeline_model.backbone.parameters() if p.requires_grad)
    assert trainable == 0, f"expected 0 trainable backbone parameters, measured {trainable}"
    print(f"[main] 0 trainable backbone parameters confirmed (measured {trainable})")

    embeds = embed_all_groups(pipeline_model, uids, ct)

    y = np.array([1 if demo.loc[u, "phq_total"] >= args.positive_threshold else 0 for u in uids], dtype=int)
    print(f"[main] y: threshold={args.positive_threshold} n_pos={int((y == 1).sum())} n_neg={int((y == 0).sum())} / n={len(y)}")

    # RB10 default configs; under RB11 --external-zscore only "separate" (row B's
    # architecture) is run, so seed_offset is keyed by config name (not loop
    # position) to stay identical to the default path's seed_offset=2 for "separate".
    configs = ["separate"] if args.external_zscore else ["hr_only", "multichannel", "separate"]
    config_seed_offset = {"hr_only": 0, "multichannel": 1, "separate": 2}

    shapes: dict = {}
    metrics_by_config: dict = {}
    repeated_metrics_by_config: dict = {}
    for config in configs:
        X, names = assemble_feature_matrix(uids, embeds, ct, config)
        shapes[config] = {
            "per-window MOMENT input(s)": {
                "hr_only": "[n_ok, 1, 512]",
                "multichannel": "[n_ok, 2, 512] (HR+Steps) and [n_ok, 8, 512] (Sleep_physio+Sleep_state)",
                "separate": "[n_ok, 1, 512] (HR) + [n_ok, 1, 512] (Steps) + [n_ok, 4, 512] (Sleep_physio) + [n_ok, 4, 512] (Sleep_state)",
            }[config],
            "feature matrix X": tuple(X.shape),
        }
        print(f"[main][{config}] X shape={X.shape}")
        if args.tracer:
            print(f"[main][{config}] tracer mode — skipping CV/metrics")
            continue
        result = run_cv(X, y, uids, seed_offset=config_seed_offset[config])
        metrics_by_config[config] = compute_metrics(result)
        m = metrics_by_config[config]
        print(
            f"[main][{config}] Accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} "
            f"balanced_acc={m['balanced_accuracy']:.4f} sens={m['sensitivity']:.4f} spec={m['specificity']:.4f}"
        )

        if args.repeats > 1:
            config_label = f"{config}_ext" if args.external_zscore else config
            rm = run_cv_repeated(X, y, uids, seed_offset=config_seed_offset[config], n_repeats=args.repeats)
            repeated_metrics_by_config[config_label] = rm
            print(
                f"[main][{config_label}] repeats={args.repeats} "
                f"Accuracy={rm['accuracy'][0]:.4f}+/-{rm['accuracy'][1]:.4f} "
                f"AUROC={rm['auroc'][0]:.4f}+/-{rm['auroc'][1]:.4f} "
                f"AUPRC={rm['auprc'][0]:.4f}+/-{rm['auprc'][1]:.4f}"
            )

    if args.tracer:
        print("[main] tracer mode complete — no report written")
        return

    non_default_threshold = args.positive_threshold != PHQ9_POSITIVE_THRESHOLD
    threshold_suffix = f"_threshold{args.positive_threshold}" if non_default_threshold else ""

    if args.external_zscore:
        if args.repeats > 1:
            ext_report = build_repeated_report(repeated_metrics_by_config)
            ext_path = REPORT_PATH.with_name(REPORT_PATH.stem + threshold_suffix + "_external_zscore_repeated.md")
            ext_path.write_text(ext_report)
            print(f"[main] wrote {ext_path}")
        else:
            print(
                "[main] --external-zscore set — RB11 opt-in comparison arm, not written to "
                f"{REPORT_PATH}; transcribe the printed Accuracy/AUROC for row C by hand."
            )
        return

    report = build_report(shapes, embeds["coverage"], metrics_by_config)
    if args.repeats > 1:
        report += "\n" + build_repeated_report(repeated_metrics_by_config)
    report_path = REPORT_PATH.with_name(REPORT_PATH.stem + threshold_suffix + ".md") if non_default_threshold else REPORT_PATH
    report_path.write_text(report)
    print(f"[main] wrote {report_path}")


if __name__ == "__main__":
    main()
