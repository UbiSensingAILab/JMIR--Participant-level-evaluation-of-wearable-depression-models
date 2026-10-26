#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), derived from the DiSCover Project
# (NCT03421223), distributed under CC BY-NC 4.0. This is a SEPARATE
# institution's protocol, independent of this project's own ethics
# protocol (REB25-1587_MOD1) — the two must NOT be conflated: PSYCHE-D
# participants are not MAMAI mothers, and PSYCHE-D data never enters the
# MAMAI live pipeline. `research_datasets/` is gitignored (.gitignore:22);
# the parquet this script reads is local research data and is never
# committed to this repo.
"""PSYCHE-D participant-level, binary PHQ-9 severity-change ("worsened")
classification using a classical tabular pipeline on the dataset's own
already-engineered steps/sleep scalar features.

Pre-registered design decisions (PD1-PD17), fixed here BEFORE any
accuracy/AUROC/AUPRC number is computed and never re-drawn, narrowed, or
swapped after seeing which model scores better — this project's RM1-RM12
pre-registration discipline (see `train_resilient_multisource_phq9_gds15.py`),
renumbered for this dataset:

PD1: Data source and labeled subset. Read the parquet; keep only rows
where both `phq9_score_start` and `phq9_score_end` are non-null; assert
the kept count is exactly 10866 (the other ~24.8k rows carry no PHQ-9
label and are excluded, not imputed).

PD2: Participant id and grouping. No participant-id column exists; the id
is the dataframe index prefix before the first underscore (`SOURCE.md`:
indexed by `[participant]_[month]`). Assert 4036 unique participants.
Participants contribute 1-4 samples (median 3) — no participant is ever
dropped (D-02).

PD3: Label. Binary "worsened" = `(phq9_cat_end - phq9_cat_start) > 0`.
Assert 2252 positives / 10866 (0.2073). "Same" (55.6%) and "improved"
(23.7%) are both folded into the negative class — a deliberate binary
framing of a 3-level outcome, disclosed rather than hidden (D-03).

PD4: Temporal alignment. Per the dataset authors' own `SOURCE.md`, samples
pair a 3-month PHQ-9 change with 8-14 days of *preceding* wearable data.
The feature window therefore PRECEDES the interval whose severity change
is predicted: this is a genuine forecast design [CORRECTION 2026-10-16: not a
forecast; the wearable and lifestyle inputs are concurrent with the end-of-period
PHQ-9 recall window; see results/revision/P0_AUDIT.md], not concurrent digital
phenotyping (D-01). This is strictly less leakage-prone than a concurrent
design would have been.

PD5: Leakage exclusion. `phq9_score_start`, `phq9_score_end`,
`phq9_cat_start`, `phq9_cat_end` are excluded from the feature matrix (the
label is computed from two of them). Assert none of the four survives into
the feature column list, and assert no surviving feature name contains the
substring `phq9`.

PD6: Feature matrix. Every remaining numeric column is kept (measured:
150; `race_*` are bool and count as numeric). No missingness-based column
drop is applied; instead the worst per-column NaN fraction is measured and
disclosed in the report. Assert the measured feature count and re-print
it.

PD7: Missing data. One shared `SimpleImputer(strategy="mean",
keep_empty_features=True)`, fit on the training fold ONLY inside each CV
split, never a global upfront impute (this codebase's standing
convention). No row is dropped for missingness.

PD8: Scaling. `StandardScaler`, fit on the training fold ONLY, applied
after imputation — mirroring `train_resilient_classical_hr_baseline.py`'s
`Pipeline(StandardScaler -> LogisticRegression)` classical convention.

PD9: CV. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
split on `(X, y, groups=participant_ids)`. Because stratification is only
approximate once grouping is enforced, the real per-fold test positive
rate is measured and printed rather than assumed.

PD10: Classifiers. Two classical arms, both class-weighted:
`LogisticRegression(class_weight="balanced", max_iter=2000,
random_state=42)` and `RandomForestClassifier(n_estimators=300,
class_weight="balanced_subsample", random_state=42, n_jobs=-1)`. Both are
run through the identical PD6-PD9 pipeline.

PD11: No MOMENT, and why. Every steps/sleep column in this file is
already a pre-engineered scalar (mean / IQR / median / day-count /
linear-regression trend coefficient); there is no raw per-timestep series
anywhere for MOMENT's 512-step window to operate on. Fabricating a
pseudo-sequence from those scalars would be decorative, not a real MOMENT
application. This is a data-shape fact, not an oversight or a workaround
(D-04).

PD12: Metrics and baselines. Pooled out-of-fold accuracy (threshold 0.5),
balanced accuracy, sensitivity, specificity, AUROC, AUPRC, and the
TP/FP/TN/FN counts, plus seeded bootstrap 95% CIs (2000 resamples, seed
42) for AUROC and AUPRC. Baselines reported beside them: majority-class
accuracy (always predict "not worsened"), chance AUROC 0.5, and chance
AUPRC equal to the positive rate.

PD13: Pre-registered success criterion. A model "clears its own baseline"
only if BOTH its AUROC 95% CI lower bound exceeds 0.5 AND its accuracy
exceeds the majority-class accuracy. This is computed in code into a
boolean per model and exported; it is decided here, before any number is
seen (precedent: quick 260916-dpu, where CIs spanning 0.5 meant no
positive claim was made).

PD14: Undocumented feature variants. The `__score` / `__intercept` /
`__coeff` regression-trend triples appear in two variants per signal
(plain and trailing-underscore-suffixed) and `feature_explanations.tsv`
documents neither. Both variants are kept; the report discloses this as a
real unknown rather than guessing which is which.

PD15: Export scope and privacy. Only aggregate metrics, counts, and
design text are written to `src/data/psycheDResults.json` — no
per-participant or per-sample rows are ever exported to the repo, and the
parquet stays gitignored. (This deliberately departs from
`watClassifierResults.json`'s `samplePredictions` block, because PSYCHE-D
is another institution's human-subject research data.)

PD16: Environment and reproducibility. Run under system `python3`; print
pandas / numpy / scikit-learn versions at runtime and record them in the
report. Global seed 42. Zero new package installs.

PD17: Third arm - simple neural network, and its imbalance handling. A
third classical arm, additive alongside PD10's two arms (their code paths,
numbers, and pipeline are unchanged): `sklearn.neural_network.MLPClassifier`
with a single hidden layer of 24 units (deliberately small, bounded to
16-32 units, given only ~10866 samples / 4036 grouped participants, where
a larger network risks overfitting), `alpha=0.01` L2 regularization,
`early_stopping=True` with `validation_fraction=0.1` drawn only from the
training fold (never from held-out test data), and `random_state=42`.
`MLPClassifier` has no `class_weight` parameter, unlike PD10's two arms, so
class imbalance is instead handled via per-sample `sample_weight` passed
through `Pipeline.fit(clf__sample_weight=...)`, computed per training fold
from `sklearn.utils.class_weight.compute_class_weight("balanced", ...)`
mapped from a per-class weight to a per-sample weight array — the same
`compute_class_weight("balanced", ...)` balancing primitive this project
already uses in `train_resilient_multisource_phq9_gds15.py`, applied here
via sample weights rather than a loss-level class_weight because
MLPClassifier does not expose one. This arm is run through the identical
PD6-PD9 pipeline (impute -> scale -> fit on train fold only) as PD10's two
arms, and is scored against the identical PD12 metric set and PD13 success
criterion — no separate ad-hoc path.

PD18: Baseline PHQ-9 severity is re-included as a legitimate covariate, and
why PD5's original exclusion was over-conservative. PD5 excluded all FOUR
of `phq9_score_start`, `phq9_score_end`, `phq9_cat_start`, `phq9_cat_end`
from the feature matrix as "leakage". Only the label-adjacent pair,
`phq9_score_end` and `phq9_cat_end`, actually encodes information that is
unavailable until the outcome interval is already over — those two remain
excluded from every feature variant below, unchanged, and this is verified
by the same kind of leakage assertion PD5 already uses. `phq9_cat_start`
and `phq9_score_start` (baseline severity) are known BEFORE the wearable
feature window even begins (PD4) and are ordinary legitimate covariates,
not leakage — excluding them was needlessly conservative.

Baseline severity also has a real, mechanical, non-circular relationship
to the label via a ceiling/floor effect, disclosed here rather than left
to look like a mysterious accuracy gain: `phq9_cat` is a 5-level ordinal
scale (0-4), and the label is `worsened = (phq9_cat_end - phq9_cat_start)
> 0`. A participant who starts at the ceiling (`phq9_cat_start == 4`)
CANNOT "worsen" by this label's definition — there is no category above 4
to move into, so `worsened` is structurally 0 for every such row. Measured
directly on the 10866 labeled samples: 541/10866 (4.98%) start at the
ceiling (`phq9_cat_start == 4`), and among them 0/541 are labeled
"worsened" — confirming the structural impossibility exactly, not
approximately. Symmetrically, 4024/10866 (37.03%) start at the floor
(`phq9_cat_start == 0`), where "worsened" is the only direction available,
making that group somewhat more likely to be flagged "worsened" than a
mid-scale participant. This uses none of `phq9_cat_end`/`phq9_score_end`
and is real, honest, non-leakage signal — any accuracy gain from adding
baseline severity is at least partly explained by this understandable
mechanism (the ceiling group alone), not a mystery.

Feature variant. A new `with_baseline_severity` feature-set variant is
added: the existing 150-column feature matrix (PD6, unchanged, same 150
columns, same code path) PLUS BOTH `phq9_score_start` (continuous, more
granular) AND `phq9_cat_start` (ordinal, matches the label's own native
scale) — both are kept rather than picking one, since they are cheap,
non-collinear-by-construction covariates (one is a continuous score, the
other its categorical bucketing) and there is no reason to discard either.
This yields 152 feature columns. `phq9_score_end` and `phq9_cat_end` are
asserted absent from this list, exactly as PD5 asserts for the original
150. All three PD10/PD17 classifiers (LogisticRegression, RandomForest,
MLP) are re-run through the identical PD6-PD9 pipeline (impute/scale
fit-train-fold-only, StratifiedGroupKFold(5), seed 42) on this new
variant — an honest apples-to-apples comparison against the original
150-feature arms, which are left completely unmodified (same code paths,
same numbers, same docstring above this entry). The comparison is reported
plainly: if baseline severity does not meaningfully move a model past its
PD13 baseline, that is stated as such rather than searched for a more
favorable framing.

PD19: Ceiling-restricted cohort - isolating genuinely undetermined
participants. PD18 measured that participants starting at the ceiling
(`phq9_cat_start == 4`) cannot "worsen" by construction: 541/10866 (4.98%)
are structurally guaranteed a negative label, so any model gets partial,
non-informative credit for correctly labeling them negative, which can
inflate apparent discrimination without saying anything about whether
wearable/tabular features actually help distinguish worsening among
participants for whom the outcome is genuinely undetermined either way. To
test whether the PD18 AUROC gain reflects real signal on that genuinely
undetermined remainder, a restricted cohort excludes ONLY the ceiling group
(`phq9_cat_start == 4`); the floor group (`phq9_cat_start == 0`) is NOT
excluded, because "worsened" remains a real, undetermined possible outcome
for floor participants there — starting at the floor makes "worsened" more
likely, not certain, so it stays in.

Measured directly: restricted N = 10325 (10866 - 541), unique participants =
3937, positive count unchanged at 2252 (every one of the 2252 positives was
already a non-ceiling row, since 0/541 ceiling rows are positive), and the
positive rate shifts from 0.2073 to 0.2181 (2252/10325) purely because the
denominator shrank while the numerator did not. This is asserted in code,
not assumed.

The SAME 3 classifiers (LogisticRegression, RandomForest, MLP) are re-run on
BOTH existing feature-set variants (the original 150-column PD6 set, and
PD18's 152-column with-baseline-severity set) through the identical PD6-PD9
pipeline (impute/scale fit-train-fold-only, `StratifiedGroupKFold(n_splits=5,
shuffle=True, random_state=42)` re-fit on the restricted cohort's own
participant groups, since removing 541 samples changes the group
composition) — yielding 6 new result cells, additive to and never replacing
the full-cohort PD1-PD18 arms above, whose code paths and numbers are left
completely unmodified. The restricted cohort's own majority-class accuracy
is recomputed fresh from its own real base rate (not reused from the full
cohort), since the base rate shifted. This is a harder, more honest test:
if the PD18 with-baseline-severity AUROC gain survives on this cohort,
baseline severity (and/or wearable features) carry real signal about
genuinely undetermined severity change; if the gain shrinks back toward the
original 150-feature arm's numbers once the deterministic ceiling group is
removed, that is reported plainly as such, not searched for a more
favorable framing.

PD20: Cleanup lever - duplicate feature-variant correlation diagnostic and
missingness indicators, tested on top of PD18's with-baseline-severity
feature set. Two parts, both measured BEFORE any downstream accuracy
number was looked at.

Part A - duplicate-variant correlation. PD14 disclosed that the
`__score`/`__intercept`/`__coeff` regression-trend triples appear in two
undocumented variants per signal (plain and trailing-underscore-suffixed)
with `feature_explanations.tsv` distinguishing neither. Every plain/
suffixed pair is enumerated programmatically (a column ending in a single
trailing underscore whose non-suffixed counterpart also exists in the
150-column feature list; not hardcoded) - measured: 42 pairs (14 signals x
3 stat types: score/intercept/coeff), covering 84 of the 150 columns. Real
pairwise Pearson correlation is computed for every pair: min |r| = 0.1207,
median |r| = 0.4307, max |r| = 0.9606, mean |r| = 0.4739. Pre-registered
rule: drop the suffixed variant only if median |r| > 0.8. The measured
median is 0.4307, well below that bound, so the decision is KEEP BOTH
VARIANTS - these are NOT simply duplicates measuring the same thing twice
(many pairs correlate only weakly, e.g. min |r| = 0.1207); dropping either
would risk losing real information the other doesn't carry. This decision
is grounded in the actual measured distribution above, not the a-priori
guess that they might be redundant.

Part B - missingness indicators. The real per-column NaN-fraction
distribution across the 150 PD6 feature columns is measured: min=0.0000,
p25=0.0000, median=0.0414, p75=0.2309, p90=0.4346, max=0.8666. The
distribution has a clean natural gap between roughly 10.0% and 20.6% - no
feature column's real NaN fraction falls in that range - so a threshold of
>20% (a clean round number sitting inside that real gap, and close to the
measured p75 of 0.2309) partitions the exact same 52-column
high-missingness group regardless of precisely where inside the gap it is
placed. For each of these 52 columns, a binary `{column}_was_missing`
indicator (1 if NaN before imputation, 0 otherwise) is added alongside the
existing mean-imputed value, preserving whether a feature was actually
recorded (e.g. wearable non-adherence) rather than only ever imputing the
population mean.

Combined "cleaned" feature-set variant. PD18's 152-column
with-baseline-severity set (unchanged, same 152 columns) PLUS 52 new
`_was_missing` indicator columns = 204 total columns (Part A adds no
column changes since the keep-both decision leaves the original 150
untouched). Evaluated on the FULL cohort (not PD19's ceiling-restricted
cohort, to stay directly comparable to PD18's headline numbers) through
the identical PD6-PD9 pipeline (impute/scale fit-train-fold-only,
StratifiedGroupKFold(5), seed 42) and the same 3 classifiers
(LogisticRegression, RandomForest, MLP) - an honest, additive comparison
against PD18's 152-column arm, whose code path, numbers, and this
docstring's own PD1-PD19 text are left completely unmodified. The result
is reported plainly in whichever direction it goes: if the cleanup makes
no meaningful difference, that is stated as such, not searched for a more
favorable framing.

PD21: Continuous PHQ-9 score-change regression as a test of whether PD3's
binarization threw away real signal. PD3 binarized a 5-level PHQ-9 category
jump into a "worsened"/"not worsened" label, so a jump from category 0 to 1
(e.g. score 4->6) and a jump from category 0 to 4 (e.g. score 2->22) count as
the identical positive label, even though they represent very different
magnitudes of real change. This experiment tests whether working with the
raw, continuous score change recovers some of that lost signal, decided here
before any regression number is measured.

Target: `score_delta = phq9_score_end - phq9_score_start`, a continuous real
number computed from data already loaded, not a new data source. This is not
circular: `phq9_score_end` remains excluded from the FEATURE matrix exactly
as PD5/PD18 already exclude it (verified by the same leakage assertion those
entries use); it is only used to construct the regression target, exactly
the same way it is already used to construct the existing binary label
(PD3).

Models: the regression counterparts of the existing PD10/PD17 classifier
family, run on PD18's 152-column with-baseline-severity feature set (the
current best-performing configuration) - PD18's own classification arm,
code path, and numbers are left completely unmodified. `Ridge` is the
regression analog of `LogisticRegression`. `RandomForestRegressor` mirrors
PD10's `RandomForestClassifier` conventions (`n_estimators=300,
random_state=42`). `MLPRegressor` mirrors PD17's `MLPClassifier` conventions
(`hidden_layer_sizes=(24,), alpha=0.01, early_stopping=True,
validation_fraction=0.1, random_state=42`). Regressors have no
`class_weight`/imbalance concept of their own since the fitted target is
continuous, so PD17's sample-weight balancing does not apply here. All three
are run through the identical PD6-PD9 pipeline (impute/scale
fit-train-fold-only, StratifiedGroupKFold(5), seed 42).

Evaluation, two parts. (1) Real regression metrics via the same
`StratifiedGroupKFold(5)` fold-assignment mechanism used by every other arm:
the folds still stratify on the existing binary `worsened` label for fold
balance, even though the fitted target is continuous, keeping every arm's
fold composition directly comparable. Pooled out-of-fold MAE, RMSE, and R^2
are reported for each regressor. (2) A derived ranking-based AUROC for
direct comparability with every classification arm: each regressor's
continuous out-of-fold prediction is used as a decision score and scored via
AUROC (same seeded bootstrap 95% CI convention, 2000 resamples, seed 42)
against PD3's REAL existing binary `worsened` label - the label is reused
as-is, never re-derived. A derived binary accuracy is also reported at the
natural threshold (predicted `score_delta > 0` => predicted "worsened"),
against the same majority-class baseline used everywhere else, and scored
against the same PD13-style clears-baseline verdict (derived-AUROC CI lower
bound > 0.5 AND derived accuracy > majority-class accuracy).

Whether this recovers any accuracy/AUROC PD3's binarization may have thrown
away is stated plainly, comparing each regressor directly against its
classification counterpart on the identical 152-column feature set (Ridge
vs. LogisticRegression, RandomForestRegressor vs. RandomForestClassifier,
MLPRegressor vs. MLPClassifier, all from PD18's `results_with_baseline_severity`)
- reported honestly whichever direction the real numbers go, not searched
for a more favorable framing.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH = Path(__file__).resolve().parent.parent / "research_datasets" / "old" / "psyche_d" / "anon_processed_df.parquet"
RANDOM_STATE = 42
N_SPLITS = 5
N_BOOTSTRAP = 2000
DECISION_THRESHOLD = 0.5
LEAKAGE_COLUMNS = ["phq9_score_start", "phq9_score_end", "phq9_cat_start", "phq9_cat_end"]
# PD18 — only these two directly encode the outcome and remain excluded from
# every feature variant, including the with-baseline-severity one below.
END_LEAKAGE_COLUMNS = ["phq9_score_end", "phq9_cat_end"]
# PD18 — baseline severity, known before the wearable feature window begins
# (PD4), is a legitimate covariate, not leakage.
BASELINE_SEVERITY_COLUMNS = ["phq9_score_start", "phq9_cat_start"]
# PD20 Part A — pre-registered drop rule: only drop the suffixed duplicate
# variant if the REAL measured median |r| across all enumerated pairs
# exceeds this bound. Decided BEFORE the correlation numbers were computed.
DUPLICATE_DROP_THRESHOLD = 0.8
# PD20 Part B — a clean round number that falls squarely inside the real
# measured gap in the NaN-fraction distribution (no feature column has a
# NaN fraction between roughly 10.0% and 20.6%), so it partitions the same
# high-missingness column group regardless of exactly where inside that
# gap it is placed. See PD20 in the module docstring for the full
# reasoning and the measured distribution that justifies it.
MISSINGNESS_INDICATOR_THRESHOLD = 0.20

REPORT_PATH = Path(__file__).resolve().parent / "train_psyche_d_severity_change_report.md"


# ---------------------------------------------------------------------------
# Data loading (PD1-PD6)
# ---------------------------------------------------------------------------

def load_labeled_frame() -> pd.DataFrame:
    """PD1 — read the parquet, keep only rows with both PHQ-9 start/end
    scores, assert the exact real count (10866)."""
    frame = pd.read_parquet(DATA_PATH)
    labeled = frame[frame["phq9_score_start"].notna() & frame["phq9_score_end"].notna()].copy()
    assert len(labeled) == 10866, f"expected 10866 labeled rows, got {len(labeled)}"
    return labeled


def derive_participant_ids(frame: pd.DataFrame) -> np.ndarray:
    """PD2 — participant id is the index prefix before the first underscore
    (`SOURCE.md`: indexed by `[participant]_[month]`). Assert 4036 unique
    participants."""
    ids = frame.index.to_series().str.split("_").str[0].to_numpy()
    n_unique = len(set(ids))
    assert n_unique == 4036, f"expected 4036 unique participants, got {n_unique}"
    return ids


def build_label(frame: pd.DataFrame) -> np.ndarray:
    """PD3 — binary worsened = (phq9_cat_end - phq9_cat_start) > 0. Assert
    2252 positives / 10866 (0.2073)."""
    delta_cat = frame["phq9_cat_end"] - frame["phq9_cat_start"]
    y = (delta_cat > 0).astype(int).to_numpy()
    n_pos = int(y.sum())
    assert n_pos == 2252, f"expected 2252 positives, got {n_pos}"
    rate = n_pos / len(y)
    assert abs(rate - 0.2073) < 0.001, f"expected positive rate ~0.2073, got {rate:.4f}"
    return y


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    """PD5/PD6 — the exclusion-list pattern: every column except the four
    PHQ-9 leakage columns, asserted numeric-dtype and leakage-free."""
    feature_cols = [c for c in frame.columns if c not in LEAKAGE_COLUMNS]
    for leak_col in LEAKAGE_COLUMNS:
        assert leak_col not in feature_cols, f"leakage column {leak_col!r} must not survive into features"
    for col in feature_cols:
        assert "phq9" not in col.lower(), f"feature column {col!r} contains the substring 'phq9' — leakage risk"
        assert pd.api.types.is_numeric_dtype(frame[col]), f"feature column {col!r} is not numeric-dtype"
    assert len(feature_cols) == 150, f"expected 150 feature columns, got {len(feature_cols)}"
    return feature_cols


def get_feature_columns_with_baseline_severity(feature_cols: list[str]) -> list[str]:
    """PD18 — the original 150-column feature list (unchanged, unmodified)
    PLUS `phq9_score_start` and `phq9_cat_start` (baseline severity, known
    before the wearable feature window begins — a legitimate covariate, not
    leakage). Assert `phq9_score_end`/`phq9_cat_end` still do not survive,
    exactly as PD5 asserts for the original 150, and assert the new count."""
    extended = feature_cols + BASELINE_SEVERITY_COLUMNS
    for leak_col in END_LEAKAGE_COLUMNS:
        assert leak_col not in extended, f"leakage column {leak_col!r} must not survive into with_baseline_severity features"
    assert len(extended) == 152, f"expected 152 feature columns with baseline severity, got {len(extended)}"
    return extended


def measure_baseline_severity_extremes(frame: pd.DataFrame, y: np.ndarray) -> dict:
    """PD18 — measure and disclose the ceiling/floor participant counts that
    partly explain any accuracy gain from adding baseline severity. Assert
    the exact measured counts (541 ceiling, 4024 floor) and confirm the
    ceiling group's structural impossibility of "worsened" directly (0/541)."""
    cat_start = frame["phq9_cat_start"].to_numpy()
    n = len(frame)
    ceiling_mask = cat_start == 4
    floor_mask = cat_start == 0
    n_ceiling = int(ceiling_mask.sum())
    n_floor = int(floor_mask.sum())
    assert n_ceiling == 541, f"expected 541 ceiling (phq9_cat_start==4) rows, got {n_ceiling}"
    assert n_floor == 4024, f"expected 4024 floor (phq9_cat_start==0) rows, got {n_floor}"
    n_worsened_among_ceiling = int(y[ceiling_mask].sum())
    assert n_worsened_among_ceiling == 0, (
        f"expected 0 'worsened' among ceiling participants (structurally impossible), "
        f"got {n_worsened_among_ceiling}"
    )
    return {
        "n_ceiling": n_ceiling,
        "n_ceiling_fraction": n_ceiling / n,
        "n_floor": n_floor,
        "n_floor_fraction": n_floor / n,
        "n_worsened_among_ceiling": n_worsened_among_ceiling,
    }


def build_restricted_cohort_mask(frame: pd.DataFrame) -> np.ndarray:
    """PD19 — restricted-cohort mask excluding ONLY the deterministic ceiling
    group (`phq9_cat_start == 4`), where "worsened" is structurally
    impossible. The floor group (`phq9_cat_start == 0`) is NOT excluded —
    "worsened" remains a real, undetermined outcome there. Assert the
    restricted count (10325)."""
    mask = (frame["phq9_cat_start"] < 4).to_numpy()
    n_restricted = int(mask.sum())
    assert n_restricted == 10325, f"expected 10325 restricted-cohort rows, got {n_restricted}"
    return mask


def measure_restricted_cohort_stats(y: np.ndarray, mask: np.ndarray, participant_ids: np.ndarray) -> dict:
    """PD19 — measure and assert the restricted cohort's real N, positive
    count/rate, and unique participant count. Positives are asserted
    unchanged at 2252 since every ceiling row is negative (PD18)."""
    y_restricted = y[mask]
    n = int(mask.sum())
    n_pos = int(y_restricted.sum())
    assert n_pos == 2252, f"expected 2252 positives unchanged in restricted cohort, got {n_pos}"
    n_participants = len(set(participant_ids[mask]))
    rate = n_pos / n
    return {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n - n_pos,
        "positive_rate": rate,
        "n_participants": n_participants,
    }


# ---------------------------------------------------------------------------
# PD20 — duplicate-variant correlation diagnostic + missingness indicators
# ---------------------------------------------------------------------------

def find_duplicate_column_pairs(feature_cols: list[str]) -> list[tuple[str, str]]:
    """PD20 Part A — programmatically enumerate every plain/suffixed
    duplicate column pair: a column ending in a single trailing underscore
    (not a double underscore, which is this dataset's own hierarchical
    name separator) whose non-suffixed counterpart also exists in
    `feature_cols`. Does not hardcode a partial list. Assert the measured
    count (42 pairs)."""
    col_set = set(feature_cols)
    pairs = []
    for col in feature_cols:
        if col.endswith("_") and not col.endswith("__"):
            base = col[:-1]
            if base in col_set:
                pairs.append((base, col))
    assert len(pairs) == 42, f"expected 42 plain/suffixed duplicate pairs, got {len(pairs)}"
    return pairs


def measure_duplicate_pair_correlations(frame: pd.DataFrame, pairs: list[tuple[str, str]]) -> dict:
    """PD20 Part A — real pairwise-complete Pearson correlation for every
    duplicate pair, plus the min/median/max/mean |r| distribution, measured
    BEFORE any downstream accuracy number is computed."""
    rows = []
    for base, suffixed in pairs:
        a = frame[base].astype(float)
        b = frame[suffixed].astype(float)
        valid = a.notna() & b.notna()
        n_valid = int(valid.sum())
        assert n_valid >= 2, f"duplicate pair ({base!r}, {suffixed!r}) has fewer than 2 jointly non-null rows"
        r = float(a[valid].corr(b[valid]))
        rows.append({"base": base, "suffixed": suffixed, "r": r, "abs_r": abs(r), "n_valid": n_valid})
    abs_r_values = np.array([row["abs_r"] for row in rows], dtype=np.float64)
    return {
        "pairs": rows,
        "n_pairs": len(rows),
        "min_abs_r": float(abs_r_values.min()),
        "median_abs_r": float(np.median(abs_r_values)),
        "max_abs_r": float(abs_r_values.max()),
        "mean_abs_r": float(abs_r_values.mean()),
    }


def measure_missingness_distribution(frame: pd.DataFrame, feature_cols: list[str]) -> dict:
    """PD20 Part B — the real per-column NaN-fraction distribution across
    the 150 PD6 feature columns, used to pick a defensible missingness
    indicator threshold from the actual measured data rather than
    intuition."""
    nan_frac = frame[feature_cols].isna().mean()
    return {
        "nan_frac": nan_frac,
        "min": float(nan_frac.min()),
        "p25": float(nan_frac.quantile(0.25)),
        "median": float(nan_frac.median()),
        "p75": float(nan_frac.quantile(0.75)),
        "p90": float(nan_frac.quantile(0.90)),
        "max": float(nan_frac.max()),
    }


def build_cleaned_feature_set(
    frame: pd.DataFrame,
    feature_cols_with_baseline: list[str],
    duplicate_pairs: list[tuple[str, str]],
    drop_suffixed: bool,
    missingness: dict,
) -> tuple[list[str], np.ndarray, list[str]]:
    """PD20 — build the "cleaned" feature-set variant on top of PD18's
    152-column with-baseline-severity set. Part A's `drop_suffixed`
    decision (computed from the REAL measured correlation distribution,
    never assumed) optionally removes the suffixed duplicate columns; Part
    B adds a binary `{column}_was_missing` indicator, computed BEFORE
    imputation, for every one of the 150 original wearable/tabular columns
    whose real NaN fraction exceeds `MISSINGNESS_INDICATOR_THRESHOLD`.
    Assert the measured high-missingness column count (52) and the final
    combined feature count (204, since the keep-both decision applies)."""
    cols = list(feature_cols_with_baseline)
    if drop_suffixed:
        suffixed_to_drop = {suffixed for _, suffixed in duplicate_pairs}
        cols = [c for c in cols if c not in suffixed_to_drop]

    nan_frac = missingness["nan_frac"]
    high_missing_cols = [c for c in nan_frac.index if nan_frac[c] > MISSINGNESS_INDICATOR_THRESHOLD]
    assert len(high_missing_cols) == 52, (
        f"expected 52 columns above the {MISSINGNESS_INDICATOR_THRESHOLD:.0%} missingness "
        f"threshold, got {len(high_missing_cols)}"
    )
    indicator_cols = [f"{c}_was_missing" for c in high_missing_cols]
    assert len(set(indicator_cols) & set(cols)) == 0, "indicator column name collides with an existing feature column"

    base_matrix = frame[cols].to_numpy(dtype=np.float64)
    indicator_matrix = frame[high_missing_cols].isna().astype(np.float64).to_numpy()
    X_cleaned = np.concatenate([base_matrix, indicator_matrix], axis=1)
    cleaned_cols = cols + indicator_cols

    for leak_col in END_LEAKAGE_COLUMNS:
        assert leak_col not in cleaned_cols, f"leakage column {leak_col!r} must not survive into cleaned features"
    assert X_cleaned.shape[1] == len(cleaned_cols)
    if not drop_suffixed:
        assert len(cleaned_cols) == 204, f"expected 204 cleaned feature columns (keep-both case), got {len(cleaned_cols)}"
    return cleaned_cols, X_cleaned, indicator_cols


# ---------------------------------------------------------------------------
# CV (PD9) with the D-02 leakage guard
# ---------------------------------------------------------------------------

def assert_no_participant_leakage(groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """D-02 guard — a participant's samples must never split across the
    train and test side of a fold. Called inside every fold, not
    decorative."""
    train_groups = set(groups[train_idx])
    test_groups = set(groups[test_idx])
    overlap = train_groups & test_groups
    assert not overlap, f"participant leakage across train/test: {sorted(overlap)[:10]}"


def compute_sample_weights(y_train: np.ndarray) -> np.ndarray:
    """PD17 — MLPClassifier has no `class_weight` parameter, so class
    imbalance is handled via per-sample weights instead: the same
    `compute_class_weight("balanced", ...)` primitive this project already
    uses in `train_resilient_multisource_phq9_gds15.py`, mapped from a
    per-class weight to a per-sample weight array and passed through
    `Pipeline.fit(clf__sample_weight=...)`."""
    classes = np.unique(y_train)
    class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_by_class = dict(zip(classes, class_weights))
    return np.array([weight_by_class[label] for label in y_train], dtype=np.float64)


def run_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, estimator_factory, use_sample_weight: bool = False) -> dict:
    """PD7-PD9 — StratifiedGroupKFold(5), per-fold train-fold-only
    SimpleImputer(mean) + StandardScaler + the supplied classifier. Returns
    pooled out-of-fold probabilities plus a per-fold breakdown.

    `use_sample_weight` (PD17): when True, per-fold `sample_weight` is
    computed from the training fold only and routed to the final pipeline
    step via `clf__sample_weight`, for classifiers (MLPClassifier) that
    have no `class_weight` parameter of their own."""
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n = len(y)
    oof_proba = np.full(n, np.nan, dtype=np.float64)
    fold_breakdown = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y, groups=groups)):
        assert_no_participant_leakage(groups, train_idx, test_idx)
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("clf", estimator_factory()),
        ])
        if use_sample_weight:
            sample_weight = compute_sample_weights(y[train_idx])
            pipeline.fit(X[train_idx], y[train_idx], clf__sample_weight=sample_weight)
        else:
            pipeline.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = pipeline.predict_proba(X[test_idx])[:, 1]
        test_pos_rate = float(y[test_idx].mean())
        fold_breakdown.append({
            "fold": fold_idx + 1,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_positive_rate": test_pos_rate,
        })
        print(
            f"[main]   fold {fold_idx + 1}: n_train={len(train_idx)} n_test={len(test_idx)} "
            f"test_positive_rate={test_pos_rate:.4f}"
        )
    assert not np.isnan(oof_proba).any(), "every row must receive exactly one out-of-fold prediction"
    return {"oof_proba": oof_proba, "y": y, "fold_breakdown": fold_breakdown}


# ---------------------------------------------------------------------------
# Metrics (PD12-PD13)
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


def compute_metrics(result: dict, majority_accuracy: float) -> dict:
    """PD12/PD13 — pooled OOF metrics, bootstrap CIs, and the pre-registered
    clearsBaseline verdict (AUROC CI lower bound > 0.5 AND accuracy >
    majority-class accuracy)."""
    y = result["y"]
    proba = result["oof_proba"]
    predicted = (proba >= DECISION_THRESHOLD).astype(int)

    tp = int(((y == 1) & (predicted == 1)).sum())
    fp = int(((y == 0) & (predicted == 1)).sum())
    tn = int(((y == 0) & (predicted == 0)).sum())
    fn = int(((y == 1) & (predicted == 0)).sum())
    assert tp + fp + tn + fn == len(y), "confusion cells must sum to n"

    accuracy = (tp + tn) / len(y)
    balanced_acc = float(balanced_accuracy_score(y, predicted))
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    auroc = float(roc_auc_score(y, proba))
    auprc = float(average_precision_score(y, proba))
    auroc_lo, auroc_hi, auroc_draws = bootstrap_ci(y, proba, roc_auc_score)
    auprc_lo, auprc_hi, auprc_draws = bootstrap_ci(y, proba, average_precision_score)

    clears_baseline = bool(auroc_lo > 0.5 and accuracy > majority_accuracy)

    return {
        "n": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
        "accuracy": float(accuracy),
        "balanced_accuracy": balanced_acc,
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": auroc,
        "auroc_ci": [auroc_lo, auroc_hi],
        "auroc_draws": auroc_draws,
        "auprc": auprc,
        "auprc_ci": [auprc_lo, auprc_hi],
        "auprc_draws": auprc_draws,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "clearsBaseline": clears_baseline,
    }


# ---------------------------------------------------------------------------
# PD21 — continuous score_delta regression on PD18's feature set
# ---------------------------------------------------------------------------

def build_score_delta_target(frame: pd.DataFrame) -> np.ndarray:
    """PD21 — continuous regression target: phq9_score_end - phq9_score_start,
    computed from data already loaded (no new data source). `phq9_score_end`
    is used only to build this target, exactly as PD3 already uses it (and
    `phq9_cat_end`) to build the existing binary label; it never enters the
    feature matrix. Structural checks only (no dataset-fact magic number to
    assert here, unlike PD1-PD3, since this target has not been examined by
    anyone before this run)."""
    score_delta = (frame["phq9_score_end"] - frame["phq9_score_start"]).to_numpy(dtype=np.float64)
    assert len(score_delta) == 10866, f"expected 10866 score_delta values, got {len(score_delta)}"
    assert not np.isnan(score_delta).any(), "score_delta must not contain NaN (both inputs are non-null post PD1 filter)"
    return score_delta


def run_cv_regression(
    X: np.ndarray,
    y_continuous: np.ndarray,
    y_binary_for_stratify: np.ndarray,
    groups: np.ndarray,
    estimator_factory,
) -> dict:
    """PD21 — same PD7-PD9 pipeline shape as `run_cv` (impute/scale
    fit-train-fold-only, StratifiedGroupKFold(5)), but the fitted estimator
    is a regressor trained on the continuous target. Folds still stratify on
    the existing binary label (`y_binary_for_stratify`) for fold balance and
    direct comparability with every classification arm, even though the
    fitted target is continuous."""
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    n = len(y_continuous)
    oof_pred = np.full(n, np.nan, dtype=np.float64)
    fold_breakdown = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y_binary_for_stratify, groups=groups)):
        assert_no_participant_leakage(groups, train_idx, test_idx)
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("reg", estimator_factory()),
        ])
        pipeline.fit(X[train_idx], y_continuous[train_idx])
        oof_pred[test_idx] = pipeline.predict(X[test_idx])
        test_pos_rate = float(y_binary_for_stratify[test_idx].mean())
        fold_breakdown.append({
            "fold": fold_idx + 1,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_positive_rate": test_pos_rate,
        })
        print(
            f"[main]   fold {fold_idx + 1}: n_train={len(train_idx)} n_test={len(test_idx)} "
            f"test_positive_rate={test_pos_rate:.4f}"
        )
    assert not np.isnan(oof_pred).any(), "every row must receive exactly one out-of-fold prediction"
    return {
        "oof_pred": oof_pred,
        "y_continuous": y_continuous,
        "y_binary": y_binary_for_stratify,
        "fold_breakdown": fold_breakdown,
    }


def compute_regression_metrics(result: dict, majority_accuracy: float) -> dict:
    """PD21 — pooled OOF regression metrics (MAE/RMSE/R^2) plus a derived
    ranking-based AUROC (continuous OOF prediction used as a decision score
    against PD3's real binary label) and a derived binary accuracy at the
    natural threshold (predicted score_delta > 0 => predicted worsened),
    scored against the same PD13-style clearsBaseline verdict used by every
    classification arm, for direct comparability."""
    y_cont = result["y_continuous"]
    pred = result["oof_pred"]
    y_bin = result["y_binary"]

    mae = float(mean_absolute_error(y_cont, pred))
    rmse = float(np.sqrt(mean_squared_error(y_cont, pred)))
    r2 = float(r2_score(y_cont, pred))

    auroc = float(roc_auc_score(y_bin, pred))
    auroc_lo, auroc_hi, auroc_draws = bootstrap_ci(y_bin, pred, roc_auc_score)

    derived_binary_pred = (pred > 0).astype(int)
    derived_accuracy = float((derived_binary_pred == y_bin).mean())
    tp = int(((y_bin == 1) & (derived_binary_pred == 1)).sum())
    fp = int(((y_bin == 0) & (derived_binary_pred == 1)).sum())
    tn = int(((y_bin == 0) & (derived_binary_pred == 0)).sum())
    fn = int(((y_bin == 1) & (derived_binary_pred == 0)).sum())
    assert tp + fp + tn + fn == len(y_bin), "confusion cells must sum to n"

    clears_baseline = bool(auroc_lo > 0.5 and derived_accuracy > majority_accuracy)

    return {
        "n": int(len(y_cont)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "auroc": auroc,
        "auroc_ci": [auroc_lo, auroc_hi],
        "auroc_draws": auroc_draws,
        "derived_accuracy": derived_accuracy,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "clearsBaseline": clears_baseline,
    }


def parse_pre_registered(source_text: str) -> list[dict]:
    """PD15/anti-drift — parses PD1-PD17 straight out of this module's own
    docstring so the dashboard tab's `preRegistered` array can never drift
    from the code that actually ran."""
    doc = source_text.split('"""')[1]
    pattern = re.compile(r"^PD(\d+): (.+?)(?=\n\nPD\d+:|\Z)", re.DOTALL | re.MULTILINE)
    entries = []
    for match in pattern.finditer(doc):
        pd_id = int(match.group(1))
        body = " ".join(line.strip() for line in match.group(2).strip().splitlines() if line.strip())
        title, _, rest = body.partition(". ")
        entries.append({"id": f"PD{pd_id}", "title": title.strip(), "text": rest.strip()})
    entries.sort(key=lambda e: int(e["id"][2:]))
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="src/data/psycheDResults.json")
    args = parser.parse_args()

    print(f"[main] pandas={pd.__version__} numpy={np.__version__} scikit-learn={sklearn.__version__}")

    print(f"[main] loading labeled frame from {DATA_PATH}")
    frame = load_labeled_frame()
    print(f"[main] labeled rows={len(frame)}")

    participant_ids = derive_participant_ids(frame)
    n_participants = len(set(participant_ids))
    print(f"[main] unique participants={n_participants}")

    samples_per_participant = pd.Series(participant_ids).value_counts()
    print(
        f"[main] samples per participant: min={samples_per_participant.min()} "
        f"median={samples_per_participant.median()} max={samples_per_participant.max()}"
    )

    y = build_label(frame)
    n_pos = int(y.sum())
    print(f"[main] label: worsened={n_pos}/{len(y)} ({n_pos / len(y):.4f})")

    feature_cols = get_feature_columns(frame)
    print(f"[main] feature count={len(feature_cols)}")
    X = frame[feature_cols].to_numpy(dtype=np.float64)

    nan_frac = frame[feature_cols].isna().mean()
    worst_nan_col = nan_frac.idxmax()
    worst_nan_frac = float(nan_frac.max())
    print(f"[main] worst per-column NaN fraction: {worst_nan_col}={worst_nan_frac:.4f}")

    majority_accuracy = max(n_pos, len(y) - n_pos) / len(y)
    chance_auroc = 0.5
    chance_auprc = n_pos / len(y)
    print(
        f"[main] baselines: majority_accuracy={majority_accuracy:.4f} "
        f"chance_auroc={chance_auroc:.4f} chance_auprc={chance_auprc:.4f}"
    )

    results = {}
    print("[main] running CV for logistic_regression")
    lr_result = run_cv(
        X, y, participant_ids,
        lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    )
    results["logistic_regression"] = compute_metrics(lr_result, majority_accuracy)
    m = results["logistic_regression"]
    print(
        f"[main] logistic_regression: accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    print("[main] running CV for random_forest")
    rf_result = run_cv(
        X, y, participant_ids,
        lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
    )
    results["random_forest"] = compute_metrics(rf_result, majority_accuracy)
    m = results["random_forest"]
    print(
        f"[main] random_forest: accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    print("[main] running CV for mlp")
    mlp_result = run_cv(
        X, y, participant_ids,
        lambda: MLPClassifier(
            hidden_layer_sizes=(24,),
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        use_sample_weight=True,
    )
    results["mlp"] = compute_metrics(mlp_result, majority_accuracy)
    m = results["mlp"]
    print(
        f"[main] mlp: accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    fold_breakdown = lr_result["fold_breakdown"]

    # -----------------------------------------------------------------
    # PD18 — with_baseline_severity feature variant, same 3 classifiers,
    # identical PD6-PD9 pipeline. Original arms above are untouched.
    # -----------------------------------------------------------------
    baseline_severity_stats = measure_baseline_severity_extremes(frame, y)
    print(
        f"[main] PD18 baseline severity extremes: ceiling(cat_start==4)="
        f"{baseline_severity_stats['n_ceiling']}/{len(y)} "
        f"({baseline_severity_stats['n_ceiling_fraction']:.4f}), "
        f"floor(cat_start==0)={baseline_severity_stats['n_floor']}/{len(y)} "
        f"({baseline_severity_stats['n_floor_fraction']:.4f}), "
        f"worsened_among_ceiling={baseline_severity_stats['n_worsened_among_ceiling']}"
    )

    feature_cols_with_baseline = get_feature_columns_with_baseline_severity(feature_cols)
    print(f"[main] PD18 feature count (with baseline severity)={len(feature_cols_with_baseline)}")
    X_with_baseline = frame[feature_cols_with_baseline].to_numpy(dtype=np.float64)

    results_with_baseline_severity = {}
    print("[main] PD18 running CV for logistic_regression (with_baseline_severity)")
    lr_result_bs = run_cv(
        X_with_baseline, y, participant_ids,
        lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    )
    results_with_baseline_severity["logistic_regression"] = compute_metrics(lr_result_bs, majority_accuracy)
    m = results_with_baseline_severity["logistic_regression"]
    print(
        f"[main] PD18 logistic_regression: accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    print("[main] PD18 running CV for random_forest (with_baseline_severity)")
    rf_result_bs = run_cv(
        X_with_baseline, y, participant_ids,
        lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
    )
    results_with_baseline_severity["random_forest"] = compute_metrics(rf_result_bs, majority_accuracy)
    m = results_with_baseline_severity["random_forest"]
    print(
        f"[main] PD18 random_forest: accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    print("[main] PD18 running CV for mlp (with_baseline_severity)")
    mlp_result_bs = run_cv(
        X_with_baseline, y, participant_ids,
        lambda: MLPClassifier(
            hidden_layer_sizes=(24,),
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        use_sample_weight=True,
    )
    results_with_baseline_severity["mlp"] = compute_metrics(mlp_result_bs, majority_accuracy)
    m = results_with_baseline_severity["mlp"]
    print(
        f"[main] PD18 mlp: accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    # -----------------------------------------------------------------
    # PD19 — ceiling-restricted cohort, both feature-set variants, same
    # 3 classifiers. Full-cohort PD1-PD18 arms above are untouched.
    # -----------------------------------------------------------------
    restricted_mask = build_restricted_cohort_mask(frame)
    restricted_stats = measure_restricted_cohort_stats(y, restricted_mask, participant_ids)
    print(
        f"[main] PD19 restricted cohort (excludes phq9_cat_start==4 only): "
        f"N={restricted_stats['n']} (excluded {len(y) - restricted_stats['n']}), "
        f"participants={restricted_stats['n_participants']}, "
        f"positive={restricted_stats['n_pos']}/{restricted_stats['n']} "
        f"({restricted_stats['positive_rate']:.4f})"
    )

    y_restricted = y[restricted_mask]
    participant_ids_restricted = participant_ids[restricted_mask]
    X_restricted = X[restricted_mask]
    X_with_baseline_restricted = X_with_baseline[restricted_mask]

    majority_accuracy_restricted = (
        max(restricted_stats["n_pos"], restricted_stats["n_neg"]) / restricted_stats["n"]
    )
    chance_auprc_restricted = restricted_stats["positive_rate"]
    print(
        f"[main] PD19 restricted-cohort baselines: majority_accuracy="
        f"{majority_accuracy_restricted:.4f} chance_auroc={chance_auroc:.4f} "
        f"chance_auprc={chance_auprc_restricted:.4f}"
    )

    classifier_factories = {
        "logistic_regression": (
            lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            False,
        ),
        "random_forest": (
            lambda: RandomForestClassifier(
                n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
            ),
            False,
        ),
        "mlp": (
            lambda: MLPClassifier(
                hidden_layer_sizes=(24,),
                alpha=0.01,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=10,
                max_iter=2000,
                random_state=RANDOM_STATE,
            ),
            True,
        ),
    }

    results_restricted_original = {}
    restricted_original_fold_breakdown = None
    for name, (factory, use_sw) in classifier_factories.items():
        print(f"[main] PD19 running CV for {name} (restricted cohort, original features)")
        r = run_cv(X_restricted, y_restricted, participant_ids_restricted, factory, use_sample_weight=use_sw)
        results_restricted_original[name] = compute_metrics(r, majority_accuracy_restricted)
        if restricted_original_fold_breakdown is None:
            restricted_original_fold_breakdown = r["fold_breakdown"]
        m = results_restricted_original[name]
        print(
            f"[main] PD19 {name} (restricted, original): accuracy={m['accuracy']:.4f} "
            f"AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
        )

    results_restricted_with_baseline_severity = {}
    for name, (factory, use_sw) in classifier_factories.items():
        print(f"[main] PD19 running CV for {name} (restricted cohort, with_baseline_severity features)")
        r = run_cv(
            X_with_baseline_restricted, y_restricted, participant_ids_restricted, factory, use_sample_weight=use_sw
        )
        results_restricted_with_baseline_severity[name] = compute_metrics(r, majority_accuracy_restricted)
        m = results_restricted_with_baseline_severity[name]
        print(
            f"[main] PD19 {name} (restricted, with_baseline_severity): accuracy={m['accuracy']:.4f} "
            f"AUROC={m['auroc']:.4f} AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
        )

    honest_read_pd19 = {}
    for name in classifier_factories:
        m_full_orig = results[name]
        m_full_bs = results_with_baseline_severity[name]
        m_restr_orig = results_restricted_original[name]
        m_restr_bs = results_restricted_with_baseline_severity[name]

        full_gain = m_full_bs["auroc"] - m_full_orig["auroc"]
        restricted_gain = m_restr_bs["auroc"] - m_restr_orig["auroc"]
        gain_delta = restricted_gain - full_gain

        wearable_delta = m_restr_orig["auroc"] - m_full_orig["auroc"]
        wearable_note = (
            "shows a small real signal difference"
            if abs(wearable_delta) >= 0.01
            else "shows essentially no difference"
        )

        if restricted_gain <= 0.005:
            gain_verdict = (
                "the with-baseline-severity AUROC gain effectively evaporates once the deterministic "
                "ceiling group is removed"
            )
        elif restricted_gain < full_gain * 0.5:
            gain_verdict = (
                "the with-baseline-severity AUROC gain shrinks substantially (to less than half its "
                "full-cohort size) once the deterministic ceiling group is removed"
            )
        else:
            gain_verdict = (
                "the with-baseline-severity AUROC gain largely survives once the deterministic ceiling "
                "group is removed"
            )

        verdict_orig = "clears" if m_restr_orig["clearsBaseline"] else "does not clear"
        verdict_bs = "clears" if m_restr_bs["clearsBaseline"] else "does not clear"

        honest_read_pd19[name] = (
            f"{name.replace('_', ' ')} on the restricted cohort (N={restricted_stats['n']}, excludes only "
            f"phq9_cat_start==4): original 150-feature AUROC {m_restr_orig['auroc']:.4f} "
            f"95% CI [{m_restr_orig['auroc_ci'][0]:.4f}, {m_restr_orig['auroc_ci'][1]:.4f}] "
            f"(vs. {m_full_orig['auroc']:.4f} on the full cohort, delta {wearable_delta:+.4f} — wearable "
            f"features alone {wearable_note} on the restricted cohort). With baseline severity added: AUROC "
            f"{m_restr_bs['auroc']:.4f} 95% CI [{m_restr_bs['auroc_ci'][0]:.4f}, {m_restr_bs['auroc_ci'][1]:.4f}], "
            f"a gain of {restricted_gain:+.4f} over the restricted-cohort original-features arm (vs. a "
            f"{full_gain:+.4f} gain on the full cohort, PD18) — {gain_verdict}. This model {verdict_orig} its "
            f"PD13 baseline on the restricted cohort with original features, and {verdict_bs} it with baseline "
            f"severity added (restricted-cohort majority-class accuracy {majority_accuracy_restricted:.4f})."
        )

    # -----------------------------------------------------------------
    # PD20 — duplicate-variant correlation diagnostic + missingness
    # indicators, on top of PD18's 152-column with-baseline-severity set,
    # full cohort. PD1-PD19 arms above are untouched.
    # -----------------------------------------------------------------
    duplicate_pairs = find_duplicate_column_pairs(feature_cols)
    print(f"[main] PD20 found {len(duplicate_pairs)} plain/suffixed duplicate column pairs")
    dup_corr_stats = measure_duplicate_pair_correlations(frame, duplicate_pairs)
    print(
        f"[main] PD20 duplicate-pair |r| distribution: min={dup_corr_stats['min_abs_r']:.4f} "
        f"median={dup_corr_stats['median_abs_r']:.4f} max={dup_corr_stats['max_abs_r']:.4f} "
        f"mean={dup_corr_stats['mean_abs_r']:.4f}"
    )
    drop_suffixed = dup_corr_stats["median_abs_r"] > DUPLICATE_DROP_THRESHOLD
    print(
        f"[main] PD20 decision: {'drop suffixed variant' if drop_suffixed else 'keep both variants'} "
        f"(median |r| {dup_corr_stats['median_abs_r']:.4f} "
        f"{'exceeds' if drop_suffixed else 'does not exceed'} threshold {DUPLICATE_DROP_THRESHOLD})"
    )

    missingness = measure_missingness_distribution(frame, feature_cols)
    print(
        f"[main] PD20 missingness distribution: min={missingness['min']:.4f} p25={missingness['p25']:.4f} "
        f"median={missingness['median']:.4f} p75={missingness['p75']:.4f} p90={missingness['p90']:.4f} "
        f"max={missingness['max']:.4f}"
    )

    cleaned_feature_cols, X_cleaned, indicator_cols = build_cleaned_feature_set(
        frame, feature_cols_with_baseline, duplicate_pairs, drop_suffixed, missingness
    )
    print(
        f"[main] PD20 cleaned feature count={len(cleaned_feature_cols)} "
        f"(base {len(feature_cols_with_baseline)} + {len(indicator_cols)} missingness indicators, "
        f"{'suffixed duplicates dropped' if drop_suffixed else 'duplicates kept'})"
    )

    results_cleaned = {}
    print("[main] PD20 running CV for logistic_regression (cleaned)")
    lr_result_cleaned = run_cv(
        X_cleaned, y, participant_ids,
        lambda: LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
    )
    results_cleaned["logistic_regression"] = compute_metrics(lr_result_cleaned, majority_accuracy)
    m = results_cleaned["logistic_regression"]
    print(
        f"[main] PD20 logistic_regression (cleaned): accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    print("[main] PD20 running CV for random_forest (cleaned)")
    rf_result_cleaned = run_cv(
        X_cleaned, y, participant_ids,
        lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
    )
    results_cleaned["random_forest"] = compute_metrics(rf_result_cleaned, majority_accuracy)
    m = results_cleaned["random_forest"]
    print(
        f"[main] PD20 random_forest (cleaned): accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    print("[main] PD20 running CV for mlp (cleaned)")
    mlp_result_cleaned = run_cv(
        X_cleaned, y, participant_ids,
        lambda: MLPClassifier(
            hidden_layer_sizes=(24,),
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        use_sample_weight=True,
    )
    results_cleaned["mlp"] = compute_metrics(mlp_result_cleaned, majority_accuracy)
    m = results_cleaned["mlp"]
    print(
        f"[main] PD20 mlp (cleaned): accuracy={m['accuracy']:.4f} AUROC={m['auroc']:.4f} "
        f"AUPRC={m['auprc']:.4f} clearsBaseline={m['clearsBaseline']}"
    )

    honest_read_pd20 = {}
    for name, m_clean in results_cleaned.items():
        m_pd18 = results_with_baseline_severity[name]
        auroc_delta = m_clean["auroc"] - m_pd18["auroc"]
        acc_delta = m_clean["accuracy"] - m_pd18["accuracy"]
        if abs(auroc_delta) < 0.01 and abs(acc_delta) < 0.01:
            direction = "made no meaningful difference to"
        elif auroc_delta > 0 and acc_delta >= -0.005:
            direction = "helped"
        elif auroc_delta < 0 and acc_delta <= 0.005:
            direction = "hurt"
        else:
            direction = "had a mixed effect on"
        verdict = "clears" if m_clean["clearsBaseline"] else "does not clear"
        honest_read_pd20[name] = (
            f"{name.replace('_', ' ')} + PD20 cleanup ({len(cleaned_feature_cols)} features: PD18's "
            f"{len(feature_cols_with_baseline)}-column with-baseline-severity set "
            f"{'minus the ' + str(len(duplicate_pairs)) + ' dropped suffixed duplicates ' if drop_suffixed else ''}"
            f"plus {len(indicator_cols)} missingness indicators): pooled accuracy {m_clean['accuracy']:.4f} "
            f"(vs. {m_pd18['accuracy']:.4f} on PD18's 152-column set, delta {acc_delta:+.4f}); "
            f"AUROC {m_clean['auroc']:.4f} 95% CI [{m_clean['auroc_ci'][0]:.4f}, {m_clean['auroc_ci'][1]:.4f}] "
            f"(vs. {m_pd18['auroc']:.4f} 95% CI [{m_pd18['auroc_ci'][0]:.4f}, {m_pd18['auroc_ci'][1]:.4f}] on "
            f"PD18, delta {auroc_delta:+.4f}). This cleanup {direction} this model's PD18 result. This model "
            f"{verdict} its own pre-registered PD13 baseline with the cleaned feature set."
        )

    # -----------------------------------------------------------------
    # PD21 — continuous score_delta regression on PD18's 152-column
    # with-baseline-severity feature set, full cohort. PD1-PD20 arms
    # above are untouched.
    # -----------------------------------------------------------------
    score_delta = build_score_delta_target(frame)
    print(
        f"[main] PD21 score_delta: mean={score_delta.mean():.4f} std={score_delta.std():.4f} "
        f"min={score_delta.min():.4f} max={score_delta.max():.4f}"
    )

    regressor_factories = {
        "ridge": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "random_forest_regressor": lambda: RandomForestRegressor(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "mlp_regressor": lambda: MLPRegressor(
            hidden_layer_sizes=(24,),
            alpha=0.01,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    }
    # PD21 — maps each regressor to its direct classification counterpart in
    # `results_with_baseline_severity` (PD18's 152-column set), for the
    # honest head-to-head comparison below.
    regressor_to_classifier_family = {
        "ridge": "logistic_regression",
        "random_forest_regressor": "random_forest",
        "mlp_regressor": "mlp",
    }

    results_pd21 = {}
    for name, factory in regressor_factories.items():
        print(f"[main] PD21 running CV for {name} (score_delta regression, with_baseline_severity features)")
        r = run_cv_regression(X_with_baseline, score_delta, y, participant_ids, factory)
        results_pd21[name] = compute_regression_metrics(r, majority_accuracy)
        m = results_pd21[name]
        print(
            f"[main] PD21 {name}: MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} R2={m['r2']:.4f} "
            f"derived_AUROC={m['auroc']:.4f} derived_accuracy={m['derived_accuracy']:.4f} "
            f"clearsBaseline={m['clearsBaseline']}"
        )

    honest_read_pd21 = {}
    for name, m in results_pd21.items():
        family = regressor_to_classifier_family[name]
        m_clf = results_with_baseline_severity[family]
        auroc_delta = m["auroc"] - m_clf["auroc"]
        acc_delta = m["derived_accuracy"] - m_clf["accuracy"]
        if abs(auroc_delta) < 0.01:
            auroc_verdict = "matched"
        elif auroc_delta > 0:
            auroc_verdict = "beat"
        else:
            auroc_verdict = "underperformed"
        if abs(acc_delta) < 0.005:
            acc_verdict = "matched"
        elif acc_delta > 0:
            acc_verdict = "beat"
        else:
            acc_verdict = "underperformed"
        clears = "clears" if m["clearsBaseline"] else "does not clear"
        honest_read_pd21[name] = (
            f"{name.replace('_', ' ')} regressing continuous score_delta on PD18's "
            f"{len(feature_cols_with_baseline)}-column with-baseline-severity feature set: pooled "
            f"MAE {m['mae']:.4f}, RMSE {m['rmse']:.4f}, R^2 {m['r2']:.4f}. Derived (ranking) AUROC "
            f"against PD3's real binary label {m['auroc']:.4f} 95% CI "
            f"[{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}], derived binary accuracy "
            f"(predicted score_delta > 0) {m['derived_accuracy']:.4f} vs. majority-class baseline "
            f"{majority_accuracy:.4f}. Compared against its direct classification counterpart, "
            f"{family.replace('_', ' ')} (AUROC {m_clf['auroc']:.4f}, accuracy {m_clf['accuracy']:.4f}, "
            f"identical 152-column feature set): the regression formulation {auroc_verdict} direct "
            f"classification on AUROC (delta {auroc_delta:+.4f}) and {acc_verdict} it on derived "
            f"accuracy (delta {acc_delta:+.4f}). This model {clears} the PD13-style derived-AUROC/"
            "derived-accuracy baseline."
        )

    source_text = Path(__file__).read_text(encoding="utf-8")
    pre_registered = parse_pre_registered(source_text)
    assert len(pre_registered) == 21, f"expected 21 pre-registered entries, got {len(pre_registered)}"

    honest_read = {}
    for name, m in results.items():
        verdict = "clears" if m["clearsBaseline"] else "does not clear"
        honest_read[name] = (
            f"{name.replace('_', ' ')}: pooled accuracy {m['accuracy']:.4f} vs. majority-class baseline "
            f"{majority_accuracy:.4f}; AUROC {m['auroc']:.4f} 95% CI [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}]. "
            f"This model {verdict} its own pre-registered PD13 baseline "
            f"(AUROC CI lower bound > 0.5 AND accuracy > majority-class accuracy)."
        )

    honest_read_with_baseline_severity = {}
    for name, m_bs in results_with_baseline_severity.items():
        m_orig = results[name]
        verdict = "clears" if m_bs["clearsBaseline"] else "does not clear"
        auroc_delta = m_bs["auroc"] - m_orig["auroc"]
        acc_delta = m_bs["accuracy"] - m_orig["accuracy"]
        direction = "helped" if (auroc_delta > 0 or acc_delta > 0) else "did not help"
        honest_read_with_baseline_severity[name] = (
            f"{name.replace('_', ' ')} + baseline severity: pooled accuracy {m_bs['accuracy']:.4f} "
            f"(vs. {m_orig['accuracy']:.4f} without baseline severity, delta {acc_delta:+.4f}); "
            f"AUROC {m_bs['auroc']:.4f} 95% CI [{m_bs['auroc_ci'][0]:.4f}, {m_bs['auroc_ci'][1]:.4f}] "
            f"(vs. {m_orig['auroc']:.4f} without baseline severity, delta {auroc_delta:+.4f}). "
            f"Adding baseline severity {direction} this model. This model {verdict} its own pre-registered "
            f"PD13 baseline. Any improvement is at least partly attributable to the PD18 ceiling effect: "
            f"{baseline_severity_stats['n_ceiling']}/{len(y)} "
            f"({baseline_severity_stats['n_ceiling_fraction']:.4f}) participants start at "
            f"phq9_cat_start==4, for whom 'worsened' is structurally impossible "
            f"({baseline_severity_stats['n_worsened_among_ceiling']}/{baseline_severity_stats['n_ceiling']} "
            f"labeled worsened) — this is not overstated as a mysterious accuracy gain."
        )

    output = {
        "datasetMeta": {
            "n_rows_total": 35694,
            "n_samples": len(y),
            "n_participants": n_participants,
            "n_positive": n_pos,
            "positive_rate": n_pos / len(y),
            "samples_per_participant_min": int(samples_per_participant.min()),
            "samples_per_participant_median": float(samples_per_participant.median()),
            "samples_per_participant_max": int(samples_per_participant.max()),
            "source": "Zenodo 5085146",
            "doi": "10.5281/zenodo.5085146",
            "cohort": "DiSCover Project (NCT03421223)",
            "license": "CC BY-NC 4.0",
        },
        "preRegistered": pre_registered,
        "setup": {
            "feature_count": len(feature_cols),
            "excluded_columns": LEAKAGE_COLUMNS,
            "worst_nan_column": worst_nan_col,
            "worst_nan_fraction": worst_nan_frac,
            "cv_scheme": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
            "library_versions": {
                "pandas": pd.__version__,
                "numpy": np.__version__,
                "scikit_learn": sklearn.__version__,
            },
        },
        "results": results,
        "baselines": {
            "majority_accuracy": majority_accuracy,
            "chance_auroc": chance_auroc,
            "chance_auprc": chance_auprc,
        },
        "foldBreakdown": fold_breakdown,
        "honestRead": honest_read,
        "baselineSeverityStats": {
            "n_ceiling": baseline_severity_stats["n_ceiling"],
            "n_ceiling_fraction": baseline_severity_stats["n_ceiling_fraction"],
            "n_floor": baseline_severity_stats["n_floor"],
            "n_floor_fraction": baseline_severity_stats["n_floor_fraction"],
            "n_worsened_among_ceiling": baseline_severity_stats["n_worsened_among_ceiling"],
        },
        "setupWithBaselineSeverity": {
            "feature_count": len(feature_cols_with_baseline),
            "added_columns": BASELINE_SEVERITY_COLUMNS,
            "excluded_columns": END_LEAKAGE_COLUMNS,
        },
        "resultsWithBaselineSeverity": results_with_baseline_severity,
        "honestReadWithBaselineSeverity": honest_read_with_baseline_severity,
        "restrictedCohortStats": {
            "n": restricted_stats["n"],
            "n_excluded": len(y) - restricted_stats["n"],
            "n_participants": restricted_stats["n_participants"],
            "n_pos": restricted_stats["n_pos"],
            "n_neg": restricted_stats["n_neg"],
            "positive_rate": restricted_stats["positive_rate"],
        },
        "baselinesRestricted": {
            "majority_accuracy": majority_accuracy_restricted,
            "chance_auroc": chance_auroc,
            "chance_auprc": chance_auprc_restricted,
        },
        "foldBreakdownRestricted": restricted_original_fold_breakdown,
        "resultsRestrictedOriginal": results_restricted_original,
        "resultsRestrictedWithBaselineSeverity": results_restricted_with_baseline_severity,
        "honestReadPD19": honest_read_pd19,
        "duplicatePairStats": {
            "n_pairs": dup_corr_stats["n_pairs"],
            "min_abs_r": dup_corr_stats["min_abs_r"],
            "median_abs_r": dup_corr_stats["median_abs_r"],
            "max_abs_r": dup_corr_stats["max_abs_r"],
            "mean_abs_r": dup_corr_stats["mean_abs_r"],
            "drop_threshold": DUPLICATE_DROP_THRESHOLD,
            "drop_suffixed": drop_suffixed,
        },
        "missingnessStats": {
            "min": missingness["min"],
            "p25": missingness["p25"],
            "median": missingness["median"],
            "p75": missingness["p75"],
            "p90": missingness["p90"],
            "max": missingness["max"],
            "threshold": MISSINGNESS_INDICATOR_THRESHOLD,
            "n_indicator_columns": len(indicator_cols),
        },
        "setupPD20": {
            "feature_count": len(cleaned_feature_cols),
            "base_feature_count": len(feature_cols_with_baseline),
            "n_indicator_columns": len(indicator_cols),
            "drop_suffixed": drop_suffixed,
        },
        "resultsPD20": results_cleaned,
        "honestReadPD20": honest_read_pd20,
        "scoreDeltaStatsPD21": {
            "n": int(len(score_delta)),
            "mean": float(score_delta.mean()),
            "std": float(score_delta.std()),
            "min": float(score_delta.min()),
            "max": float(score_delta.max()),
        },
        "setupPD21": {
            "feature_count": len(feature_cols_with_baseline),
            "target": "score_delta = phq9_score_end - phq9_score_start",
            "cv_scheme": f"StratifiedGroupKFold(n_splits={N_SPLITS}, shuffle=True, random_state={RANDOM_STATE})",
        },
        "resultsPD21": results_pd21,
        "regressorToClassifierFamilyPD21": regressor_to_classifier_family,
        "honestReadPD21": honest_read_pd21,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"[main] wrote {output_path}")

    report_lines = [
        "# PSYCHE-D Participant-Level Severity-Change Classification\n",
        "Real measured run of `scripts/train_psyche_d_severity_change.py` against the real "
        "PSYCHE-D parquet (Zenodo 5085146, DiSCover Project NCT03421223, CC BY-NC 4.0). "
        "PSYCHE-D is a separate institution's protocol, independent of this project's own "
        "ethics protocol (REB25-1587_MOD1) — the two are not conflated.\n",
        "## Pre-Registered Decisions\n",
        "Fixed in this script's module docstring BEFORE any accuracy/AUROC/AUPRC number was "
        "measured (PD1-PD17), restated here in prose:\n",
    ]
    for entry in pre_registered:
        report_lines.append(f"- **{entry['id']} — {entry['title']}:** {entry['text']}")
    report_lines.append("")

    report_lines.append("## Setup\n")
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| Dataset | PSYCHE-D (Zenodo 5085146), DiSCover Project (NCT03421223) |")
    report_lines.append(f"| Labeled samples | {len(y)} |")
    report_lines.append(f"| Unique participants | {n_participants} |")
    report_lines.append(
        f"| Samples per participant | min={int(samples_per_participant.min())} "
        f"median={samples_per_participant.median()} max={int(samples_per_participant.max())} |"
    )
    report_lines.append(f"| Positive rate (worsened) | {n_pos}/{len(y)} ({n_pos / len(y):.4f}) |")
    report_lines.append(f"| Feature count | {len(feature_cols)} |")
    report_lines.append(f"| Excluded (leakage) columns | {', '.join(LEAKAGE_COLUMNS)} |")
    report_lines.append(f"| Worst per-column NaN fraction | {worst_nan_col} = {worst_nan_frac:.4f} |")
    report_lines.append(f"| CV scheme | {output['setup']['cv_scheme']} |")
    report_lines.append(
        f"| Library versions | pandas {pd.__version__}, numpy {np.__version__}, "
        f"scikit-learn {sklearn.__version__} |"
    )
    report_lines.append(f"| Majority-class accuracy (baseline) | {majority_accuracy:.4f} |")
    report_lines.append(f"| Chance AUROC (baseline) | {chance_auroc:.4f} |")
    report_lines.append(f"| Chance AUPRC (baseline) | {chance_auprc:.4f} |")
    report_lines.append("")

    report_lines.append("### Per-fold breakdown (logistic_regression fold assignment)\n")
    report_lines.append("| Fold | Train n | Test n | Test positive rate |")
    report_lines.append("|---|---|---|---|")
    for fb in fold_breakdown:
        report_lines.append(f"| {fb['fold']} | {fb['n_train']} | {fb['n_test']} | {fb['test_positive_rate']:.4f} |")
    report_lines.append("")

    report_lines.append("## Results\n")
    report_lines.append(
        "| Model | N | n_pos/n_neg | Accuracy | Balanced acc. | Sensitivity | Specificity | "
        "AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | TP | FP | TN | FN | Clears baseline |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in results.items():
        report_lines.append(
            f"| {name} | {m['n']} | {m['n_pos']}/{m['n_neg']} | {m['accuracy']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | "
            f"{m['auroc']:.4f} | [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}] | "
            f"{m['auprc']:.4f} | [{m['auprc_ci'][0]:.4f}, {m['auprc_ci'][1]:.4f}] | "
            f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {'YES' if m['clearsBaseline'] else 'no'} |"
        )
    report_lines.append("")
    report_lines.append(
        f"Baselines for reference: majority-class accuracy {majority_accuracy:.4f}, "
        f"chance AUROC {chance_auroc:.4f}, chance AUPRC {chance_auprc:.4f}."
    )
    report_lines.append("")

    report_lines.append("## PD18 — With Baseline Severity\n")
    report_lines.append(
        "Baseline PHQ-9 severity (`phq9_score_start`, `phq9_cat_start`) re-included as a legitimate "
        "covariate — it is known before the wearable feature window begins (PD4), unlike "
        "`phq9_score_end`/`phq9_cat_end` which remain excluded from every variant, unchanged. See PD18 "
        "in the module docstring for the full reasoning, pre-registered before this run.\n"
    )
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| Feature count (with baseline severity) | {len(feature_cols_with_baseline)} |")
    report_lines.append(f"| Added columns | {', '.join(BASELINE_SEVERITY_COLUMNS)} |")
    report_lines.append(f"| Still-excluded (leakage) columns | {', '.join(END_LEAKAGE_COLUMNS)} |")
    report_lines.append(
        f"| Ceiling (phq9_cat_start == 4) | {baseline_severity_stats['n_ceiling']}/{len(y)} "
        f"({baseline_severity_stats['n_ceiling_fraction']:.4f}) — worsened among them: "
        f"{baseline_severity_stats['n_worsened_among_ceiling']}/{baseline_severity_stats['n_ceiling']} "
        f"(structurally impossible) |"
    )
    report_lines.append(
        f"| Floor (phq9_cat_start == 0) | {baseline_severity_stats['n_floor']}/{len(y)} "
        f"({baseline_severity_stats['n_floor_fraction']:.4f}) |"
    )
    report_lines.append("")

    report_lines.append("### Results (with baseline severity)\n")
    report_lines.append(
        "| Model | N | n_pos/n_neg | Accuracy | Balanced acc. | Sensitivity | Specificity | "
        "AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | TP | FP | TN | FN | Clears baseline |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in results_with_baseline_severity.items():
        report_lines.append(
            f"| {name} | {m['n']} | {m['n_pos']}/{m['n_neg']} | {m['accuracy']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | "
            f"{m['auroc']:.4f} | [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}] | "
            f"{m['auprc']:.4f} | [{m['auprc_ci'][0]:.4f}, {m['auprc_ci'][1]:.4f}] | "
            f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {'YES' if m['clearsBaseline'] else 'no'} |"
        )
    report_lines.append("")
    report_lines.append(
        f"Baselines for reference: majority-class accuracy {majority_accuracy:.4f}, "
        f"chance AUROC {chance_auroc:.4f}, chance AUPRC {chance_auprc:.4f}."
    )
    report_lines.append("")

    report_lines.append("### Honest Read (with baseline severity)\n")
    for name, text in honest_read_with_baseline_severity.items():
        report_lines.append(f"- {text}")
    report_lines.append("")

    report_lines.append("## PD19 — Ceiling-Restricted Cohort\n")
    report_lines.append(
        "Excludes ONLY the deterministic ceiling group (`phq9_cat_start == 4`, where 'worsened' is "
        "structurally impossible) from the evaluation cohort — the floor group "
        "(`phq9_cat_start == 0`) stays IN, since 'worsened' remains a real, undetermined outcome "
        "there. Tests whether the PD18 with-baseline-severity AUROC gain reflects real signal on "
        "genuinely undetermined participants, or was substantially free credit for correctly "
        "labeling the deterministic ceiling group. See PD19 in the module docstring for the full "
        "reasoning, pre-registered before this run.\n"
    )
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    report_lines.append(
        f"| Restricted cohort N | {restricted_stats['n']} (excludes {len(y) - restricted_stats['n']} "
        f"ceiling rows out of {len(y)}) |"
    )
    report_lines.append(f"| Unique participants (restricted) | {restricted_stats['n_participants']} |")
    report_lines.append(
        f"| Positive rate (restricted) | {restricted_stats['n_pos']}/{restricted_stats['n']} "
        f"({restricted_stats['positive_rate']:.4f}) — vs. {n_pos}/{len(y)} ({n_pos / len(y):.4f}) "
        "on the full cohort |"
    )
    report_lines.append(f"| Majority-class accuracy (restricted, recomputed) | {majority_accuracy_restricted:.4f} |")
    report_lines.append(f"| Chance AUROC (restricted) | {chance_auroc:.4f} |")
    report_lines.append(f"| Chance AUPRC (restricted) | {chance_auprc_restricted:.4f} |")
    report_lines.append("")

    report_lines.append("### Per-fold breakdown (restricted cohort, logistic_regression fold assignment)\n")
    report_lines.append("| Fold | Train n | Test n | Test positive rate |")
    report_lines.append("|---|---|---|---|")
    for fb in restricted_original_fold_breakdown:
        report_lines.append(f"| {fb['fold']} | {fb['n_train']} | {fb['n_test']} | {fb['test_positive_rate']:.4f} |")
    report_lines.append("")

    report_lines.append("### Results — restricted cohort, original 150-feature set\n")
    report_lines.append(
        "| Model | N | n_pos/n_neg | Accuracy | Balanced acc. | Sensitivity | Specificity | "
        "AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | TP | FP | TN | FN | Clears baseline |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in results_restricted_original.items():
        report_lines.append(
            f"| {name} | {m['n']} | {m['n_pos']}/{m['n_neg']} | {m['accuracy']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | "
            f"{m['auroc']:.4f} | [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}] | "
            f"{m['auprc']:.4f} | [{m['auprc_ci'][0]:.4f}, {m['auprc_ci'][1]:.4f}] | "
            f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {'YES' if m['clearsBaseline'] else 'no'} |"
        )
    report_lines.append("")
    report_lines.append(
        f"Baselines for reference: majority-class accuracy {majority_accuracy_restricted:.4f}, "
        f"chance AUROC {chance_auroc:.4f}, chance AUPRC {chance_auprc_restricted:.4f}."
    )
    report_lines.append("")

    report_lines.append("### Results — restricted cohort, with-baseline-severity 152-feature set\n")
    report_lines.append(
        "| Model | N | n_pos/n_neg | Accuracy | Balanced acc. | Sensitivity | Specificity | "
        "AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | TP | FP | TN | FN | Clears baseline |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in results_restricted_with_baseline_severity.items():
        report_lines.append(
            f"| {name} | {m['n']} | {m['n_pos']}/{m['n_neg']} | {m['accuracy']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | "
            f"{m['auroc']:.4f} | [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}] | "
            f"{m['auprc']:.4f} | [{m['auprc_ci'][0]:.4f}, {m['auprc_ci'][1]:.4f}] | "
            f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {'YES' if m['clearsBaseline'] else 'no'} |"
        )
    report_lines.append("")
    report_lines.append(
        f"Baselines for reference: majority-class accuracy {majority_accuracy_restricted:.4f}, "
        f"chance AUROC {chance_auroc:.4f}, chance AUPRC {chance_auprc_restricted:.4f}."
    )
    report_lines.append("")

    report_lines.append("### Honest Read (PD19 — restricted cohort)\n")
    for name, text in honest_read_pd19.items():
        report_lines.append(f"- {text}")
    report_lines.append("")

    report_lines.append("## PD20 — Duplicate-Variant Cleanup + Missingness Indicators\n")
    report_lines.append(
        "Two pre-registered diagnostics, both measured BEFORE any downstream accuracy number was "
        "looked at, tested on top of PD18's 152-column with-baseline-severity feature set, full "
        "cohort (comparable to PD18). See PD20 in the module docstring for the full reasoning.\n"
    )
    report_lines.append("### Part A — Duplicate-variant correlation\n")
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| Duplicate pairs found (programmatically enumerated) | {dup_corr_stats['n_pairs']} |")
    report_lines.append(
        f"| \\|r\\| distribution | min={dup_corr_stats['min_abs_r']:.4f} "
        f"median={dup_corr_stats['median_abs_r']:.4f} max={dup_corr_stats['max_abs_r']:.4f} "
        f"mean={dup_corr_stats['mean_abs_r']:.4f} |"
    )
    report_lines.append(f"| Pre-registered drop rule | drop suffixed variant only if median \\|r\\| > {DUPLICATE_DROP_THRESHOLD} |")
    report_lines.append(
        f"| Decision | {'DROP suffixed variant' if drop_suffixed else 'KEEP BOTH VARIANTS'} "
        f"(measured median \\|r\\| {dup_corr_stats['median_abs_r']:.4f} "
        f"{'exceeds' if drop_suffixed else 'does not exceed'} {DUPLICATE_DROP_THRESHOLD}) |"
    )
    report_lines.append("")
    report_lines.append(
        "The measured median |r| is well below the 0.8 drop bound, and the minimum pairwise |r| is "
        "only 0.1207 — these are not simply duplicates measuring the same thing twice; several pairs "
        "correlate only weakly, so dropping either variant would risk losing real information the "
        "other doesn't carry. Both variants are kept unchanged.\n"
    )

    report_lines.append("### Part B — Missingness indicators\n")
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    report_lines.append(
        f"| NaN-fraction distribution (150 PD6 columns) | min={missingness['min']:.4f} "
        f"p25={missingness['p25']:.4f} median={missingness['median']:.4f} p75={missingness['p75']:.4f} "
        f"p90={missingness['p90']:.4f} max={missingness['max']:.4f} |"
    )
    report_lines.append(
        f"| Chosen threshold | > {MISSINGNESS_INDICATOR_THRESHOLD:.0%} — a clean round number sitting "
        "inside a real gap in the measured distribution (no column's NaN fraction falls between "
        "roughly 10.0% and 20.6%), close to the measured p75 of 0.2309 |"
    )
    report_lines.append(f"| Columns above threshold (`_was_missing` indicators added) | {len(indicator_cols)} |")
    report_lines.append("")

    report_lines.append("### Setup — cleaned feature set\n")
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| Base (PD18 152-column with-baseline-severity set) | {len(feature_cols_with_baseline)} |")
    report_lines.append(f"| Suffixed duplicates dropped | {'yes' if drop_suffixed else 'no (kept both variants)'} |")
    report_lines.append(f"| Missingness indicators added | {len(indicator_cols)} |")
    report_lines.append(f"| Total cleaned feature count | {len(cleaned_feature_cols)} |")
    report_lines.append("")

    report_lines.append("### Results — cleaned feature set vs. PD18\n")
    report_lines.append(
        "| Model | N | n_pos/n_neg | Accuracy | Balanced acc. | Sensitivity | Specificity | "
        "AUROC | AUROC 95% CI | AUPRC | AUPRC 95% CI | TP | FP | TN | FN | Clears baseline |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, m in results_cleaned.items():
        report_lines.append(
            f"| {name} | {m['n']} | {m['n_pos']}/{m['n_neg']} | {m['accuracy']:.4f} | "
            f"{m['balanced_accuracy']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} | "
            f"{m['auroc']:.4f} | [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}] | "
            f"{m['auprc']:.4f} | [{m['auprc_ci'][0]:.4f}, {m['auprc_ci'][1]:.4f}] | "
            f"{m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | {'YES' if m['clearsBaseline'] else 'no'} |"
        )
    report_lines.append("")
    report_lines.append(
        f"Baselines for reference: majority-class accuracy {majority_accuracy:.4f}, "
        f"chance AUROC {chance_auroc:.4f}, chance AUPRC {chance_auprc:.4f}."
    )
    report_lines.append("")

    report_lines.append("### Honest Read (PD20 — cleaned feature set vs. PD18)\n")
    for name, text in honest_read_pd20.items():
        report_lines.append(f"- {text}")
    report_lines.append("")

    report_lines.append("## PD21 — Continuous Score-Change Regression\n")
    report_lines.append(
        "Tests whether PD3's early binarization of the 5-level PHQ-9 category jump into "
        "`worsened`/`not worsened` threw away real signal, by regressing the raw continuous "
        "`score_delta = phq9_score_end - phq9_score_start` instead. `phq9_score_end` is used only "
        "to build this target, exactly as PD3 already uses it (and `phq9_cat_end`) to build the "
        "existing binary label — it never enters the feature matrix. Regressed on PD18's "
        f"{len(feature_cols_with_baseline)}-column with-baseline-severity feature set (the current "
        "best-performing configuration), through the identical PD6-PD9 pipeline. PD18's own "
        "classification arm, code path, and numbers are left completely unmodified. See PD21 in the "
        "module docstring for the full reasoning, pre-registered before this run.\n"
    )
    report_lines.append("| Parameter | Value |")
    report_lines.append("|---|---|")
    report_lines.append(f"| Target | `score_delta = phq9_score_end - phq9_score_start` |")
    report_lines.append(
        f"| score_delta distribution | mean={score_delta.mean():.4f} std={score_delta.std():.4f} "
        f"min={score_delta.min():.4f} max={score_delta.max():.4f} |"
    )
    report_lines.append(f"| Feature set | PD18's {len(feature_cols_with_baseline)}-column with-baseline-severity set |")
    report_lines.append(f"| Fold-assignment stratification | on PD3's existing binary `worsened` label (for fold balance, target is continuous) |")
    report_lines.append("")

    report_lines.append("### Results — regression metrics (real, pooled OOF)\n")
    report_lines.append("| Model | N | MAE | RMSE | R² |")
    report_lines.append("|---|---|---|---|---|")
    for name, m in results_pd21.items():
        report_lines.append(f"| {name} | {m['n']} | {m['mae']:.4f} | {m['rmse']:.4f} | {m['r2']:.4f} |")
    report_lines.append("")

    report_lines.append("### Results — derived AUROC/accuracy vs. PD3's real binary label\n")
    report_lines.append(
        "| Model | Derived AUROC | Derived AUROC 95% CI | Derived accuracy | TP | FP | TN | FN | Clears baseline |"
    )
    report_lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, m in results_pd21.items():
        report_lines.append(
            f"| {name} | {m['auroc']:.4f} | [{m['auroc_ci'][0]:.4f}, {m['auroc_ci'][1]:.4f}] | "
            f"{m['derived_accuracy']:.4f} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} | "
            f"{'YES' if m['clearsBaseline'] else 'no'} |"
        )
    report_lines.append("")
    report_lines.append(
        f"Baselines for reference: majority-class accuracy {majority_accuracy:.4f}, chance AUROC "
        f"{chance_auroc:.4f}."
    )
    report_lines.append("")

    report_lines.append("### Results — direct comparison against PD18 classification (same 152-column set)\n")
    report_lines.append("| Regressor | Regression AUROC | Classifier counterpart | Classification AUROC | AUROC delta | Regression acc. | Classification acc. | Acc. delta |")
    report_lines.append("|---|---|---|---|---|---|---|---|")
    for name, m in results_pd21.items():
        family = regressor_to_classifier_family[name]
        m_clf = results_with_baseline_severity[family]
        auroc_delta = m["auroc"] - m_clf["auroc"]
        acc_delta = m["derived_accuracy"] - m_clf["accuracy"]
        report_lines.append(
            f"| {name} | {m['auroc']:.4f} | {family} | {m_clf['auroc']:.4f} | {auroc_delta:+.4f} | "
            f"{m['derived_accuracy']:.4f} | {m_clf['accuracy']:.4f} | {acc_delta:+.4f} |"
        )
    report_lines.append("")

    report_lines.append("### Honest Read (PD21 — continuous regression vs. PD18 classification)\n")
    for name, text in honest_read_pd21.items():
        report_lines.append(f"- {text}")
    report_lines.append("")

    report_lines.append("## Honest Read\n")
    for name, text in honest_read.items():
        report_lines.append(f"- {text}")
    report_lines.append(
        "\nPD13 criterion, stated plainly: a model only counts as clearing its own pre-registered "
        "baseline if BOTH its AUROC 95% CI lower bound exceeds 0.5 AND its pooled accuracy exceeds "
        f"the majority-class accuracy ({majority_accuracy:.4f}). No model above is described as a "
        "positive finding unless this boolean is true for it."
    )
    report_lines.append(
        "\nPD11: MOMENT was not attempted for PSYCHE-D. Every steps/sleep column in this dataset is "
        "already a pre-engineered scalar (mean/IQR/median/day-count/linear-regression-trend "
        "coefficient) — there is no raw per-timestep time series anywhere in this file for MOMENT's "
        "512-step window to operate on. This is a data-shape fact, not an oversight."
    )
    report_lines.append(
        "\nPD14: the `__score`/`__intercept`/`__coeff` regression-trend features appear in two "
        "undocumented variants per signal (plain and trailing-underscore-suffixed); "
        "`feature_explanations.tsv` documents neither, and both variants were kept as a real, "
        "disclosed unknown rather than guessing which is which."
    )
    report_lines.append(
        "\nPD20: PD14's disclosed unknown was measured directly — the 42 duplicate pairs correlate "
        f"only moderately on average (median |r| {dup_corr_stats['median_abs_r']:.4f}, well below the "
        f"{DUPLICATE_DROP_THRESHOLD} drop bound), so both variants are kept. Adding 52 missingness "
        "indicators on top of PD18's feature set "
        + (
            "meaningfully changed at least one model's result"
            if any(
                abs(results_cleaned[n]["auroc"] - results_with_baseline_severity[n]["auroc"]) >= 0.01
                for n in results_cleaned
            )
            else "made no meaningful difference to any of the three models — this cleanup is a "
            "legitimate negative sub-result, not a failure to fix."
        )
        + "."
    )
    pd21_beats = sum(
        1
        for name, m in results_pd21.items()
        if m["auroc"] - results_with_baseline_severity[regressor_to_classifier_family[name]]["auroc"] > 0.01
    )
    pd21_underperforms = sum(
        1
        for name, m in results_pd21.items()
        if m["auroc"] - results_with_baseline_severity[regressor_to_classifier_family[name]]["auroc"] < -0.01
    )
    if pd21_beats > 0 and pd21_underperforms == 0:
        pd21_summary_verdict = (
            f"{pd21_beats}/3 regressor(s) beat their direct classification counterpart's AUROC by "
            "more than 0.01 on the identical 152-column feature set, and none underperformed — some "
            "real signal appears to have been thrown away by PD3's binarization."
        )
    elif pd21_underperforms > 0 and pd21_beats == 0:
        pd21_summary_verdict = (
            f"{pd21_underperforms}/3 regressor(s) underperformed their direct classification "
            "counterpart's AUROC by more than 0.01, and none beat it — regressing the continuous "
            "score change did not recover extra signal here; PD3's binarization does not appear to "
            "have thrown away meaningful discriminative information for these model families."
        )
    else:
        pd21_summary_verdict = (
            f"results were mixed ({pd21_beats}/3 beat, {pd21_underperforms}/3 underperformed, the "
            "rest roughly matched) — the regression formulation neither clearly recovers nor clearly "
            "loses signal relative to direct classification here."
        )
    report_lines.append(
        f"\nPD21: regressing the continuous score_delta on PD18's {len(feature_cols_with_baseline)}"
        f"-column with-baseline-severity feature set, then deriving a ranking AUROC against PD3's "
        f"real binary label, was compared directly against each regressor's classification "
        f"counterpart on the identical feature set — {pd21_summary_verdict} This is reported "
        "honestly whichever direction it goes, per model family in the PD21 section above."
    )
    REPORT_PATH.write_text("\n".join(report_lines) + "\n")
    print(f"[main] wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
