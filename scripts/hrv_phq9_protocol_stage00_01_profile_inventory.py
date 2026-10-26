#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stage 0 (dataset profile + applicability) and Stage 1 (inventory + raw-data gate)
for the HRV + PHQ-9 (Ref 2) protocol run.
Outputs: hrv_phq9_protocol_profile.json, hrv_phq9_protocol_inventory.json
"""
from __future__ import annotations

import gzip
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrv_phq9_protocol_common as C  # noqa: E402


def readme_text() -> str:
    x = zipfile.ZipFile(C.DATA / "README.docx").read("word/document.xml").decode()
    return re.sub(r"<[^>]+>", " ", re.sub(r"</w:p>", "\n", x))


def sample_interval_raw(fname: str, n: int = 20000) -> float:
    with gzip.open(C.DATA / "raw_data" / fname, "rt") as fh:
        df = pd.read_csv(fh, nrows=n)
    d = df.groupby("deviceId")["ts"].diff().dropna()
    return float(d.median() / 1000.0)


def main() -> None:
    survey = pd.read_csv(C.SURVEY_CSV)
    s = C.load_sensor_rows()
    diary = pd.read_csv(C.DIARY_CSV)
    seg = C.build_segments(force=True)

    # ---------------- Stage 1 inventory
    rpp = s.groupby("deviceId").size()
    miss = s[C.SENSOR_CHANNELS + ["missingness_score"]].isna().mean().sort_values(ascending=False)
    local = pd.to_datetime(s["ts_start"] + C.KST_OFFSET_MS, unit="ms")
    cal = C.calendar_rows(s)
    # old anchor used by the earlier train_hrv_phq9_* scripts: day 1 = first sensor row of each device
    first = s.groupby("deviceId")["ts_start"].transform("min")
    old_day = (s["ts_start"] - first) // 86_400_000 + 1
    old_keep = (old_day >= 1) & (old_day <= 28)
    first_local = pd.to_datetime(s.groupby("deviceId")["ts_start"].min() + C.KST_OFFSET_MS, unit="ms")
    offset_days = (first_local.dt.normalize() - C.STUDY_DAY1).dt.days
    rows_seg = cal.groupby(["deviceId", "segment"]).size().unstack(fill_value=0)

    descr = readme_text()
    agg_words = sorted({w for w in ["averaged per the 5-minute interval", "mean", "average", "daily", "per"] if w in descr})
    raw_files = sorted(p.name for p in (C.DATA / "raw_data").glob("*.csv.gz"))
    raw_sizes = {p.name: p.stat().st_size for p in (C.DATA / "raw_data").glob("*.csv.gz")}
    hrm_dt = sample_interval_raw("hrm.csv.gz")
    lit_dt = sample_interval_raw("lit.csv.gz")
    agg_dt = float(((s["ts_end"] - s["ts_start"]) / 1000).median())

    data_dict = []
    groups = {"survey": list(survey.columns), "sensor_hrv": [c for c in s.columns if c in pd.read_csv(C.SENSOR_CSV, nrows=1).columns],
              "sleep_diary": list(diary.columns)}
    for src, cols in groups.items():
        frame = {"survey": survey, "sensor_hrv": s, "sleep_diary": diary}[src]
        for c in cols:
            role = ("label" if c in C.LEAK_COLS and c.startswith("PHQ9") else "leak (concurrent with outcome)" if c in C.LEAK_COLS
                    else "id/time" if c in ("deviceId", "userId", "ts_start", "ts_end", "date") else
                    "baseline covariate" if c in C.BASELINE_COLS + C.DEMO_COLS else "QC" if c == "missingness_score" else "feature")
            data_dict.append({"source": src, "column": c, "dtype": str(frame[c].dtype), "role": role,
                              "pct_missing": float(frame[c].isna().mean() * 100)})

    inventory = {
        "survey_shape": list(survey.shape), "sensor_hrv_shape": [int(len(s)), 28], "sleep_diary_shape": list(diary.shape),
        "n_participants": int(survey["deviceId"].nunique()),
        "sensor_rows_per_participant": {"min": int(rpp.min()), "median": float(rpp.median()), "max": int(rpp.max())},
        "sensor_date_range_kst": [str(local.min()), str(local.max())],
        "sleep_diary_date_range": [diary["date"].min(), diary["date"].max()],
        "sleep_diary_days_per_participant": sorted(diary.groupby("userId").size().unique().tolist()),
        "calendar_anchor": {
            "rule": "study day 1 = 2021-03-09 KST (first sleep-diary date, identical for all 49 participants); day 28 = 2021-04-05",
            "rows_in_calendar_window": int(len(cal)),
            "first_sensor_row_offset_days_vs_day1": {"min": int(offset_days.min()), "median": float(offset_days.median()), "max": int(offset_days.max())},
            "old_first_row_anchor_rows_kept": int(old_keep.sum()),
            "rows_kept_by_both_anchors": int((old_keep & s["study_day"].between(1, 28)).sum()),
            "devices_with_clock_glitch_before_2021": int((pd.to_datetime(s.groupby("deviceId")["ts_start"].min(), unit="ms") < pd.Timestamp("2021-01-01")).sum()),
            "segment_rows": {"A_median": float(rows_seg["A"].median()), "B_median": float(rows_seg["B"].median()),
                             "segments_with_0_rows": int((rows_seg == 0).sum().sum() + (2 * 49 - rows_seg.size)),
                             "segments_with_lt_288_rows_(1_day)": int((rows_seg < 288).sum().sum())},
        },
        "hrv_power_artifacts_raw_rows_gt_1e6": {c: int((pd.read_csv(C.SENSOR_CSV, usecols=[c])[c] > 1e6).sum()) for c in ("lf", "hf")},
        "hrv_power_artifact_rule": "lf / hf > 1e6 ms^2 set to NaN, then log1p; applied as a fixed rule to every row",
        "missingness_sensor_columns_sorted":{k: float(v) for k, v in miss.items()},
        "data_dictionary": data_dict,
        "granularity_gate": {
            "dtype_scan": "no object/list columns hold sequences; sensor_hrv.csv is all scalar float columns (one row per ~5-min slice)",
            "description_scan_aggregation_phrases": agg_words,
            "file_search_raw_tables": raw_files, "raw_table_bytes": raw_sizes,
            "sampling_interval_sec": {"sensor_hrv (ts_end - ts_start) median": agg_dt, "raw hrm.csv.gz median": hrm_dt, "raw lit.csv.gz median": lit_dt},
            "verdict": "raw sequences (1 Hz raw HR / light / motion tables) plus a 5-minute aggregated table; Stage 6 = branch R",
        },
    }

    # ---------------- Stage 0 profile
    pos_participants = int(seg.groupby("deviceId")["y"].max().sum())
    pos10 = int(seg.groupby("deviceId")["y10"].max().sum())
    # FL feasibility estimate: average outer-train fold (80%) minus 20% validation, split K ways
    pos_seg = int(seg["y"].sum())
    est = {K: {"pos_segments_per_client": round(pos_seg * 0.8 * 0.8 / K, 1), "pos_participants_per_client": round(pos_participants * 0.8 * 0.8 / K, 1)}
           for K in (2, 3)}
    profile = {
        "inputs": {
            "dataset": "HRV + PHQ-9 (Ref 2): Baigutanova et al. 2025, Scientific Data (PMC12375003), Samsung Galaxy Watch Active 2",
            "data_path": "research_datasets/old/hrv_sleep_diaries/ (survey.csv, sensor_hrv.csv, sleep_diary.csv, raw_data/*.csv.gz)",
            "terms": "public Figshare research dataset (KAIST IRB KH2020-027); used here under REB25-1587_MOD1 as research data",
            "unit": "participant-segment (2 per participant: days 1-14 and days 15-28 of the shared study calendar)",
            "id_rule": "deviceId (identical in survey, sensor and diary tables; diary column userId)",
            "instrument": "PHQ-9 total score 0-27 at day 0 (PHQ9_1), mid-study (PHQ9_2) and end (PHQ9_F)",
            "target": "PHQ-9 >= 5 (at least mild symptoms) on the score that covers the segment: PHQ9_2 for days 1-14, PHQ9_F for days 15-28",
            "target_type": "binary (cut-off)",
            "leak_cols": C.LEAK_COLS,
            "baseline_covs": C.BASELINE_COLS + C.DEMO_COLS,
            "sources": {"sensor (5-min aggregates)": 51, "diurnal derived": 11, "sleep diary": 12, "baseline questionnaires + demographics": 16},
            "clinical_target_sensitivity": C.CLINICAL_TARGET_SENS,
            "published_paper_same_target": "none: the data paper reports descriptive statistics and correlations only, no depression prediction",
        },
        "P1_individual_rows": True,
        "P2_rows_per_participant": "many (2 labelled segments; ~1,600 five-minute sensor rows)",
        "P3_design": "longitudinal (3 PHQ-9 time points over 4 weeks), but only 2 labelled segments per person",
        "P4_size": {"N_participants": 49, "n_rows_segments": int(len(seg)), "positives_segments_phq9_ge5": pos_seg,
                    "positive_participants_ge5": pos_participants, "positives_segments_phq9_ge10": int(seg["y10"].sum()),
                    "positive_participants_ge10": pos10, "small_n": True},
        "P5_target_type": "binary cut-off (continuous score available)",
        "P6_signal_granularity": "raw high-frequency (1 Hz) watch tables + 5-min aggregates",
        "P7_natural_sites": None,
        "P8_published_reference": False,
        "P9_survey_weights": False,
        "target_choice_reason": ("PHQ-9 >= 10 gives only 7 positive segments from 5 participants (3 at mid-study, 4 at the end), "
                                 "too few for a stable AUROC or any client split. PHQ-9 >= 5 (mild or worse) gives "
                                 f"{pos_seg} positive segments from {pos_participants} participants. >= 10 is kept as a sensitivity analysis."),
        "fl_feasibility_estimate": est,
        "applicability": [
            {"stage": 0, "status": "run", "branch": None},
            {"stage": 1, "status": "run", "branch": None},
            {"stage": 2, "status": "run", "branch": "participant-block permutation; row vs grouped on 5-min rows and on segments"},
            {"stage": 3, "status": "branch", "branch": "SMALL-N (repeated nested grouped CV 5x5, inner 3-fold); Part B skipped (no published model)"},
            {"stage": 4, "status": "branch", "branch": "SMALL-N; temporal check = train on segment A, test on segment B"},
            {"stage": 5, "status": "branch", "branch": "SMALL-N: thresholds chosen on inner OOF of each outer training fold; clinical target sensitivity 0.80"},
            {"stage": 6, "status": "branch", "branch": "R (raw 1 Hz HR reduced to 1-min; 5-min multichannel); ppg not used"},
            {"stage": 7, "status": "branch", "branch": "SMALL-N inside outer folds, K=2 simulated clients (borderline, see counts); K=3 not feasible"},
            {"stage": 8, "status": "run", "branch": None},
            {"stage": 9, "status": "branch", "branch": "SMALL-N: seeds + fold spread + leave-one-participant-out"},
            {"stage": 10, "status": "branch", "branch": "fallback: no paper with the same target; clinical target sensitivity 0.80"},
            {"stage": 11, "status": "run", "branch": "SVG figures (matplotlib not installed in any project venv)"},
            {"stage": 12, "status": "run", "branch": None},
        ],
    }
    C.write_json("hrv_phq9_protocol_inventory.json", inventory)
    C.write_json("hrv_phq9_protocol_profile.json", profile)
    print({"n_seg": len(seg), "pos": pos_seg, "pos_part": pos_participants, "pos10": int(seg['y10'].sum()), "pos10_part": pos10})
    print(inventory["calendar_anchor"])
    print(est)


if __name__ == "__main__":
    main()
