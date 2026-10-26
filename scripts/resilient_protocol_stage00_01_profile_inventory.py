#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): a different institution's REC/IRAS-approved
# protocol (IRAS 321104), independent of this project's REB25-1587_MOD1. Research data.
"""Protocol Stage 0 (profile) + Stage 1 (inventory, raw-data gate) + participant-level
feature table, for RESILIENT. Run with .venv311 (reuses the existing loaders in
train_resilient_multimodal_binary_phq9.py UNMODIFIED: get_participant_uids,
load_demographics, load_raw_frames, build_hourly_channel, WINDOW_HOURS).

Feature window (reused design decision RB1 of the existing binary script, re-verified
here): [phq_date, phq_date + 512 h). Reason: there is (re-measured below) almost no
recording before phq_date, and a bounded near-term window avoids months of drift.

Writes:
  scripts/resilient_protocol_profile.json
  scripts/resilient_protocol_inventory.json
  scripts/resilient_protocol_features.pkl   (participant-level features + labels +
                                             hourly HR/Steps arrays for the raw control)
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_resilient_multimodal_binary_phq9 as B  # noqa: E402  (reused, unmodified)

OUT_PROFILE = SCRIPT_DIR / "resilient_protocol_profile.json"
OUT_INV = SCRIPT_DIR / "resilient_protocol_inventory.json"
OUT_PKL = SCRIPT_DIR / "resilient_protocol_features.pkl"
PHQ_CUTOFF = 10   # Kroenke, Spitzer & Williams 2001: PHQ-9 >= 10 = moderate+; same cut-off as the existing binary script
GDS_CUTOFF = 5    # GDS-15 >= 5 suggests depression (Yesavage/Sheikh short form convention)
H = B.WINDOW_HOURS

SENSOR_GROUPS = {
    "watch_hr": ["hr_mean", "hr_sd", "hr_p10", "hr_p90", "hr_night_mean", "hr_day_mean", "hr_day_minus_night", "hr_coverage"],
    "watch_steps": ["steps_daily_median", "steps_daily_mean", "steps_daily_sd", "steps_low_days_frac", "steps_coverage"],
    "mat_physio": ["mat_hr_mean", "mat_hr_nightly_sd", "mat_rr_mean", "mat_snoring_mean", "mat_sdnn_mean"],
    "mat_state": ["sleep_rem_frac", "sleep_deep_frac", "sleep_light_frac", "sleep_wake_frac", "sleep_hours_mean", "sleep_n_nights"],
}
DEMO_COLS = ["sex_female", "age_group", "hypertension", "osteoarthritis"]
QUEST_COLS = ["gad_total", "ace_total"]
DESCRIPTIONS = {
    "hr_mean": "mean watch HR in window (bpm)", "hr_sd": "SD of watch HR samples", "hr_p10": "10th percentile watch HR",
    "hr_p90": "90th percentile watch HR", "hr_night_mean": "mean watch HR 00:00-05:59 UTC", "hr_day_mean": "mean watch HR 09:00-20:59 UTC",
    "hr_day_minus_night": "day mean minus night mean HR", "hr_coverage": "fraction of 512 hourly bins with >=1 HR sample",
    "steps_daily_median": "median of daily step sums", "steps_daily_mean": "mean of daily step sums", "steps_daily_sd": "SD of daily step sums",
    "steps_low_days_frac": "fraction of recorded days with < 1000 steps", "steps_coverage": "fraction of 512 hourly bins with steps data",
    "mat_hr_mean": "mean of nightly mean sleep-mat HR", "mat_hr_nightly_sd": "SD across nights of nightly mat HR",
    "mat_rr_mean": "mean of nightly mean respiration rate", "mat_snoring_mean": "mean nightly snoring seconds (sum per night)",
    "mat_sdnn_mean": "mean of nightly mean SDNN_1 (ms)", "sleep_rem_frac": "share of staged time in REM",
    "sleep_deep_frac": "share of staged time in deep", "sleep_light_frac": "share of staged time in light", "sleep_wake_frac": "share of staged time awake",
    "sleep_hours_mean": "mean nightly non-wake staged hours", "sleep_n_nights": "nights with sleep-state data in window",
    "sex_female": "1 = female", "age_group": "ordinal 0/1/2 for [72,75]/[76,87]/[88,99]", "hypertension": "essential hypertension flag",
    "osteoarthritis": "osteoarthritis flag", "gad_total": "GAD-7 total (same visit as PHQ-9)", "ace_total": "ACE-III total (baseline)",
}


def window(df, col, t0):
    if df is None or len(df) == 0:
        return None
    rel = (df[col] - t0).dt.total_seconds() / 3600.0
    sub = df.loc[(rel >= 0) & (rel < H)]
    return sub if len(sub) else None


def participant_features(fr, t0):
    f = {}
    hr = window(fr["hr"], "ts", t0)
    hourly_hr, cov = B.build_hourly_channel(fr["hr"], "ts", "Heart Rate", t0, "mean")
    f["hr_coverage"] = cov
    if hr is not None:
        v = hr["Heart Rate"].astype(float)
        hrs = hr["ts"].dt.hour
        f.update(hr_mean=v.mean(), hr_sd=v.std(), hr_p10=v.quantile(0.1), hr_p90=v.quantile(0.9))
        night, day = v[hrs < 6], v[(hrs >= 9) & (hrs < 21)]
        f["hr_night_mean"] = night.mean() if len(night) else np.nan
        f["hr_day_mean"] = day.mean() if len(day) else np.nan
        f["hr_day_minus_night"] = f["hr_day_mean"] - f["hr_night_mean"]
    hourly_steps, scov = B.build_hourly_channel(fr["steps"], "ts", "Steps", t0, "sum")
    f["steps_coverage"] = scov
    st = window(fr["steps"], "ts", t0)
    if st is not None:
        daily = st.groupby(st["ts"].dt.date)["Steps"].sum()
        f.update(steps_daily_median=daily.median(), steps_daily_mean=daily.mean(),
                 steps_daily_sd=daily.std() if len(daily) > 1 else np.nan, steps_low_days_frac=float((daily < 1000).mean()))
    ph = window(fr["sleep_physio"], "ts", t0)
    if ph is not None:
        g = ph.groupby(ph["ts"].dt.date)
        nightly = pd.DataFrame({"hr": g["Heart Rate"].mean(), "rr": g["Respiration Rate"].mean(),
                                "sn": g["Snoring"].sum(), "sdnn": g["SDNN_1"].mean()})
        f.update(mat_hr_mean=nightly.hr.mean(), mat_hr_nightly_sd=nightly.hr.std() if len(nightly) > 1 else np.nan,
                 mat_rr_mean=nightly.rr.mean(), mat_snoring_mean=nightly.sn.mean(), mat_sdnn_mean=nightly.sdnn.mean())
    ss = window(fr["sleep_state"], "start", t0)
    if ss is not None:
        ss = ss.copy()
        ss["dur"] = (ss["end"] - ss["start"]).dt.total_seconds().clip(lower=0)
        tot = ss["dur"].sum()
        if tot > 0:
            for s, k in [("REM", "rem"), ("deep", "deep"), ("light", "light"), ("wakeup", "wake")]:
                f[f"sleep_{k}_frac"] = ss.loc[ss["Sleep state"] == s, "dur"].sum() / tot
            ss["date"] = ss["start"].dt.date
            asleep = ss[ss["Sleep state"] != "wakeup"].groupby("date")["dur"].sum() / 3600.0
            f["sleep_hours_mean"] = asleep.mean() if len(asleep) else np.nan
            f["sleep_n_nights"] = float(ss["date"].nunique())
    return f, hourly_hr, cov, hourly_steps, scov


def main():
    raw_demo = pd.read_csv(B.DEMOGRAPHICS_CSV)
    demo = B.load_demographics()
    uids = B.get_participant_uids()
    assert len(uids) == len(raw_demo) == raw_demo.user_id.nunique()
    assert set(uids) == set(raw_demo.user_id)

    rows, hourly_hr, hourly_steps = [], {}, {}
    inv_files = {k: [] for k in ["hr", "steps", "sleep_physio", "sleep_state"]}
    intervals = {k: [] for k in inv_files}
    pre_baseline = {k: 0 for k in inv_files}
    any_pre_baseline = 0
    dates = []
    obj_cols = set()
    for u in uids:
        fr = B.load_raw_frames(u)
        t0 = demo.loc[u, "phq_date"]
        has_pre = False
        for k in inv_files:
            df = fr[k]
            tcol = "start" if k == "sleep_state" else "ts"
            n = 0 if df is None else len(df)
            inv_files[k].append(n)
            if df is not None and n:
                ts = df[tcol].sort_values()
                dates += [ts.iloc[0], ts.iloc[-1]]
                d = ts.diff().dt.total_seconds().dropna()
                if len(d):
                    intervals[k].append(float(d.median()))
                n_pre = int((ts < t0).sum())
                if n_pre > 0:
                    pre_baseline[k] += 1
                    has_pre = True
                for c in df.columns:
                    if df[c].dtype == object and c not in ("HR Timestamp", "Steps Timestamp", "Timestamp", "Start time", "End time", "Sleep state"):
                        obj_cols.add(f"{k}:{c}")
        any_pre_baseline += int(has_pre)
        f, hh, hc, sh, sc = participant_features(fr, t0)
        hourly_hr[u] = hh if hc > 0 else None
        hourly_steps[u] = sh if sc > 0 else None
        r = raw_demo.set_index("user_id").loc[u]
        f.update(uid=u, sex_female=1.0 if r["Sex"] == "Female" else (0.0 if r["Sex"] == "Male" else np.nan),
                 age_group=B.AGE_GROUP_MAP.get(r["Age group"], np.nan),
                 hypertension=float(r["Essential hypertension"]) if pd.notna(r["Essential hypertension"]) else np.nan,
                 osteoarthritis=float(r["Osteoarthritis"]) if pd.notna(r["Osteoarthritis"]) else np.nan,
                 gad_total=r["gad_total"], ace_total=r["ace_total"], phq_total=r["phq_total"], gds_total=r["gds_total"],
                 phq_date_ord=float(t0.toordinal()) if pd.notna(t0) else np.nan)
        rows.append(f)

    feat = pd.DataFrame(rows).set_index("uid")
    sensor_cols = [c for g in SENSOR_GROUPS.values() for c in g]
    for c in sensor_cols + DEMO_COLS + QUEST_COLS:
        if c not in feat:
            feat[c] = np.nan
    y_phq = (feat["phq_total"] >= PHQ_CUTOFF).astype(int).to_numpy()
    y_gds = (feat["gds_total"] >= GDS_CUTOFF).astype(int).to_numpy()
    n_pos = int(y_phq.sum())

    # ---------------- Stage 1 inventory ----------------
    def mmm(a):
        a = np.asarray(a, float)
        return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}

    miss_demo = raw_demo.isna().mean().sort_values(ascending=False)
    miss_feat = feat[sensor_cols + DEMO_COLS + QUEST_COLS].isna().mean().sort_values(ascending=False)
    group_of = {c: g for g, cs in SENSOR_GROUPS.items() for c in cs}
    group_of.update({c: "demographics" for c in DEMO_COLS})
    group_of.update({c: "questionnaire (same visit)" for c in QUEST_COLS})
    dictionary = [{"column": c, "type": "float", "group": group_of[c], "description": DESCRIPTIONS[c],
                   "pct_missing": round(100 * float(feat[c].isna().mean()), 1)} for c in sensor_cols + DEMO_COLS + QUEST_COLS]
    dict_md = pd.read_csv(B.RAW_DIR.parent.parent / "Resilient_metadata" / "Resilient_metadata" / "Sleepmat_Watch_dictionary.csv") \
        if (B.RAW_DIR.parent.parent / "Resilient_metadata" / "Resilient_metadata" / "Sleepmat_Watch_dictionary.csv").exists() else None
    agg_words = ["mean", "median", "sum", "count", "daily", "weekly", "average"]
    desc_hits = []
    if dict_md is not None:
        txt = dict_md.astype(str).apply(lambda r: " ".join(r).lower(), axis=1)
        desc_hits = [w for w in agg_words if txt.str.contains(w).any()]
    raw_file_count = sum(1 for _ in B.RAW_DIR.rglob("*.csv"))
    median_interval = {k: (float(np.median(v)) if v else None) for k, v in intervals.items()}
    verdict = "raw sequences" if median_interval["hr"] is not None and median_interval["hr"] < 3600 else "aggregates only"

    inventory = {
        "demographics_shape": list(raw_demo.shape),
        "n_participants": len(uids),
        "raw_rows_per_participant": {k: mmm(v) for k, v in inv_files.items()},
        "participants_with_zero_rows": {k: int(sum(1 for n in v if n == 0)) for k, v in inv_files.items()},
        "date_range": [str(min(dates).date()), str(max(dates).date())],
        "demographics_top_missing": {k: round(float(v), 3) for k, v in miss_demo.head(12).items()},
        "feature_missing_sorted": {k: round(float(v), 3) for k, v in miss_feat.items()},
        "data_dictionary": dictionary,
        "granularity_gate": {
            "check1_dtype_scan": f"object columns other than timestamps/state labels: {sorted(obj_cols) or 'none'}; raw files are long-format per-sample tables (no array cells)",
            "check2_description_scan": f"aggregation words in Sleepmat_Watch_dictionary.csv: {desc_hits or 'none'}" if dict_md is not None else "dictionary not found",
            "check3_file_search": f"{raw_file_count} per-participant raw CSV files under Sleepmat_Watch_Data/<uid>/",
            "check4_sampling_interval_median_sec": median_interval,
            "verdict": verdict,
        },
        "pre_baseline_data": {"participants_with_any_row_before_phq_date_by_modality": pre_baseline,
                              "participants_with_any_pre_baseline_row": any_pre_baseline},
        "feature_window": "[phq_date, phq_date + 512 h) (21.3 days after the PHQ-9 visit)",
        "label_counts": {"phq9_ge10": n_pos, "phq9_ge5": int((feat.phq_total >= 5).sum()), "gds15_ge5": int(y_gds.sum()),
                         "phq_total_hist": {int(k): int(v) for k, v in feat.phq_total.value_counts().sort_index().items()}},
    }

    # ---------------- Stage 0 profile ----------------
    profile = {
        "P1_individual_rows": {"answer": True, "evidence": f"Demographics.csv has {len(raw_demo)} rows, one per user_id; per-participant raw folders: {len(uids)}"},
        "P2_rows_per_participant": {"answer": 1, "evidence": "one PHQ-9 per participant (no _6months PHQ columns); unit of analysis = participant"},
        "P3_design": {"answer": "cross-sectional label (baseline PHQ-9) with longitudinal sensors", "evidence": "only ACE-III has a 6-month follow-up; PHQ-9/GDS-15/GAD-7 baseline only, so no change target and no temporal validity check"},
        "P4_size": {"N": len(uids), "P": n_pos, "small_n": True, "evidence": f"N={len(uids)} < 200 and P={n_pos} < 50"},
        "P5_target_type": {"answer": "binary (PHQ-9 >= 10)", "evidence": "cut-off from Kroenke et al. 2001; same as train_resilient_multimodal_binary_phq9.py PHQ9_POSITIVE_THRESHOLD"},
        "P6_signal_granularity": {"answer": verdict, "evidence": f"median sampling interval (s): {median_interval}"},
        "P7_natural_sites": {"answer": None, "evidence": "single-site study (Surrey); no site/clinic/region column; one device model"},
        "P8_published_reference": {"answer": False, "evidence": "RESILIENT paper (Scientific Data 2025, PMC12546717) is a data descriptor: descriptive correlations only, no prediction metrics"},
        "P9_survey_weights": {"answer": False, "evidence": "no weight column in Demographics.csv"},
        "branches": {
            "stages_3_to_9": "SMALL-N: repeated nested stratified CV (outer 5x5, inner 3), N=73 > 60 so not LOPO by default (LOPO run as a check in Stage 9)",
            "stage_6": "branch R (raw sub-hourly HR, hourly steps, per-minute sleep mat): hourly bins on a 512 h window, MOMENT per modality",
            "stage_7": "FL feasibility check: K=2 needs >= ~10 positives per client, total positives is 10",
        },
    }
    for p in (OUT_PROFILE, OUT_INV):
        pass
    OUT_PROFILE.write_text(json.dumps(profile, indent=2, default=str))
    OUT_INV.write_text(json.dumps(inventory, indent=2, default=str))
    OUT_PKL.write_bytes(pickle.dumps({
        "features": feat, "sensor_groups": SENSOR_GROUPS, "demo_cols": DEMO_COLS, "quest_cols": QUEST_COLS,
        "y_phq10": y_phq, "y_gds5": y_gds, "y_phq5": (feat["phq_total"] >= 5).astype(int).to_numpy(),
        "hourly_hr": hourly_hr, "hourly_steps": hourly_steps, "phq_cutoff": PHQ_CUTOFF, "gds_cutoff": GDS_CUTOFF,
    }))
    print(json.dumps({"n": len(uids), "pos": n_pos, "verdict": verdict, "pre": pre_baseline, "any_pre": any_pre_baseline,
                      "intervals": median_interval, "missing_top": dict(list(inventory["feature_missing_sorted"].items())[:6])}, indent=1, default=str))


if __name__ == "__main__":
    main()
