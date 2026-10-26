#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs (device IDs are pseudonymous codes)
"""Stage 6 prep (Branch R): stream the raw ~1 Hz heart-rate table (raw_data/hrm.csv.gz)
in chunks and reduce it to per-device, per-minute mean HR (HR > 0 only; 0 = no reading /
watch off-wrist) plus a sample count. Peak memory stays small (chunked read, only the
running per-minute sums are kept). Output cache: scripts/.cache/hrv_phq9_protocol_hrm_minute.pkl

ppg/acc/gyr/grv raw tables are NOT streamed here (4.7 GB / 4.9 GB / 3.3 GB / 6.2 GB gz):
their 5-minute averages already exist in sensor_hrv.csv, and streaming them would exceed
the time budget of this run. This is recorded in the protocol report.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
HRM = REPO / "research_datasets/old/hrv_sleep_diaries/raw_data/hrm.csv.gz"
OUT = REPO / "scripts/.cache/hrv_phq9_protocol_hrm_minute.pkl"
META = REPO / "scripts/hrv_phq9_protocol_stage06a_hrm_minute_bins_results.json"
# study calendar (KST) 2021-03-09 .. 2021-04-05 -> UTC bounds with a 1-day margin
T0 = int(pd.Timestamp("2021-03-07 15:00", tz="UTC").value // 10**6)
T1 = int(pd.Timestamp("2021-04-06 15:00", tz="UTC").value // 10**6)
CHUNK = 2_000_000


def main() -> None:
    t = time.time()
    parts = []
    n_rows = n_kept = 0
    for ch in pd.read_csv(HRM, chunksize=CHUNK, dtype={"deviceId": "category", "ts": "int64", "HR": "float32"}):
        n_rows += len(ch)
        ch = ch[(ch["HR"] > 0) & (ch["ts"] >= T0) & (ch["ts"] < T1)]
        n_kept += len(ch)
        if ch.empty:
            continue
        minute = (ch["ts"].to_numpy() // 60000).astype(np.int64)
        g = pd.DataFrame({"deviceId": ch["deviceId"].astype(str).to_numpy(), "minute": minute,
                          "hr": ch["HR"].to_numpy(dtype=np.float64)})
        parts.append(g.groupby(["deviceId", "minute"])["hr"].agg(["sum", "count"]))
        if len(parts) >= 20:
            parts = [pd.concat(parts).groupby(level=[0, 1]).sum()]
    agg = pd.concat(parts).groupby(level=[0, 1]).sum()
    agg["hr_mean"] = agg["sum"] / agg["count"]
    agg = agg[["hr_mean", "count"]].reset_index()
    agg.to_pickle(OUT)
    meta = {"source": str(HRM.relative_to(REPO)), "rows_read": int(n_rows), "rows_kept_hr_gt0_in_window": int(n_kept),
            "minutes": int(len(agg)), "devices": int(agg["deviceId"].nunique()), "seconds": round(time.time() - t, 1),
            "cache": str(OUT.relative_to(REPO))}
    META.write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
