#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stage 6 branch R: frozen MOMENT-1-small embeddings per participant-segment.
Run with .venv311 (momentfm + torch). Reuses mood_monitoring_agent.model.MomentRiskPipeline
(frozen backbone, task "embedding"), never trains it.

Two signal views, both cut into non-overlapping 512-sample windows inside each segment
on the shared KST study calendar:
  HRmin : raw 1 Hz watch HR (hrm.csv.gz) reduced to 1-minute means (cache from stage06a);
          window = 512 min (~8.5 h); keep windows with >= 50% observed minutes.
  M5    : 8 channels from the 5-minute sensor_hrv.csv slices (HR, rmssd, sdnn, lf/hf,
          |acc|, |gyr|, steps, light); window = 512 slots (~42.7 h); keep >= 10% observed.
Missing samples are set to 0 with input_mask = 0. No Butterworth filter: the signals are
already at 1/60 Hz or 1/300 Hz, below every cut-off in the project's filter table. MOMENT's
own RevIN instance normalisation rescales each window, so no global z-score is applied
(a train-fit z-score would be cancelled by RevIN anyway).
Output: scripts/.cache/hrv_phq9_protocol_moment_embeddings.npz + window arrays for the TCN.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import hrv_phq9_protocol_common as C  # noqa: E402
from mood_monitoring_agent.model import MomentRiskPipeline  # noqa: E402

W = 512
OUT = C.CACHE / "hrv_phq9_protocol_moment_embeddings.npz"
WIN_OUT = C.CACHE / "hrv_phq9_protocol_windows.npz"
DAY1_UTC_MS = int((C.STUDY_DAY1 - pd.Timedelta(hours=9)).value // 10**6)


def seg_bounds(segment: str, step_ms: int):
    start = DAY1_UTC_MS + (0 if segment == "A" else 14 * 86_400_000)
    n = 14 * 86_400_000 // step_ms
    return start, n


def windows_from_grid(vals: np.ndarray, obs: np.ndarray, min_frac: float):
    out_x, out_m = [], []
    n = vals.shape[-1]
    for s in range(0, n - W + 1, W):
        m = obs[s:s + W]
        if m.mean() >= min_frac:
            out_x.append(np.nan_to_num(vals[:, s:s + W], nan=0.0) * m)
            out_m.append(m.astype(np.float32))
    return out_x, out_m


def build_hrmin(seg):
    hm = pd.read_pickle(C.HRM_MINUTE_CACHE)
    hm["ts_ms"] = hm["minute"] * 60000
    X, M, owner = [], [], []
    for i, r in seg.iterrows():
        start, n = seg_bounds(r["segment"], 60000)
        d = hm[(hm["deviceId"] == r["deviceId"]) & (hm["ts_ms"] >= start) & (hm["ts_ms"] < start + n * 60000)]
        grid = np.full(n, np.nan)
        grid[((d["ts_ms"] - start) // 60000).astype(int).to_numpy()] = d["hr_mean"].to_numpy()
        obs = ~np.isnan(grid)
        xs, ms = windows_from_grid(grid[None, :], obs.astype(float), 0.5)
        X += xs
        M += ms
        owner += [i] * len(xs)
    return np.array(X, dtype=np.float32), np.array(M, dtype=np.float32), np.array(owner)


def build_m5(seg):
    s = C.calendar_rows(C.load_sensor_rows())
    s["acc_mag"] = np.sqrt(s["acc_x_avg"] ** 2 + s["acc_y_avg"] ** 2 + s["acc_z_avg"] ** 2)
    s["gyr_mag"] = np.sqrt(s["gyr_x_avg"] ** 2 + s["gyr_y_avg"] ** 2 + s["gyr_z_avg"] ** 2)
    chans = ["HR", "rmssd", "sdnn", "lf/hf", "acc_mag", "gyr_mag", "steps", "light_avg"]
    X, M, owner = [], [], []
    for i, r in seg.iterrows():
        start, n = seg_bounds(r["segment"], 300_000)
        d = s[(s["deviceId"] == r["deviceId"]) & (s["segment"] == r["segment"])]
        slot = ((d["ts_start"] - start) // 300_000).astype(int).to_numpy()
        ok = (slot >= 0) & (slot < n)
        grid = np.full((len(chans), n), np.nan)
        grid[:, slot[ok]] = d[chans].to_numpy(dtype=float)[ok].T
        obs = np.zeros(n)
        obs[slot[ok]] = 1.0
        xs, ms = windows_from_grid(grid, obs, 0.10)
        X += xs
        M += ms
        owner += [i] * len(xs)
    return np.array(X, dtype=np.float32), np.array(M, dtype=np.float32), np.array(owner), chans


def embed(pipe, X, M, bs=32):
    outs = []
    for s in range(0, len(X), bs):
        e = pipe.embed(x_enc=torch.from_numpy(X[s:s + bs]), input_mask=torch.from_numpy(M[s:s + bs]))
        outs.append(e.detach().cpu().numpy())
    return np.concatenate(outs) if outs else np.zeros((0, 512))


def pool(E, owner, n_seg):
    mean = np.full((n_seg, E.shape[1]), np.nan)
    sd = np.full((n_seg, E.shape[1]), np.nan)
    cnt = np.zeros(n_seg, dtype=int)
    for i in range(n_seg):
        e = E[owner == i]
        cnt[i] = len(e)
        if len(e):
            mean[i] = e.mean(0)
            sd[i] = e.std(0) if len(e) > 1 else 0.0
    return mean, sd, cnt


def main():
    t = time.time()
    torch.manual_seed(0)
    seg = C.build_segments()
    Xh, Mh, oh = build_hrmin(seg)
    Xm, Mm, om, chans = build_m5(seg)
    print("windows", Xh.shape, Xm.shape, flush=True)
    pipe = MomentRiskPipeline.from_pretrained()
    pipe.eval()
    n_params = sum(p.numel() for p in pipe.backbone.parameters())
    Eh = embed(pipe, Xh, Mh)
    Em = embed(pipe, Xm, Mm)
    hm, hs, hc = pool(Eh, oh, len(seg))
    mm, ms, mc = pool(Em, om, len(seg))
    np.savez(OUT, hr_mean=hm, hr_sd=hs, hr_count=hc, m5_mean=mm, m5_sd=ms, m5_count=mc,
             device=seg["deviceId"].to_numpy(), segment=seg["segment"].to_numpy())
    np.savez_compressed(WIN_OUT, hr_X=Xh, hr_M=Mh, hr_owner=oh, m5_X=Xm, m5_M=Mm, m5_owner=om)
    meta = {"model": "AutonLab/MOMENT-1-small (frozen, task=embedding)", "backbone_params": int(n_params),
            "hr_minute": {"windows": int(len(Xh)), "segments_with_windows": int((hc > 0).sum()),
                          "windows_per_segment_median": float(np.median(hc)), "window": "512 x 1 min", "min_observed": 0.5},
            "m5_multichannel": {"windows": int(len(Xm)), "segments_with_windows": int((mc > 0).sum()), "channels": chans,
                                "windows_per_segment_median": float(np.median(mc)), "window": "512 x 5 min", "min_observed": 0.1},
            "seconds": round(time.time() - t, 1), "embedding_dim": 512}
    (C.SCRIPTS / "hrv_phq9_protocol_stage06b_moment_embed_results.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
