#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. Local research data only.
"""Torch-only FedAvg worker for JMIR revision Phase 2 (no xgboost import).
Logic identical to scripts/psyche_d_fedavg_vs_fedtree_step23_fedavg.py, plus
per-round logging of validation AUROC and upload/download bytes.
Usage: p2_fl_mlp.py in.pkl out.pkl  (in: {"partitions": [[set,...],...], "local_only": bool})."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
# The common module imports the baseline script, which imports lightgbm only for
# the DART factory (unused here). In this venv, lightgbm + torch in one process
# segfaults (numpy was upgraded to 2.5.4 by a matplotlib install on 2026-10-16),
# so a placeholder module is registered; FedAvg never touches lightgbm.
import types  # noqa: E402
sys.modules.setdefault("lightgbm", types.ModuleType("lightgbm"))
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
import psyche_d_fedavg_vs_fedtree_mlp_model as M  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step23_fedavg import LOCAL_EPOCHS, LR, N_ROUNDS, load_split_and_frame  # noqa: E402


def run_fedavg(train_df, val_df, test_df, y, fc, client_dfs, local_only):
    y_val, y_test = y.loc[val_df.index].to_numpy(), y.loc[test_df.index].to_numpy()
    stats = C.federated_sufficient_stats(client_dfs, fc)
    mean, std = stats["mean"], stats["std"]
    X_val, X_test = C.apply_impute_scale(val_df, fc, mean, std), C.apply_impute_scale(test_df, fc, mean, std)
    cX = [C.apply_impute_scale(d, fc, mean, std) for d in client_dfs]
    cy = [y.loc[d.index].to_numpy() for d in client_dfs]
    n_rows = [len(d) for d in client_dfs]
    K = len(client_dfs)
    local = []
    if local_only:
        for cid in range(K):
            lm = M.new_model(seed=M.SEED)
            M.train_local(lm, cX[cid], cy[cid], epochs=N_ROUNDS * LOCAL_EPOCHS, lr=LR, seed=1000 + cid)
            local.append(M.predict_proba(lm, X_test))
    gm = M.new_model(seed=M.SEED)
    model_bytes = M.count_params(gm) * 4  # float32 state dict, as measured in s14
    best = (-1.0, None, -1, None)
    hist = []
    for rnd in range(N_ROUNDS):
        gs = M.get_state(gm)
        states = []
        for cid in range(K):
            lm = M.SmallMLP()
            M.set_state(lm, gs)
            M.train_local(lm, cX[cid], cy[cid], epochs=LOCAL_EPOCHS, lr=LR, seed=rnd * 100 + cid)
            states.append(M.get_state(lm))
        M.set_state(gm, M.fedavg_aggregate(states, n_rows))
        vp = M.predict_proba(gm, X_val)
        va = float(roc_auc_score(y_val, vp))
        hist.append({"round": rnd + 1, "val_auroc": va, "upload_bytes": model_bytes * K, "download_bytes": model_bytes * K})
        if va > best[0]:
            best = (va, M.get_state(gm), rnd + 1, vp.copy())
    M.set_state(gm, best[1])
    return {"test_pred": M.predict_proba(gm, X_test), "val_pred": best[3], "best_round": best[2],
            "best_val_auroc": best[0], "rounds": hist, "local_only_test_preds": local}


if __name__ == "__main__":
    job = pickle.loads(Path(sys.argv[1]).read_bytes())
    cache, labeled, fc, y, tr, va, te = load_split_and_frame()
    train_df, val_df, test_df = labeled.loc[tr], labeled.loc[va], labeled.loc[te]
    if job.get("cols"):  # optional input subset (final run, task 2: P+S+L, 41 inputs)
        fc = list(job["cols"])
        M.SmallMLP.__init__.__defaults__ = (len(fc),)
    out = []
    for sites in job["partitions"]:
        cdfs = [train_df[train_df["user_id"].isin(s)] for s in sites]
        out.append(run_fedavg(train_df, val_df, test_df, y, fc, cdfs, job["local_only"]))
        print(f"FedAvg partition {len(out)}/{len(job['partitions'])} done", flush=True)
    Path(sys.argv[2]).write_bytes(pickle.dumps(out))
