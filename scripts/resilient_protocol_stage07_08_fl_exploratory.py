#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): IRAS 321104, independent of REB25-1587_MOD1. Research data.
"""Protocol Stage 7 (federated learning) + Stage 8 (FL audit), RESILIENT.

FEASIBILITY (protocol rule): FL needs every client to hold >= ~10 positives. RESILIENT has
10 positives in total and 8 per outer training fold, so even K=2 leaves ~4 per client:
FL is marked "not feasible at this N". Everything below is EXPLORATORY ONLY, run so the
mechanics and communication cost are measured, not to claim an FL result.

Exploratory design (SMALL-N: FL inside the outer CV folds, no fixed 70/15/15 split):
- Simulated K=2 clients per outer training fold, label-stratified seeded split (so each
  client gets 4 positives; a plain random permutation could leave a client with 0-1).
- Federated preprocessing: per-client count / sum / sum of squares -> global mean/std,
  train clients only; mean-impute + z-score for the MLPs. Trees use native NaN.
- No validation split exists at this N, so NO round or eta is selected on data: all
  methods run a fixed, pre-declared number of rounds and the final round is evaluated.
  Per-round test-fold AUROC curves are recorded as diagnostics only.
- Methods: FedAvg MLP (PSYCHE-D SmallMLP, 40 rounds x 2 local epochs, Adam 1e-3, batch
  64, pos_weight, n_k/N weights); FedXgbBagging (Flower aggregate, 1 tree/client/round,
  30 rounds); TreeAvg (eta 0.1 frozen anchor, depth 4, 30 rounds; eta grid and a
  60-round check reported as sensitivity only); Centralized XGBoost (150 rounds),
  Centralized LightGBM (150 rounds), Centralized MLP (80 epochs); Local only (each
  client's MLP alone, 80 epochs).
- Communication measured from serialized bytes (torch.save state_dict; booster JSON).

Reuses: SmallMLP / fedavg_aggregate / train_local / predict_proba from
psyche_d_fedavg_vs_fedtree_mlp_model.py; flwr FedXgbBagging aggregate and the local_boost
pattern of psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging.py; the TreeAvg update of
psyche_d_fedavg_vs_fedtree_step5_tree_avg.py.

Writes scripts/resilient_protocol_stage07_08_results.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # avoid torch / xgboost libomp deadlock

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import xgboost as xgb
from flwr.server.strategy.fedxgb_bagging import aggregate as flwr_fedxgb_aggregate
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psyche_d_fedavg_vs_fedtree_mlp_model as M  # noqa: E402
import resilient_protocol_common as C  # noqa: E402

OUT = C.SCRIPT_DIR / "resilient_protocol_stage07_08_results.json"
K = 2
FEDAVG_ROUNDS, LOCAL_EPOCHS = 40, 2
TREE_ROUNDS = 30
ETA_GRID = [0.03, 0.05, 0.10, 0.20, 0.30]
ETA_ANCHOR = 0.10
XGB = {"objective": "binary:logistic", "eta": 0.1, "max_depth": 4, "subsample": 0.8, "colsample_bytree": 0.8,
       "tree_method": "hist", "base_score": 0.5, "nthread": 1, "verbosity": 0}


def sig(z):
    return 1 / (1 + np.exp(-z))


def split_clients(tr, y, seed):
    rng = np.random.default_rng(seed)
    pos, neg = rng.permutation(tr[y[tr] == 1]), rng.permutation(tr[y[tr] == 0])
    cl = [np.concatenate([pos[k::K], neg[k::K]]) for k in range(K)]
    assert set(cl[0]).isdisjoint(cl[1]) and set(np.concatenate(cl)) == set(tr)
    return cl


def fed_stats(Xs):
    n = sum(np.sum(~np.isnan(X), axis=0) for X in Xs)
    s = sum(np.nansum(X, axis=0) for X in Xs)
    ss = sum(np.nansum(X ** 2, axis=0) for X in Xs)
    mean = s / np.maximum(n, 1)
    std = np.sqrt(np.maximum(ss / np.maximum(n, 1) - mean ** 2, 0))
    return mean, np.where(std < 1e-8, 1.0, std)


def prep(X, mean, std):
    Z = np.where(np.isnan(X), mean, X)
    return (Z - mean) / std


def nbytes_state(state):
    b = io.BytesIO()
    torch.save(state, b)
    return b.getbuffer().nbytes


def fedavg(Zc, yc, Zte, seed):
    torch.manual_seed(seed)
    glob = M.SmallMLP(input_dim=Zte.shape[1])
    state = M.get_state(glob)
    comm, curve = 0, []
    nk = [len(v) for v in yc]
    assert abs(sum(n / sum(nk) for n in nk) - 1) < 1e-12
    for rnd in range(FEDAVG_ROUNDS):
        states = []
        for k in range(K):
            local = M.SmallMLP(input_dim=Zte.shape[1])
            M.set_state(local, state)
            M.train_local(local, Zc[k], yc[k], epochs=LOCAL_EPOCHS, lr=1e-3, batch_size=64, seed=seed + 1000 * rnd + k)
            states.append(M.get_state(local))
        comm += 2 * sum(nbytes_state(s) for s in states)  # upload K states + broadcast global to K clients
        state = M.fedavg_aggregate(states, nk)
        M.set_state(glob, state)
        curve.append(M.predict_proba(glob, Zte))
    return curve[-1], comm, curve


def mlp_alone(Z, yv, Zte, seed, epochs=80):
    torch.manual_seed(seed)
    m = M.SmallMLP(input_dim=Zte.shape[1])
    M.train_local(m, Z, yv, epochs=epochs, lr=1e-3, batch_size=64, seed=seed)
    return M.predict_proba(m, Zte)


def local_boost(bst, dtrain):
    bst.update(dtrain, bst.num_boosted_rounds())
    n = bst.num_boosted_rounds()
    return bst[n - 1:n]


def fedxgb(Xc, yc, Xte, spw):
    params = dict(XGB, scale_pos_weight=spw)
    dms = [xgb.DMatrix(Xc[k], label=yc[k], missing=np.nan) for k in range(K)]
    dte = xgb.DMatrix(Xte, missing=np.nan)
    g, comm, curve = None, 0, []
    for _ in range(TREE_ROUNDS):
        deltas = []
        for dm in dms:
            if g is None:
                delta = xgb.train(params, dm, num_boost_round=1)
            else:
                b = xgb.Booster(params=params)
                b.load_model(bytearray(g))
                delta = local_boost(b, dm)
            deltas.append(delta.save_raw("json"))
        comm += sum(len(x) for x in deltas)
        for x in deltas:
            g = flwr_fedxgb_aggregate(g, x)
        comm += len(g) * K
        b = xgb.Booster(params=params)
        b.load_model(bytearray(g))
        curve.append(b.predict(dte))
    return curve[-1], comm, curve


def treeavg(Xc, yc, Xte, spw, eta, rounds):
    params = dict(XGB, eta=eta, scale_pos_weight=spw)
    nk = np.array([len(v) for v in yc], float)
    w = nk / nk.sum()
    assert abs(w.sum() - 1) < 1e-12
    dms = [xgb.DMatrix(Xc[k], label=yc[k], missing=np.nan) for k in range(K)]
    dce = [xgb.DMatrix(Xc[k], missing=np.nan) for k in range(K)]
    dte = xgb.DMatrix(Xte, missing=np.nan)
    marg = [np.zeros(len(v)) for v in yc]
    tm = np.zeros(Xte.shape[0])
    comm, curve = 0, []
    for _ in range(rounds):
        bs = []
        for k in range(K):
            dms[k].set_base_margin(marg[k])
            bs.append(xgb.train(params, dms[k], num_boost_round=1))
        tb = sum(len(b.save_raw("json")) for b in bs)
        comm += tb + tb * K
        for k in range(K):
            marg[k] = marg[k] + sum(w[j] * bs[j].predict(dce[k], output_margin=True) for j in range(K))
        tm = tm + sum(w[j] * bs[j].predict(dte, output_margin=True) for j in range(K))
        curve.append(sig(tm))
    return curve[-1], comm, curve


def main():
    t_start = time.time()
    torch.set_num_threads(1)
    d = C.load_features()
    F = d["features"]
    y = d["y_phq10"]
    frozen = json.loads((C.SCRIPT_DIR / "resilient_protocol_frozen_config.json").read_text())
    X = F[frozen["features"]].to_numpy(dtype=float)
    n = len(y)
    methods = ["FedAvg MLP", "FedXgbBagging", "TreeAvg", "Centralized XGBoost", "Centralized LightGBM", "Centralized MLP", "Local only"]
    oof = {m: np.full((C.N_REPEATS, n), np.nan) for m in methods}
    eta_oof = {e: np.full((C.N_REPEATS, n), np.nan) for e in ETA_GRID}
    ta60 = np.full((C.N_REPEATS, n), np.nan)
    local_k = [np.full((C.N_REPEATS, n), np.nan) for _ in range(K)]
    curves = {"FedAvg MLP": np.full((C.N_REPEATS, FEDAVG_ROUNDS, n), np.nan),
              "FedXgbBagging": np.full((C.N_REPEATS, TREE_ROUNDS, n), np.nan),
              "TreeAvg": np.full((C.N_REPEATS, TREE_ROUNDS, n), np.nan)}
    comm = {m: [] for m in ["FedAvg MLP", "FedXgbBagging", "TreeAvg"]}
    feas = []
    # Centralized LightGBM pre-pass: LightGBM aborts (OMP Error #179, pthread_mutex_init)
    # if it first initializes OpenMP after torch has, so it runs before any torch work.
    for r in range(C.N_REPEATS):
        outer = StratifiedKFold(C.N_OUTER, shuffle=True, random_state=C.SEED + r)
        for f, (tr, te) in enumerate(outer.split(X, y)):
            seed = C.SEED + 100 * r + f
            lg = LGBMClassifier(num_leaves=8, learning_rate=0.05, n_estimators=150, min_child_samples=5, is_unbalance=True,
                                subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, n_jobs=1, verbose=-1)
            oof["Centralized LightGBM"][r, te] = lg.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    for r in range(C.N_REPEATS):
        outer = StratifiedKFold(C.N_OUTER, shuffle=True, random_state=C.SEED + r)
        for f, (tr, te) in enumerate(outer.split(X, y)):
            seed = C.SEED + 100 * r + f
            cl = split_clients(tr, y, seed)
            assert all(set(c).isdisjoint(te) for c in cl)
            feas.append({"repeat": r, "fold": f, "client_n": [len(c) for c in cl], "client_pos": [int(y[c].sum()) for c in cl]})
            assert all(len(np.unique(y[c])) == 2 for c in cl)
            mean, std = fed_stats([X[c] for c in cl])
            Zc, yc = [prep(X[c], mean, std) for c in cl], [y[c] for c in cl]
            Zte = prep(X[te], mean, std)
            Xc = [X[c] for c in cl]
            pos = sum(int(v.sum()) for v in yc); neg = sum(int((v == 0).sum()) for v in yc)
            spw = neg / max(pos, 1)

            p, cb, cur = fedavg(Zc, yc, Zte, seed); oof["FedAvg MLP"][r, te] = p; comm["FedAvg MLP"].append(cb)
            curves["FedAvg MLP"][r][:, te] = np.array(cur)
            p, cb, cur = fedxgb(Xc, yc, X[te], spw); oof["FedXgbBagging"][r, te] = p; comm["FedXgbBagging"].append(cb)
            curves["FedXgbBagging"][r][:, te] = np.array(cur)
            for e in ETA_GRID:
                p, cb, cur = treeavg(Xc, yc, X[te], spw, e, TREE_ROUNDS)
                eta_oof[e][r, te] = p
                if e == ETA_ANCHOR:
                    oof["TreeAvg"][r, te] = p; comm["TreeAvg"].append(cb)
                    curves["TreeAvg"][r][:, te] = np.array(cur)
            ta60[r, te] = treeavg(Xc, yc, X[te], spw, ETA_ANCHOR, 60)[0]
            bst = xgb.train(dict(XGB, scale_pos_weight=spw), xgb.DMatrix(X[tr], label=y[tr], missing=np.nan), num_boost_round=150)
            oof["Centralized XGBoost"][r, te] = bst.predict(xgb.DMatrix(X[te], missing=np.nan))
            oof["Centralized MLP"][r, te] = mlp_alone(np.vstack(Zc), np.concatenate(yc), Zte, seed)
            for k in range(K):
                local_k[k][r, te] = mlp_alone(Zc[k], yc[k], Zte, seed + 7 + k)
        print(f"[fl] repeat {r} done ({time.time() - t_start:.0f}s)", flush=True)
    # Local only: average of the two client-alone AUROCs (each scored on the same test folds)
    oof["Local only"] = (local_k[0] + local_k[1]) / 2  # placeholder pooled score for paired tests

    res = {"feasibility": {
        "total_positives": int(y.sum()), "positives_per_outer_train_fold": sorted({fr["client_pos"][0] + fr["client_pos"][1] for fr in feas}),
        "K2_client_positives_range": [min(min(fr["client_pos"]) for fr in feas), max(max(fr["client_pos"]) for fr in feas)],
        "rule": "each client needs >= ~10 positives", "feasible": False,
        "verdict": "FL not feasible at this N: K=2 leaves 4 positives per client; results below are exploratory"},
        "design": {"K": K, "clients": "simulated, label-stratified seeded split of each outer training fold", "fedavg_rounds": FEDAVG_ROUNDS,
                   "local_epochs": LOCAL_EPOCHS, "tree_rounds": TREE_ROUNDS, "eta_anchor": ETA_ANCHOR, "round_selection": "none (final round, no validation data at this N)",
                   "features": frozen["features"]},
        "methods": {}, "fold_clients": feas}
    for m in methods:
        s = C.summarize(y, oof[m])
        if m in comm:
            s["comm_mb_per_fold_mean"] = float(np.mean(comm[m]) / 1e6)
        res["methods"][m] = s
    la = [C.summarize(y, local_k[k]) for k in range(K)]
    res["methods"]["Local only"]["per_client"] = [{"auroc": v["auroc"], "auroc_ci": v["auroc_ci"]} for v in la]
    res["methods"]["Local only"]["auroc_mean_of_clients"] = float(np.mean([v["auroc"] for v in la]))
    res["methods"]["Local only"]["note"] = "headline auroc here is the averaged score of the two client-alone models; per_client gives each client alone"
    res["treeavg_eta_sensitivity_not_used_for_selection"] = {str(e): C.summarize(y, eta_oof[e])["auroc"] for e in ETA_GRID}
    res["treeavg_60_rounds_check"] = C.summarize(y, ta60)["auroc"]
    res["comm_ratio"] = {"FedXgbBagging_over_TreeAvg": float(np.mean(comm["FedXgbBagging"]) / np.mean(comm["TreeAvg"])),
                         "FedAvg_over_TreeAvg": float(np.mean(comm["FedAvg MLP"]) / np.mean(comm["TreeAvg"]))}
    res["curves_test_fold_diagnostic_only"] = {m: [float(np.mean([roc_auc_score(y, curves[m][r, i]) for r in range(C.N_REPEATS)])) for i in range(curves[m].shape[1])] for m in curves}
    avg = {m: oof[m].mean(axis=0) for m in methods}
    pairs = [("FedAvg MLP", "Centralized MLP"), ("FedXgbBagging", "Centralized XGBoost"), ("TreeAvg", "Centralized XGBoost"),
             ("FedAvg MLP", "TreeAvg"), ("FedXgbBagging", "TreeAvg"), ("FedAvg MLP", "FedXgbBagging"), ("Centralized LightGBM", "Centralized XGBoost")]
    res["paired"] = {f"{a} - {b}": C.paired_boot_diff(y, avg[a], avg[b]) for a, b in pairs}
    res["audit"] = {"aggregation_weights_sum_to_1": "asserted for FedAvg and TreeAvg each round",
                    "client_disjointness": "asserted (clients disjoint, union = outer training fold, disjoint from test)",
                    "test_seen_during_rounds": "test-fold scores are recorded per round for the diagnostic curve only; no round, eta or hyperparameter is chosen from them",
                    "stale_weights": "FedAvg aggregates exactly the K fresh client states of the current round"}
    res["oof_avg"] = {m: avg[m].tolist() for m in methods}
    res["oof_local_client"] = [local_k[k].mean(axis=0).tolist() for k in range(K)]
    res["seconds"] = round(time.time() - t_start, 1)
    for m in methods:
        print(f"[fl] {m}: {res['methods'][m]['auroc']:.3f} {res['methods'][m]['auroc_ci']} comm {res['methods'][m].get('comm_mb_per_fold_mean')}")
    C.write_json(OUT, res)


if __name__ == "__main__":
    main()
