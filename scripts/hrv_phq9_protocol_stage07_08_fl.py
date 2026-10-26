#!/usr/bin/env python3
# REB25-1587_MOD1 research data module; no PII in logs
"""Stage 7 (federated learning) + Stage 8 (FL audit and diagnostics), SMALL-N branch.

FL runs INSIDE the outer grouped CV: 5 outer folds x 3 repeats (outer seeds 42..44, paired with
client-partition seeds 1..3). In every outer fold:
  * 20% of training participants (one StratifiedGroupKFold(5) fold) = validation (round / eta /
    early-stopping selection only);
  * remaining training participants -> K=2 SIMULATED clients by a seeded permutation, stratified
    on whether the participant ever screens positive (so both clients see positives);
  * the outer test fold is never touched until a method's selection is finished.
Preprocessing from federated sufficient statistics (per-client count / sum / sum of squares).
Methods reuse the PSYCHE-D implementations: SmallMLP / fedavg_aggregate / train_local from
psyche_d_fedavg_vs_fedtree_mlp_model.py, Flower's real fedxgb_bagging.aggregate, and the TreeAvg
margin rule from psyche_d_fedavg_vs_fedtree_step5_tree_avg.py (re-implemented in-loop because that
script is tied to the PSYCHE-D split cache).
Two feature sets: the frozen configuration (main) and the best wearable-only arm (supplementary).
Output: hrv_phq9_protocol_stage07_08_results.json
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import pickle
import subprocess

import numpy as np

# torch and xgboost/lightgbm ship different OpenMP runtimes; in one process on this macOS
# machine they segfault (inside BCEWithLogits or XGDMatrixCreateFromDense). So the MLP part and
# the tree part run in two separate child processes (HRV_FL_PART=mlp / tree) on identical folds,
# clients and validation splits, and the parent merges their test predictions.
PART = os.environ.get("HRV_FL_PART", "driver")
if PART == "mlp":
    import torch  # noqa: E402
if PART == "tree":
    import xgboost as xgb  # noqa: E402
    import lightgbm as lgb  # noqa: E402
    from flwr.server.strategy.fedxgb_bagging import aggregate as flwr_fedxgb_aggregate  # noqa: E402
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hrv_phq9_protocol_common as C  # noqa: E402
if PART == "mlp":
    import psyche_d_fedavg_vs_fedtree_mlp_model as M  # noqa: E402

K = 2
REPEATS = 3
PARTITION_SEEDS = [1, 2, 3]
MLP_ROUNDS, MLP_LOCAL_EPOCHS = 40, 2
XGB_ROUNDS = 30
TREEAVG_ETAS = [0.03, 0.05, 0.10, 0.20, 0.30]
TREEAVG_ROUNDS = 30
XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": "auc", "eta": 0.1, "max_depth": 4, "subsample": 0.8,
              "colsample_bytree": 0.8, "num_parallel_tree": 1, "tree_method": "hist", "seed": 0, "nthread": 1}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def auc(y, p):
    return C.safe_auc(y, p)


def local_boost(bst, n_local, dtrain):
    """Flower example client_app._local_boost (bagging branch): continue boosting the loaded
    global model, return only the newly added tree(s)."""
    for _ in range(n_local):
        bst.update(dtrain, bst.num_boosted_rounds())
    n = bst.num_boosted_rounds()
    return bst[n - n_local: n]


def split_clients(train_parts, part_pos, seed):
    rng = np.random.RandomState(seed)
    pos = rng.permutation(sorted(p for p in train_parts if part_pos[p]))
    neg = rng.permutation(sorted(p for p in train_parts if not part_pos[p]))
    clients = [set(), set()]
    for i, p in enumerate(pos):
        clients[i % K].add(p)
    for i, p in enumerate(neg):
        clients[(i + 1) % K].add(p)
    return clients


def fed_stats(Xs):
    cnt = sum(np.sum(~np.isnan(X), 0) for X in Xs)
    s = sum(np.nansum(X, 0) for X in Xs)
    ss = sum(np.nansum(X ** 2, 0) for X in Xs)
    mean = s / np.maximum(cnt, 1)
    std = np.sqrt(np.maximum(ss / np.maximum(cnt, 1) - mean ** 2, 0))
    std[std < 1e-9] = 1.0
    mean[cnt == 0] = 0.0
    return mean, std


def prep(X, mean, std):
    Z = np.where(np.isnan(X), mean, X)
    return ((Z - mean) / std).astype(np.float32)




def state_bytes(state):
    buf = io.BytesIO()
    torch.save(state, buf)
    return len(buf.getvalue())


def run_fold(X, y, g, tr, te, part_pos, rep, fold, part_seed):
    # validation carve-out from training participants
    sg = StratifiedGroupKFold(5, shuffle=True, random_state=C.SEED + 7 * rep + fold)
    a, b = next(sg.split(X[tr], y[tr], g[tr]))
    va, trn = tr[b], tr[a]
    clients_p = split_clients(set(g[trn]), part_pos, part_seed * 100 + fold)
    cl_idx = [trn[np.isin(g[trn], list(cp))] for cp in clients_p]
    # ---- disjointness and weight assertions (Stage 8 audit)
    assert clients_p[0].isdisjoint(clients_p[1])
    for cp in clients_p:
        assert cp.isdisjoint(set(g[va])) and cp.isdisjoint(set(g[te]))
    assert set(g[va]).isdisjoint(set(g[te]))
    n_k = np.array([len(i) for i in cl_idx], float)
    w = n_k / n_k.sum()
    assert abs(w.sum() - 1) < 1e-12
    info = {"repeat": rep, "fold": fold, "partition_seed": part_seed,
            "clients": [{"participants": len(cp), "rows": int(len(ix)), "pos_rows": int(y[ix].sum()),
                         "pos_participants": int(sum(part_pos[p] for p in cp))} for cp, ix in zip(clients_p, cl_idx)],
            "val": {"participants": int(len(set(g[va]))), "rows": int(len(va)), "pos_rows": int(y[va].sum())},
            "test": {"participants": int(len(set(g[te]))), "rows": int(len(te)), "pos_rows": int(y[te].sum())}}
    yv, yt = y[va], y[te]
    preds, curves, comm = {}, {}, {}

    # ---- federated preprocessing for the MLP
    mean, std = fed_stats([X[ix] for ix in cl_idx])
    Zc = [prep(X[ix], mean, std) for ix in cl_idx]
    Zv, Zt = prep(X[va], mean, std), prep(X[te], mean, std)

    if PART == "tree":
        return info, *tree_part(X, y, g, va, te, cl_idx, w, yv, info)
    # FedAvg MLP
    torch.manual_seed(M.SEED); glob = M.SmallMLP(input_dim=X.shape[1])
    gstate = M.get_state(glob)
    best = (-np.inf, None, 0); curve = []; bytes_total = 0
    for r in range(1, MLP_ROUNDS + 1):
        states = []
        for k in range(K):
            loc = M.SmallMLP(input_dim=X.shape[1]); M.set_state(loc, gstate)
            M.train_local(loc, Zc[k], y[cl_idx[k]], epochs=MLP_LOCAL_EPOCHS, seed=1000 * r + k)
            states.append(M.get_state(loc))
            bytes_total += state_bytes(states[-1])           # upload
        gstate = M.fedavg_aggregate(states, [len(i) for i in cl_idx])
        bytes_total += state_bytes(gstate) * K                # broadcast
        M.set_state(glob, gstate)
        va_auc = auc(yv, M.predict_proba(glob, Zv)); curve.append(va_auc)
        if np.nan_to_num(va_auc, nan=-1) > best[0]:
            best = (np.nan_to_num(va_auc, nan=-1), {k2: v.clone() for k2, v in gstate.items()}, r)
    M.set_state(glob, best[1]); preds["FedAvg MLP"] = M.predict_proba(glob, Zt); curves["FedAvg MLP"] = curve
    comm["FedAvg MLP"] = bytes_total; info["fedavg_best_round"] = best[2]
    U = np.concatenate(cl_idx)
    # centralized MLP, early stop on validation (patience 20, max 300 epochs)
    mean_c, std_c = fed_stats([X[U]])
    Zu, Zv2, Zt2 = prep(X[U], mean_c, std_c), prep(X[va], mean_c, std_c), prep(X[te], mean_c, std_c)
    torch.manual_seed(M.SEED); cm_ = M.SmallMLP(input_dim=X.shape[1])
    best = (-1, M.get_state(cm_)); bad = 0
    for ep in range(300):
        M.train_local(cm_, Zu, y[U], epochs=1, seed=ep)
        v = np.nan_to_num(auc(yv, M.predict_proba(cm_, Zv2)), nan=-1)
        if v > best[0]:
            best, bad = (v, M.get_state(cm_)), 0
        else:
            bad += 1
            if bad >= 20:
                break
    M.set_state(cm_, best[1]); preds["Centralized MLP"] = M.predict_proba(cm_, Zt2)
    # local only: each client alone, MLP 80 epochs, client-local preprocessing
    for k in range(K):
        mk, sk = fed_stats([X[cl_idx[k]]])
        torch.manual_seed(M.SEED); lm = M.SmallMLP(input_dim=X.shape[1])
        M.train_local(lm, prep(X[cl_idx[k]], mk, sk), y[cl_idx[k]], epochs=80, seed=k)
        preds[f"Local only (client {k + 1})"] = M.predict_proba(lm, prep(X[te], mk, sk))
    return info, preds, curves, comm


def tree_part(X, y, g, va, te, cl_idx, w, yv, info):
    preds, curves, comm = {}, {}, {}

    # ---- XGBoost-family: native NaN, federated scale_pos_weight from counts
    tp = sum(int(y[ix].sum()) for ix in cl_idx); tn = sum(int((y[ix] == 0).sum()) for ix in cl_idx)
    params = dict(XGB_PARAMS, scale_pos_weight=tn / max(tp, 1))
    dcl = [xgb.DMatrix(X[ix], label=y[ix], missing=np.nan) for ix in cl_idx]
    dv, dt = xgb.DMatrix(X[va], missing=np.nan), xgb.DMatrix(X[te], missing=np.nan)

    # FedXgbBagging
    gbytes = None; best = (-np.inf, None, 0); curve = []; up = down = 0
    for r in range(1, XGB_ROUNDS + 1):
        deltas = []
        for d in dcl:
            if gbytes is None:
                delta = xgb.train(params, d, num_boost_round=1)
            else:
                bst = xgb.Booster(params=params); bst.load_model(bytearray(gbytes))
                delta = local_boost(bst, 1, d)
            deltas.append(delta.save_raw("json"))
        up += sum(len(z) for z in deltas)
        for z in deltas:
            gbytes = flwr_fedxgb_aggregate(gbytes, z)
        down += len(gbytes) * K
        eb = xgb.Booster(params=params); eb.load_model(bytearray(gbytes))
        va_auc = auc(yv, eb.predict(dv)); curve.append(va_auc)
        if np.nan_to_num(va_auc, nan=-1) > best[0]:
            best = (np.nan_to_num(va_auc, nan=-1), bytes(gbytes), r)
    eb = xgb.Booster(params=params); eb.load_model(bytearray(best[1]))
    preds["FedXgbBagging"] = eb.predict(dt); curves["FedXgbBagging"] = curve; comm["FedXgbBagging"] = up + down
    info["fedxgb_best_round"] = best[2]

    # TreeAvg: margin = sum_r sum_k eta * w_k * h_{k,r}  (eta applied inside the tree leaves)
    def treeavg(eta, rounds):
        p2 = dict(params, eta=eta)
        cm = [np.zeros(len(ix)) for ix in cl_idx]
        dce = [xgb.DMatrix(X[ix], missing=np.nan) for ix in cl_idx]
        for z in dce + [dv, dt]:
            pass
        dv0 = xgb.DMatrix(X[va], missing=np.nan); dv0.set_base_margin(np.zeros(len(va)))
        dt0 = xgb.DMatrix(X[te], missing=np.nan); dt0.set_base_margin(np.zeros(len(te)))
        for k, z in enumerate(dce):
            z.set_base_margin(np.zeros(len(cl_idx[k])))
        vm, tm = np.zeros(len(va)), np.zeros(len(te))
        best = (-np.inf, None, 0); curve = []; byt = 0
        for r in range(1, rounds + 1):
            trees, tb = [], 0
            for k in range(K):
                dcl[k].set_base_margin(cm[k])
                bst = xgb.train(p2, dcl[k], num_boost_round=1)
                trees.append(bst); tb += len(bst.save_raw("json"))
            byt += tb + tb * K
            for k in range(K):
                cm[k] = cm[k] + sum(w[k2] * t.predict(dce[k], output_margin=True) for k2, t in enumerate(trees))
            vm = vm + sum(w[k2] * t.predict(dv0, output_margin=True) for k2, t in enumerate(trees))
            tm = tm + sum(w[k2] * t.predict(dt0, output_margin=True) for k2, t in enumerate(trees))
            va_auc = auc(yv, sigmoid(vm)); curve.append(va_auc)
            if np.nan_to_num(va_auc, nan=-1) > best[0]:
                best = (np.nan_to_num(va_auc, nan=-1), tm.copy(), r)
        return best, curve, byt

    ta = {}
    for eta in TREEAVG_ETAS:
        ta[eta] = treeavg(eta, TREEAVG_ROUNDS)
    eta_best = max(TREEAVG_ETAS, key=lambda e: ta[e][0][0])
    preds["TreeAvg"] = sigmoid(ta[eta_best][0][1]); curves["TreeAvg"] = ta[eta_best][1]; comm["TreeAvg"] = ta[eta_best][2]
    preds["TreeAvg (anchor eta 0.10)"] = sigmoid(ta[0.10][0][1])
    b60 = treeavg(eta_best, 60)
    preds["TreeAvg 60 rounds"] = sigmoid(b60[0][1])
    info["treeavg_eta_val_auroc"] = {str(e): float(ta[e][0][0]) for e in TREEAVG_ETAS}
    info["treeavg_eta_chosen"] = eta_best; info["treeavg_best_round"] = ta[eta_best][0][2]

    # ---- centralized comparators on the union of client rows (same data FL saw)
    U = np.concatenate(cl_idx)
    du = xgb.DMatrix(X[U], label=y[U], missing=np.nan); dvl = xgb.DMatrix(X[va], label=yv, missing=np.nan)
    if len(np.unique(yv)) == 2:
        cx = xgb.train(params, du, num_boost_round=150, evals=[(dvl, "val")], early_stopping_rounds=20, verbose_eval=False)
        preds["Centralized XGBoost"] = cx.predict(dt, iteration_range=(0, cx.best_iteration + 1))
    else:
        preds["Centralized XGBoost"] = xgb.train(params, du, num_boost_round=150).predict(dt)
    lg = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=15, min_child_samples=5, class_weight="balanced",
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1)
    if len(np.unique(yv)) == 2:
        lg.fit(X[U], y[U], eval_set=[(X[va], yv)], eval_metric="auc", callbacks=[lgb.early_stopping(20, verbose=False)])
    else:
        lg.fit(X[U], y[U])
    preds["Centralized LightGBM"] = lg.predict_proba(X[te])[:, 1]
    return preds, curves, comm


def collect(seg, cols, label):
    """Child process: run every outer fold for this PART and return raw predictions."""
    X, y, g = C.Xy(seg, cols)
    part_pos = seg.groupby("deviceId")["y"].max().to_dict()
    infos, all_curves, all_comm, pred_mat = [], {}, {}, {}
    for rep in range(REPEATS):
        for fold, (tr, te) in enumerate(C.outer_folds(y, g, rep)):
            info, preds, curves, comm = run_fold(X, y, g, tr, te, part_pos, rep, fold, PARTITION_SEEDS[rep])
            infos.append(info)
            for m, p in preds.items():
                pred_mat.setdefault(m, np.full((len(y), REPEATS), np.nan))[te, rep] = p
            for m, c in curves.items():
                all_curves.setdefault(m, []).append(c)
            for m, b in comm.items():
                all_comm.setdefault(m, []).append(b)
            print(PART, label, rep, fold, {m: round(C.safe_auc(y[te], p), 3) for m, p in preds.items()}, flush=True)
    return {"infos": infos, "curves": all_curves, "comm": all_comm, "pred_mat": pred_mat}


def summarize(seg, cols, label, raws):
    X, y, g = C.Xy(seg, cols)
    infos = raws["mlp"]["infos"]
    # identical folds / clients in both child processes
    for a, b in zip(raws["mlp"]["infos"], raws["tree"]["infos"]):
        assert a["clients"] == b["clients"] and a["val"] == b["val"] and a["test"] == b["test"]
        a.update({k: v for k, v in b.items() if k.startswith(("fedxgb", "treeavg"))})
    pred_mat = {**raws["mlp"]["pred_mat"], **raws["tree"]["pred_mat"]}
    all_curves = {**raws["mlp"]["curves"], **raws["tree"]["curves"]}
    all_comm = {**raws["mlp"]["comm"], **raws["tree"]["comm"]}
    methods = {}
    ref = np.nanmean(pred_mat["Centralized XGBoost"], axis=1)
    for m, P in pred_mat.items():
        pa = np.nanmean(P, axis=1)
        per = [C.safe_auc(y, P[:, r]) for r in range(REPEATS)]
        ci, _ = C.boot_ci(y, pa, g)
        methods[m] = {"mean_auroc_over_repeats": float(np.mean(per)), "sd_auroc_over_repeats": float(np.std(per)), "per_repeat": per,
                      "auroc_repeat_avg_pred": float(roc_auc_score(y, pa)), "auroc_ci": ci,
                      "paired_vs_centralized_xgboost": None if m == "Centralized XGBoost" else C.paired_boot_diff(y, pa, ref, g),
                      "comm_mb_per_fold_mean": float(np.mean(all_comm[m]) / 1e6) if m in all_comm else None}
    fl = ["FedAvg MLP", "FedXgbBagging", "TreeAvg"]
    pairwise = {}
    for i, a in enumerate(fl):
        for b in fl[i + 1:]:
            pairwise[f"{a} - {b}"] = C.paired_boot_diff(y, np.nanmean(pred_mat[a], 1), np.nanmean(pred_mat[b], 1), g)
    pairwise["FedAvg MLP - Centralized MLP"] = C.paired_boot_diff(y, np.nanmean(pred_mat["FedAvg MLP"], 1), np.nanmean(pred_mat["Centralized MLP"], 1), g)
    for a in ["FedXgbBagging", "TreeAvg"]:
        pairwise[f"{a} - Centralized XGBoost"] = C.paired_boot_diff(y, np.nanmean(pred_mat[a], 1), ref, g)
    curves_mean = {m: np.nanmean(np.array(c, dtype=float), axis=0).tolist() for m, c in all_curves.items()}
    pos_rows = [c["pos_rows"] for i in infos for c in i["clients"]]
    pos_parts = [c["pos_participants"] for i in infos for c in i["clients"]]
    return {"feature_set": label, "n_features": len(cols), "methods": methods, "pairwise": pairwise,
            "learning_curves_val_auroc_mean": curves_mean, "folds": infos,
            "client_pos_rows_min_median_max": [int(np.min(pos_rows)), float(np.median(pos_rows)), int(np.max(pos_rows))],
            "client_pos_participants_min_median_max": [int(np.min(pos_parts)), float(np.median(pos_parts)), int(np.max(pos_parts))],
            "val_pos_rows_min": int(min(i["val"]["pos_rows"] for i in infos))}


def featuresets(seg):
    fz = C.load_json("hrv_phq9_protocol_frozen_config.json")
    s4 = C.load_json("hrv_phq9_protocol_stage04_results.json")
    wear_arm = s4["best_wearable_only"].split("|")[0]
    return {"main": (fz["features"], f"frozen ({fz['arm']})"),
            "supplementary_wearable": (C.arm_columns(seg, wear_arm), f"wearable only ({wear_arm})")}


def main():
    t = time.time()
    seg = C.build_segments()
    fsets = featuresets(seg)
    if PART in ("mlp", "tree"):
        if PART == "mlp":
            torch.set_num_threads(1)
        out = {k: collect(seg, cols, lab) for k, (cols, lab) in fsets.items()}
        (C.CACHE / f"hrv_phq9_protocol_fl_part_{PART}.pkl").write_bytes(pickle.dumps(out))
        return
    for part in ("mlp", "tree"):
        r = subprocess.run([sys.executable, "-u", __file__], env={**os.environ, "HRV_FL_PART": part, "OMP_NUM_THREADS": "1"})
        assert r.returncode == 0, f"FL child process {part} failed with code {r.returncode}"
    raws = {p: pickle.loads((C.CACHE / f"hrv_phq9_protocol_fl_part_{p}.pkl").read_bytes()) for p in ("mlp", "tree")}
    res = {k: summarize(seg, cols, lab, {p: raws[p][k] for p in raws}) for k, (cols, lab) in fsets.items()}
    audit = {
        "aggregation_weights_sum_to_1": True, "client_disjointness_asserted": True,
        "same_folds_clients_in_both_processes_asserted": True,
        "test_never_used_in_rounds": "test predictions are stored only as the snapshot at the validation-best round; selection uses validation AUROC only",
        "no_double_aggregation": "FedAvg aggregates exactly one fresh local state per client per round; FedXgbBagging appends each client's one new tree once",
        "mlp_comm_measured": "torch.save(state_dict) bytes, upload K x per round + broadcast K x per round, all 40 rounds",
        "xgb_comm_measured": "uploaded delta-booster JSON + broadcast of the growing global ensemble x K",
        "treeavg_comm_measured": "K tree JSONs up + the same K trees re-broadcast to K clients, every round",
    }
    out = {"K": K, "clients": "simulated: seeded permutation of training participants, stratified on ever-positive; inside each outer fold",
           "repeats": REPEATS, "partition_seeds": PARTITION_SEEDS, "main": res["main"],
           "supplementary_wearable": res["supplementary_wearable"], "audit": audit,
           "treeavg_equation": "margin(x) = sum_r sum_k eta * w_k * h_{k,r}(x), w_k = n_k / N, p = sigmoid(margin)",
           "seconds": round(time.time() - t, 1)}
    C.write_json("hrv_phq9_protocol_stage07_08_results.json", out)
    for k in res:
        for m, v in res[k]["methods"].items():
            print(k, m, round(v["mean_auroc_over_repeats"], 4), v["auroc_ci"], v["comm_mb_per_fold_mean"])


if __name__ == "__main__":
    main()
