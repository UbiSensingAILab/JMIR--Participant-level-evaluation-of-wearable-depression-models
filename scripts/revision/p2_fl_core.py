#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. `research_datasets/` is
# gitignored; local research data, never committed to this repo.
"""JMIR revision Phase 2: shared FL core, parameterized by an arbitrary client
partition (list of participant-ID sets) so the IID reruns, per-round logging
(2.4), local-only baselines (2.3) and non-IID partitions (2.5) all use one code
path. Each method reproduces the original step scripts' logic exactly:

  FedAvg        = scripts/psyche_d_fedavg_vs_fedtree_step23_fedavg.py
  FedXgbBagging = scripts/psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging.py
  TreeAvg       = scripts/psyche_d_fedavg_vs_fedtree_step5_tree_avg.py

The only addition is per-round logging of validation AUROC and upload/download
bytes. p2_fl_rerun.py checks that the IID reruns reproduce the saved test
predictions of those scripts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xgboost as xgb
from flwr.server.strategy.fedxgb_bagging import aggregate as flwr_fedxgb_aggregate
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import audit_psyche_d_baseline_reproduction as REPRO  # noqa: E402
import psyche_d_fedavg_vs_fedtree_common as C  # noqa: E402
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import (  # noqa: E402
    N_ROUNDS as XGB_ROUNDS, NUM_LOCAL_ROUND, XGB_BASE_PARAMS, load_split_and_frame, local_boost)

OUT = ROOT / "results" / "revision"
OUT.mkdir(parents=True, exist_ok=True)
N_BOOT, BOOT_SEED = 2000, 42
# Final run (2026-10-16): the original scripts build a fresh booster every round with
# XGBoost's default seed 0, so subsample/colsample_bytree draw the SAME rows and columns
# in every round. VARY_SEED=True gives each (round, client) tree its own seed
# (100 * round + client), as one continuous xgb.train run would. Default False = original.
VARY_SEED = False


def _seeded(params, rnd, k):
    return dict(params, seed=100 * rnd + k) if VARY_SEED else params


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class Data:
    def __init__(self):
        cache, labeled, fc, y, tr, va, te = load_split_and_frame()
        self.cache, self.labeled, self.fc, self.y = cache, labeled, fc, y
        self.train_df, self.val_df, self.test_df = labeled.loc[tr], labeled.loc[va], labeled.loc[te]
        self.y_val, self.y_test = y.loc[va].to_numpy(), y.loc[te].to_numpy()
        self.test_users = self.test_df["user_id"].to_numpy()
        self.X_val_raw = self.val_df[fc].to_numpy(dtype=np.float64)
        self.X_test_raw = self.test_df[fc].to_numpy(dtype=np.float64)

    def clients(self, sites):
        return [self.train_df[self.train_df["user_id"].isin(s)] for s in sites]

    def yc(self, d):
        return self.y.loc[d.index].to_numpy()


# ---------------------------------------------------------------- FedAvg
def run_fedavg_batch(partitions, local_only=True, cols=None):
    """FedAvg runs in a separate torch-only process (p2_fl_mlp.py): on this
    macOS build, torch and xgboost segfault or deadlock when used in the same
    process. The original step scripts also ran them in separate processes."""
    import pickle
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        i, o = Path(td) / "in.pkl", Path(td) / "out.pkl"
        i.write_bytes(pickle.dumps({"partitions": [[set(s) for s in p] for p in partitions], "local_only": local_only, "cols": cols}))
        subprocess.run([sys.executable, str(Path(__file__).with_name("p2_fl_mlp.py")), str(i), str(o)], check=True)
        return pickle.loads(o.read_bytes())


# ---------------------------------------------------------------- shared XGB setup
def _xgb_params(D, client_dfs, eta=None):
    pos = sum(int((D.yc(d) == 1).sum()) for d in client_dfs)
    neg = sum(int((D.yc(d) == 0).sum()) for d in client_dfs)
    p = dict(XGB_BASE_PARAMS, scale_pos_weight=neg / max(pos, 1))
    if eta is not None:
        p["eta"] = eta
    return p


# ---------------------------------------------------------------- FedXgbBagging
def run_fedxgb(D: Data, client_dfs):
    params = _xgb_params(D, client_dfs)
    dms = [xgb.DMatrix(d[D.fc].to_numpy(dtype=np.float64), label=D.yc(d), missing=np.nan) for d in client_dfs]
    dval = xgb.DMatrix(D.X_val_raw, label=D.y_val, missing=np.nan)
    dtest = xgb.DMatrix(D.X_test_raw, label=D.y_test, missing=np.nan)
    gbytes = None
    best = (-1.0, None, -1, None)
    hist = []
    for rnd in range(1, XGB_ROUNDS + 1):
        deltas = []
        for k, dtrain in enumerate(dms):
            if gbytes is None:
                delta = xgb.train(_seeded(params, rnd, k), dtrain, num_boost_round=NUM_LOCAL_ROUND)
            else:
                bst = xgb.Booster(params=params)
                bst.load_model(bytearray(gbytes))
                if VARY_SEED:
                    bst.set_param({"seed": 100 * rnd + k})
                delta = local_boost(bst, NUM_LOCAL_ROUND, dtrain)
            deltas.append(delta.save_raw("json"))
        up = sum(len(d) for d in deltas)
        for db in deltas:
            gbytes = flwr_fedxgb_aggregate(gbytes, db)
        down = len(gbytes) * len(dms)
        eb = xgb.Booster(params=params)
        eb.load_model(bytearray(gbytes))
        vp = eb.predict(dval)
        va = float(roc_auc_score(D.y_val, vp))
        hist.append({"round": rnd, "val_auroc": va, "upload_bytes": up, "download_bytes": down})
        if va > best[0]:
            best = (va, bytes(gbytes), rnd, vp.copy())
    fb = xgb.Booster(params=params)
    fb.load_model(bytearray(best[1]))
    return {"test_pred": fb.predict(dtest), "val_pred": best[3], "best_round": best[2], "best_val_auroc": best[0], "rounds": hist}


# ---------------------------------------------------------------- TreeAvg
def run_treeavg(D: Data, client_dfs, eta=None):
    params = _xgb_params(D, client_dfs, eta)
    K = len(client_dfs)
    n = np.array([len(d) for d in client_dfs], dtype=np.float64)
    w = n / n.sum()
    cX = [d[D.fc].to_numpy(dtype=np.float64) for d in client_dfs]
    dms, cm = [], []
    for d, X in zip(client_dfs, cX):
        dm = xgb.DMatrix(X, label=D.yc(d), missing=np.nan)
        dm.set_base_margin(np.zeros(len(d)))
        dms.append(dm)
        cm.append(np.zeros(len(d)))
    dval = xgb.DMatrix(D.X_val_raw, missing=np.nan)
    dtest = xgb.DMatrix(D.X_test_raw, missing=np.nan)
    dce = [xgb.DMatrix(X, missing=np.nan) for X in cX]
    vm, tm = np.zeros(len(D.y_val)), np.zeros(len(D.y_test))
    best = (-1.0, None, -1, None)
    hist = []
    for rnd in range(1, XGB_ROUNDS + 1):
        boosters, tb = [], []
        for k in range(K):
            dms[k].set_base_margin(cm[k])
            b = xgb.train(_seeded(params, rnd, k), dms[k], num_boost_round=NUM_LOCAL_ROUND)
            boosters.append(b)
            tb.append(len(b.save_raw("json")))
        for k in range(K):
            cm[k] = cm[k] + np.sum([w[j] * b.predict(dce[k], output_margin=True) for j, b in enumerate(boosters)], axis=0)
        vm = vm + np.sum([w[j] * b.predict(dval, output_margin=True) for j, b in enumerate(boosters)], axis=0)
        tm = tm + np.sum([w[j] * b.predict(dtest, output_margin=True) for j, b in enumerate(boosters)], axis=0)
        va = float(roc_auc_score(D.y_val, sigmoid(vm)))
        hist.append({"round": rnd, "val_auroc": va, "upload_bytes": sum(tb), "download_bytes": sum(tb) * K})
        if va > best[0]:
            best = (va, tm.copy(), rnd, sigmoid(vm))
    return {"test_pred": sigmoid(best[1]), "val_pred": best[3], "best_round": best[2], "best_val_auroc": best[0], "rounds": hist}


# ---------------------------------------------------------------- local-only trees
def local_only_trees(D: Data, client_dfs):
    """Local XGBoost (same params family as centralized s4: max 150 rounds,
    early stopping 20 on the shared validation set) and local LightGBM-DART
    (same recipe as centralized step1: mean imputation, balanced weights)."""
    from sklearn.impute import SimpleImputer
    from sklearn.utils.class_weight import compute_sample_weight
    dval = xgb.DMatrix(D.X_val_raw, label=D.y_val, missing=np.nan)
    dtest = xgb.DMatrix(D.X_test_raw, missing=np.nan)
    out = {"xgboost": [], "lightgbm": []}
    for d in client_dfs:
        yc = D.yc(d)
        X = d[D.fc].to_numpy(dtype=np.float64)
        p = dict(XGB_BASE_PARAMS, scale_pos_weight=int((yc == 0).sum()) / max(int((yc == 1).sum()), 1))
        b = xgb.train(p, xgb.DMatrix(X, label=yc, missing=np.nan), num_boost_round=150,
                      evals=[(dval, "val")], early_stopping_rounds=20, verbose_eval=False)
        out["xgboost"].append(b.predict(dtest, iteration_range=(0, b.best_iteration + 1)))
        imp = SimpleImputer(strategy="mean", keep_empty_features=True)
        clf = REPRO.phase2_dart_factory()
        clf.fit(imp.fit_transform(X), yc, sample_weight=compute_sample_weight("balanced", yc))
        out["lightgbm"].append(clf.predict_proba(imp.transform(D.X_test_raw))[:, 1])
    return out


# ---------------------------------------------------------------- participant bootstrap
class Boot:
    """2000 participant-level resamples of the fixed test set, generated once
    (rng seed 42) and reused for every model so all comparisons are paired."""

    def __init__(self, users, y, n_boot=N_BOOT, seed=BOOT_SEED):
        uniq, inv = np.unique(users, return_inverse=True)
        rows = [np.where(inv == i)[0] for i in range(len(uniq))]
        rng = np.random.default_rng(seed)
        self.ix = [np.concatenate([rows[i] for i in rng.integers(0, len(uniq), len(uniq))]) for _ in range(n_boot)]
        self.y = y
        self.n_participants = len(uniq)

    def dist(self, pred, fn=roc_auc_score):
        return np.array([fn(self.y[i], pred[i]) for i in self.ix])

    @staticmethod
    def ci(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    def metric(self, pred):
        a, p = self.dist(pred), self.dist(pred, average_precision_score)
        return {"auroc": float(roc_auc_score(self.y, pred)), "auroc_ci": self.ci(a),
                "auprc": float(average_precision_score(self.y, pred)), "auprc_ci": self.ci(p)}, a
