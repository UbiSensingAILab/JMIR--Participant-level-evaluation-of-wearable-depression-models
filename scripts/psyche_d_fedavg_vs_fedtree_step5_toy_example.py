#!/usr/bin/env python3
"""Hand-checkable toy example (task Part 4): 2 clients, 2 communication
rounds, 1 new tree per client per round -> 4 trees total. Demonstrates,
with numbers a reader can verify by hand, that:

  1. TreeAvg (round-wise SAMPLE-COUNT-WEIGHTED n_k/N averaging of same-
     round client trees) is NOT the same model as simple concatenation/
     bagging (FedXgbBagging).
  2. TreeAvg is also NOT the same as averaging the two clients' fully
     independent final local-model predictions (one-shot ensembling).

This is a from-scratch rewrite (per explicit task instruction) superseding
an earlier same-day uncommitted version of this file that used a UNIFORM
1/K weight. The two clients here are deliberately asymmetric in SIZE too
(client A has 4 rows, client B has 5 rows) -- this was already necessary
so the two clients' trees are not identical, and it now doubles as a
concrete, non-uniform-weight demonstration: w_A = 4/9, w_B = 5/9, not 1/2
each.

Mechanism under test (TreeAvg, as agreed with the user):
  - At round r, client k grows ONE tree h_{k,r} continuing from the
    CURRENT SHARED global margin (exactly FedXgbBagging's client-side
    mechanic -- this part is identical between the two algorithms).
  - Bagging combines round r's K client trees by adding all of them to
    the ensemble at full weight: contribution_r(x) = sum_k h_{k,r}(x).
  - TreeAvg combines round r's K client trees by a SAMPLE-COUNT-WEIGHTED
    SUM of their outputs: contribution_r(x) = sum_k (n_k/N) * h_{k,r}(x),
    where n_k is client k's row count and N = sum_k n_k (n_A=4, n_B=5,
    N=9 here -- NOT recomputed per round, since client row counts don't
    change across rounds within one partition seed; this mirrors the real
    experiment, where N is recomputed once per partition SEED, not once
    per round).
  - Both accumulate contributions across rounds: margin(x) = sum_r contribution_r(x).
  - Final prediction = sigmoid(base_margin + margin(x)).

This script does not touch flwr's aggregate() at all (that IS the bagging
mechanic) -- TreeAvg's server-side combination is a plain, explicit
weighted arithmetic sum of margins, implemented below so every number is
traceable.
"""
from __future__ import annotations

import json

import numpy as np
import xgboost as xgb

RNG = np.random.default_rng(0)

# ---- Tiny synthetic 2-client dataset, DELIBERATELY asymmetric in both
# (a) label structure and (b) SIZE, so the two clients' per-round trees are
# NOT identical AND their TreeAvg weights are NOT uniform (a naive
# symmetric/equal-size toy set would make TreeAvg coincide with uniform
# averaging by accident -- not a proof of anything). ----
# Client A: 4 rows, perfectly separable by a single stump -> clean leaf
# values +-2.
XA = np.array([[0.1], [0.2], [0.8], [0.9]])
yA = np.array([0, 0, 1, 1])
# Client B: 5 rows, NOT perfectly separable by any single-feature stump
# (label at x=0.45 is a "1" surrounded by "0"s) -> its best depth-1 split
# is necessarily impure, giving a genuinely different (non +-2) leaf value
# and a different split threshold than client A's.
XB = np.array([[0.05], [0.25], [0.45], [0.65], [0.85]])
yB = np.array([0, 0, 1, 0, 1])

N_A, N_B = len(XA), len(XB)  # 4, 5
N_TOTAL = N_A + N_B  # 9
W_A, W_B = N_A / N_TOTAL, N_B / N_TOTAL  # 4/9 ~= 0.4444, 5/9 ~= 0.5556
CLIENT_WEIGHTS = {0: W_A, 1: W_B}

X_EVAL = np.array([[0.05], [0.30], [0.55], [0.99]])  # 4 evaluation points

BASE_SCORE = 0.0  # base_margin for both clients/rounds (log-odds space)
PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 1,          # depth-1 stumps -> trivially hand-verifiable split/leaf values
    "eta": 1.0,              # eta=1.0 so the tree's own leaf value IS the raw margin contribution
    "base_score": 0.5,       # neutral prior in probability space (margin 0)
    "tree_method": "exact",  # exact (not hist) so a 4-row toy set reliably finds its obvious split
    "min_child_weight": 0,   # relaxed regularization -- purely so this TINY toy set actually splits;
    "reg_lambda": 0,         # the real experiment (Experiment A) uses the production XGB_BASE_PARAMS,
    "gamma": 0,              # not these relaxed toy values.
}
K = 2  # number of clients


def grow_one_tree(X: np.ndarray, y: np.ndarray, current_margin: np.ndarray) -> xgb.Booster:
    """Grow exactly one new tree from (X, y) using `current_margin` as the
    base_margin (i.e. continuing boosting from wherever the global model
    currently is) -- this is the client-side step, identical in bagging
    and TreeAvg."""
    d = xgb.DMatrix(X, label=y, base_margin=current_margin)
    bst = xgb.train(PARAMS, d, num_boost_round=1)
    return bst


def tree_margin(bst: xgb.Booster, X: np.ndarray) -> np.ndarray:
    """Margin (log-odds) contribution of a single tree, evaluated with a
    zero base_margin so the returned numbers are exactly that tree's own
    leaf outputs (not cumulative)."""
    d = xgb.DMatrix(X, base_margin=np.zeros(len(X)))
    return bst.predict(d, output_margin=True)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_toy():
    client_data = [(XA, yA), (XB, yB)]

    # ---- Grow round 1 and round 2 trees for each client, TreeAvg-style
    # (each client's round-r tree is grown from the TREEAVG global margin
    # accumulated so far, since that's the actual synchronized state each
    # client would see in the real experiment). ----
    treeavg_margin_train = {0: np.zeros(len(XA)), 1: np.zeros(len(XB))}
    trees_by_round = {1: [], 2: []}
    margins_eval_by_round_client = {}  # (round, client) -> margin array on X_EVAL

    for rnd in [1, 2]:
        round_trees = []
        for k, (X, y) in enumerate(client_data):
            bst = grow_one_tree(X, y, treeavg_margin_train[k])
            round_trees.append(bst)
            margins_eval_by_round_client[(rnd, k)] = tree_margin(bst, X_EVAL)
        trees_by_round[rnd] = round_trees
        # Update each client's local view of the TreeAvg global margin by
        # the SAMPLE-COUNT-WEIGHTED SUM (n_k/N, not a uniform 1/K mean) of
        # this round's two trees. Both clients see the identical weighted
        # update (this is the "synchronized global model" requirement).
        avg_contribution_train = {
            0: sum(CLIENT_WEIGHTS[k2] * tree_margin(bst, XA) for k2, bst in enumerate(round_trees)),
            1: sum(CLIENT_WEIGHTS[k2] * tree_margin(bst, XB) for k2, bst in enumerate(round_trees)),
        }
        treeavg_margin_train[0] = treeavg_margin_train[0] + avg_contribution_train[0]
        treeavg_margin_train[1] = treeavg_margin_train[1] + avg_contribution_train[1]

    # ---- TreeAvg final prediction on X_EVAL: base_margin + sum over
    # (round, client) of (n_k/N) * that tree's margin. ----
    treeavg_total_margin = np.zeros(len(X_EVAL))
    per_tree_eval_margins = {}
    for rnd in [1, 2]:
        for k in [0, 1]:
            m = margins_eval_by_round_client[(rnd, k)]
            per_tree_eval_margins[(rnd, k)] = m
            treeavg_total_margin += CLIENT_WEIGHTS[k] * m
    treeavg_pred = sigmoid(BASE_SCORE + treeavg_total_margin)

    # ---- Bagging (FedXgbBagging-equivalent) final prediction: same 4
    # trees, but summed at FULL weight (no n_k/N scaling) -- exactly what
    # flwr.server.strategy.fedxgb_bagging.aggregate does (concatenate all
    # client trees onto a shared ensemble, weight 1 each). Note: bagging's
    # OWN client-side trees would actually differ from TreeAvg's after
    # round 1, because bagging clients see the FULL (unscaled) global
    # margin, not the weighted-averaged one. To isolate the "combination
    # rule" as the only variable (apples-to-apples, same trees), we first
    # report naive-concatenation-of-the-SAME-4-trees (a lower bound /
    # illustrative comparison), then separately grow the true independent
    # bagging ensemble so the comparison is also faithful to the real
    # algorithm. ----
    bagging_pred_same_trees = sigmoid(BASE_SCORE + sum(per_tree_eval_margins.values()))

    # ---- True, faithful bagging ensemble (client sees the REAL global
    # margin = sum of all previously-appended trees, unscaled) ----
    bagging_margin_train = {0: np.zeros(len(XA)), 1: np.zeros(len(XB))}
    bagging_eval_margin = np.zeros(len(X_EVAL))
    for rnd in [1, 2]:
        round_trees = []
        for k, (X, y) in enumerate(client_data):
            bst = grow_one_tree(X, y, bagging_margin_train[k])
            round_trees.append(bst)
        # bagging: append ALL client trees to the shared global ensemble
        # at full weight; every client's NEXT round sees the full updated
        # margin (sum of all trees so far from every client).
        for k in [0, 1]:
            contribution = sum(tree_margin(bst, [XA, XB][k]) for bst in round_trees)
            bagging_margin_train[k] = bagging_margin_train[k] + contribution
        bagging_eval_margin += sum(tree_margin(bst, X_EVAL) for bst in round_trees)
    bagging_pred_faithful = sigmoid(BASE_SCORE + bagging_eval_margin)

    # ---- Final-model-averaging (one-shot ensembling): each client trains
    # its OWN fully independent 2-round local model (no cross-client
    # communication at all during training), then we average the two
    # clients' finished-model PREDICTED PROBABILITIES (plain unweighted
    # mean -- this comparator is deliberately NOT n_k/N-weighted, since
    # "averaging two finished models' predictions" is a materially
    # different, simpler one-shot-ensembling idea than TreeAvg's per-round
    # weighted combination; the point of this comparator is that it is a
    # DIFFERENT algorithm, not a weight-matched variant of TreeAvg). ----
    final_model_preds = []
    for X, y in client_data:
        local_margin = np.zeros(len(X))
        local_trees = []
        for rnd in [1, 2]:
            bst = grow_one_tree(X, y, local_margin)
            local_trees.append(bst)
            local_margin = local_margin + tree_margin(bst, X)
        eval_margin = sum(tree_margin(bst, X_EVAL) for bst in local_trees)
        final_model_preds.append(sigmoid(BASE_SCORE + eval_margin))
    final_model_avg_pred = np.mean(final_model_preds, axis=0)

    result = {
        "client_sizes": {"n_A": N_A, "n_B": N_B, "N_total": N_TOTAL},
        "treeavg_weights_n_k_over_N": {"w_A": W_A, "w_B": W_B},
        "X_eval": X_EVAL.flatten().tolist(),
        "treeavg_pred": treeavg_pred.tolist(),
        "bagging_pred_same_4_trees_unscaled": bagging_pred_same_trees.tolist(),
        "bagging_pred_faithful_independent_ensemble": bagging_pred_faithful.tolist(),
        "final_model_averaging_pred": final_model_avg_pred.tolist(),
        "per_tree_eval_margins": {f"round{r}_client{k}": v.tolist() for (r, k), v in per_tree_eval_margins.items()},
        "equivalence_checks": {
            "treeavg_equals_bagging_same_trees": bool(np.allclose(treeavg_pred, bagging_pred_same_trees, atol=1e-9)),
            "treeavg_equals_bagging_faithful": bool(np.allclose(treeavg_pred, bagging_pred_faithful, atol=1e-9)),
            "treeavg_equals_final_model_averaging": bool(np.allclose(treeavg_pred, final_model_avg_pred, atol=1e-9)),
            "max_abs_diff_treeavg_vs_bagging_same_trees": float(np.max(np.abs(treeavg_pred - bagging_pred_same_trees))),
            "max_abs_diff_treeavg_vs_bagging_faithful": float(np.max(np.abs(treeavg_pred - bagging_pred_faithful))),
            "max_abs_diff_treeavg_vs_final_model_averaging": float(np.max(np.abs(treeavg_pred - final_model_avg_pred))),
        },
    }
    return result


if __name__ == "__main__":
    r = run_toy()
    print(json.dumps(r, indent=2))
    out_path = "scripts/psyche_d_fedavg_vs_fedtree_step5_toy_example_results.json"
    with open(out_path, "w") as f:
        json.dump(r, f, indent=2)
    print(f"\nWrote {out_path}")
    print(f"\nClient sizes: n_A={N_A} n_B={N_B} N_total={N_TOTAL}  ->  weights w_A={W_A:.4f} w_B={W_B:.4f} (NOT 0.5/0.5)")
    print("\n=== Equivalence checks (should all be False -- distinct algorithms) ===")
    for k, v in r["equivalence_checks"].items():
        print(f"  {k}: {v}")
