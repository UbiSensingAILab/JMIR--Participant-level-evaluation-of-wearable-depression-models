#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. Participant-level predictions
# go to research_datasets/revision_cache/ (gitignored); results/ holds aggregates only.
"""JMIR revision Phase 2, tasks 2.1-2.4 on the original IID partitions.

Seeds: FL split 20260928 (fixed, cached), partition seeds 1/2/3 (C.assign_clients),
MLP init seed 20260928 and per-round training seeds as in step23, XGBoost
default seed 0, LightGBM random_state 42, participant bootstrap rng 42 with 2000
resamples generated once and shared by all models (paired).
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

import p2_fl_core as F
from p2_fl_core import C, ROOT, OUT, Boot


def main():

    t0 = time.time()
    CACHE = ROOT / "research_datasets" / "revision_cache"
    CACHE.mkdir(parents=True, exist_ok=True)
    S = lambda f: json.loads((ROOT / "scripts" / f).read_text())
    saved = {"FedAvg": (S("psyche_d_fedavg_vs_fedtree_step23_results.json"), "fedavg_test_predictions"),
             "FedXgbBagging": (S("psyche_d_fedavg_vs_fedtree_step4_results.json"), "fed_xgb_bagging_test_predictions"),
             "TreeAvg": (S("psyche_d_fedavg_vs_fedtree_step5_results.json"), "tree_avg_test_predictions")}
    s5 = S("psyche_d_treeavg_audit_extend_s5_results.json")
    step1, s4 = S("psyche_d_fedavg_vs_fedtree_step1_results.json"), S("psyche_d_treeavg_audit_extend_s4_results.json")

    D = F.Data()
    B = Boot(D.test_users, D.y_test)
    assert np.array_equal(np.asarray(step1["y_test"]), D.y_test) and np.array_equal(np.asarray(s4["y_test"]), D.y_test)
    central = {"Centralized XGBoost": np.asarray(s4["centralized_xgboost_test_predictions"]),
               "Centralized LightGBM": np.asarray(step1["lightgbm_dart_same_split_test_predictions"])}

    preds, rounds, repro, local, best_rounds = {}, [], [], [], {}
    all_sites = {s: C.assign_clients(D.cache["train_participants"], k=C.N_CLIENTS, seed=s) for s in C.PARTITION_SEEDS}
    fedavg = dict(zip(C.PARTITION_SEEDS, F.run_fedavg_batch([all_sites[s] for s in C.PARTITION_SEEDS])))
    for seed in C.PARTITION_SEEDS:
        sites = all_sites[seed]
        cdfs = D.clients(sites)
        runs = {"FedAvg": fedavg[seed], "FedXgbBagging": F.run_fedxgb(D, cdfs), "TreeAvg": F.run_treeavg(D, cdfs),
                "TreeAvg eta 0.05 (sensitivity)": F.run_treeavg(D, cdfs, eta=0.05)}
        for m, r in runs.items():
            preds[(m, seed)] = r["test_pred"]
            best_rounds[(m, seed)] = r["best_round"]
            for h in r["rounds"]:
                rounds.append({"method": m, "partition_seed": seed, **h})
            if m in saved:
                d, key = saved[m]
                old = np.asarray(d["per_seed_results"][seed - 1][key])
                repro.append({"method": m, "partition_seed": seed, "max_abs_diff_vs_saved": float(np.max(np.abs(old - r["test_pred"])))})
        old05 = np.asarray(s5["grid_results_by_eta_then_seed"]["0.05"][str(seed)]["test_predictions"])
        repro.append({"method": "TreeAvg eta 0.05", "partition_seed": seed, "max_abs_diff_vs_saved": float(np.max(np.abs(old05 - preds[("TreeAvg eta 0.05 (sensitivity)", seed)])))})
        lt = F.local_only_trees(D, cdfs)
        for fam, plist in [("MLP", runs["FedAvg"]["local_only_test_preds"]), ("XGBoost", lt["xgboost"]), ("LightGBM", lt["lightgbm"])]:
            for cid, p in enumerate(plist):
                from sklearn.metrics import roc_auc_score, average_precision_score
                local.append({"family": fam, "partition_seed": seed, "client": cid, "n_rows": len(cdfs[cid]),
                              "n_participants": len(sites[cid]), "auroc": float(roc_auc_score(D.y_test, p)),
                              "auprc": float(average_precision_score(D.y_test, p))})
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(CACHE / "P2_iid_test_predictions.npz", users=D.test_users, y=D.y_test,
                        **{f"{m}|{s}": v for (m, s), v in preds.items()}, **{f"{m}|central": v for m, v in central.items()})
    pd.DataFrame(repro).to_csv(OUT / "P2_0_reproduction_check.csv", index=False)

    # ---------------- 2.1 per method x seed, participant CIs
    dist, rows = {}, []
    for (m, s), p in preds.items():
        met, dist[(m, s)] = B.metric(p)
        rows.append({"method": m, "partition_seed": s, "best_round": best_rounds[(m, s)], **{k: (round(v, 6) if not isinstance(v, list) else "%.6f-%.6f" % tuple(v)) for k, v in met.items()}})
    for m, p in central.items():
        met, dist[(m, "central")] = B.metric(p)
        rows.append({"method": m, "partition_seed": "n/a (single model)", "best_round": "", **{k: (round(v, 6) if not isinstance(v, list) else "%.6f-%.6f" % tuple(v)) for k, v in met.items()}})
    t21 = pd.DataFrame(rows)
    t21.to_csv(OUT / "P2_1_per_seed_auroc.csv", index=False)
    summ = []
    for m in ["FedAvg", "FedXgbBagging", "TreeAvg", "TreeAvg eta 0.05 (sensitivity)"]:
        a = np.array([t21[(t21.method == m) & (t21.partition_seed == s)]["auroc"].iloc[0] for s in C.PARTITION_SEEDS])
        pr = np.array([t21[(t21.method == m) & (t21.partition_seed == s)]["auprc"].iloc[0] for s in C.PARTITION_SEEDS])
        summ.append({"method": m, "auroc_mean": round(a.mean(), 4), "auroc_sd_ddof0": round(a.std(), 4), "auroc_sd_ddof1": round(a.std(ddof=1), 4),
                     "auprc_mean": round(pr.mean(), 4), "auprc_sd_ddof1": round(pr.std(ddof=1), 4)})
    pd.DataFrame(summ).to_csv(OUT / "P2_1_seed_mean_sd.csv", index=False)

    # ---------------- 2.2 paired differences (TreeAvg minus comparator)
    comps = ["FedAvg", "FedXgbBagging", "Centralized XGBoost", "Centralized LightGBM"]
    prow = []
    for c in comps:
        per_seed_d, per_seed_pt = [], []
        for s in C.PARTITION_SEEDS:
            key_c = (c, s) if (c, s) in dist else (c, "central")
            pc = preds[(c, s)] if (c, s) in preds else central[c]
            from sklearn.metrics import roc_auc_score
            pt = roc_auc_score(D.y_test, preds[("TreeAvg", s)]) - roc_auc_score(D.y_test, pc)
            dd = dist[("TreeAvg", s)] - dist[key_c]
            per_seed_d.append(dd); per_seed_pt.append(pt)
            prow.append({"comparison": f"TreeAvg - {c}", "partition_seed": s, "delta_auroc": round(pt, 6),
                         "ci": "%.6f to %.6f" % tuple(Boot.ci(dd)), "ci_excludes_0": bool(Boot.ci(dd)[0] > 0 or Boot.ci(dd)[1] < 0)})
        pooled = np.mean(per_seed_d, axis=0)
        ci = Boot.ci(pooled)
        prow.append({"comparison": f"TreeAvg - {c}", "partition_seed": "pooled (mean of 3 seed-aligned differences per bootstrap draw)",
                     "delta_auroc": round(float(np.mean(per_seed_pt)), 4), "ci": "%.6f to %.6f" % tuple(ci), "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)})
    pd.DataFrame(prow).to_csv(OUT / "P2_2_paired_differences.csv", index=False)

    # ---------------- 2.3 local-only
    ldf = pd.DataFrame(local)
    ldf.to_csv(OUT / "P2_3_local_only_per_client.csv", index=False)
    lsum = []
    for fam in ["MLP", "XGBoost", "LightGBM"]:
        for s in C.PARTITION_SEEDS + ["all"]:
            sub = ldf[ldf.family == fam] if s == "all" else ldf[(ldf.family == fam) & (ldf.partition_seed == s)]
            lsum.append({"family": fam, "partition_seed": s, "n_client_models": len(sub), "auroc_mean": round(sub.auroc.mean(), 4),
                         "auroc_sd_ddof1": round(sub.auroc.std(ddof=1), 4), "auroc_min": round(sub.auroc.min(), 4), "auroc_max": round(sub.auroc.max(), 4)})
    pd.DataFrame(lsum).to_csv(OUT / "P2_3_local_only_summary.csv", index=False)

    # ---------------- 2.4 communication curves
    rdf = pd.DataFrame([r for r in rounds if not r["method"].startswith("TreeAvg eta")])
    rdf["round_mb"] = (rdf.upload_bytes + rdf.download_bytes) / 1e6
    rdf = rdf.sort_values(["method", "partition_seed", "round"])
    for col, src in [("cum_upload_mb", "upload_bytes"), ("cum_download_mb", "download_bytes")]:
        rdf[col] = rdf.groupby(["method", "partition_seed"])[src].cumsum() / 1e6
    rdf["cum_total_mb"] = rdf.cum_upload_mb + rdf.cum_download_mb
    rdf.to_csv(OUT / "P2_4_per_round_communication.csv", index=False)
    BUDGET = 2.16
    crow = []
    for (m, s), g in rdf.groupby(["method", "partition_seed"]):
        within = g[g.cum_total_mb <= BUDGET + 1e-12]
        crow.append({"method": m, "partition_seed": s, "rounds": int(g["round"].max()),
                     "upload_bytes_per_round_mean": round(g.upload_bytes.mean()), "download_bytes_per_round_first": int(g.download_bytes.iloc[0]),
                     "download_bytes_per_round_last": int(g.download_bytes.iloc[-1]),
                     "total_upload_mb": round(g.upload_bytes.sum() / 1e6, 3), "total_download_mb": round(g.download_bytes.sum() / 1e6, 3),
                     "total_mb": round(g.round_mb.sum(), 3),
                     "rounds_within_2.16MB": len(within),
                     "val_auroc_at_last_round_within_budget": round(within.val_auroc.iloc[-1], 6) if len(within) else None,
                     "best_val_auroc_within_budget": round(within.val_auroc.max(), 4) if len(within) else None,
                     "best_val_auroc_full_run": round(g.val_auroc.max(), 4)})
    cdf = pd.DataFrame(crow)
    cdf.to_csv(OUT / "P2_4_communication_summary.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"FedAvg": "#2a6fdb", "FedXgbBagging": "#d9822b", "TreeAvg": "#1a9e5c"}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m, col in colors.items():
        for s in C.PARTITION_SEEDS:
            g = rdf[(rdf.method == m) & (rdf.partition_seed == s)]
            ax.plot(g.cum_total_mb, g.val_auroc, color=col, lw=1.6 if s == 1 else 0.8, alpha=1 if s == 1 else 0.45,
                    label=m if s == 1 else None, marker="o" if s == 1 else None, ms=2.5)
    ax.axvline(BUDGET, color="0.4", ls="--", lw=0.9)
    ax.text(BUDGET * 1.05, ax.get_ylim()[0] + 0.005, "2.16 MB", fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("Cumulative communication, MB (upload + download, log scale)")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("PSYCHE-D federated learning: validation AUROC by communication", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.01, 0.01, "Bold: partition seed 1; faint: seeds 2 and 3. 5 clients.", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "P2_4_communication_curves.png", dpi=300)

    res = {"seeds": {"fl_split_seed": C.SPLIT_SEED, "partition_seeds": C.PARTITION_SEEDS, "mlp_init_seed": 20260928,
                     "bootstrap_seed": F.BOOT_SEED, "n_boot": F.N_BOOT, "lightgbm_random_state": 42, "xgboost_seed": "default 0"},
           "test_rows": int(len(D.y_test)), "test_participants": B.n_participants, "runtime_seconds": time.time() - t0}
    (OUT / "P2_iid_meta.json").write_text(json.dumps(res, indent=2))
    print(pd.DataFrame(repro).to_string()); print(t21.to_string()); print(pd.DataFrame(summ).to_string())
    print(pd.DataFrame(prow).to_string()); print(pd.DataFrame(lsum).to_string()); print(cdf.to_string())


if __name__ == "__main__":
    main()
