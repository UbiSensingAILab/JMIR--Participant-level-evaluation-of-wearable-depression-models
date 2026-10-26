#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. Participant-level predictions go to
# research_datasets/revision_cache/ (gitignored); results/ holds aggregates only.
"""Final analysis run, tasks 2-5 (federated learning).

T2  FedAvg, FedXgbBagging and TreeAvg on the development-selected input set of R6
    (P+S+L, 41 inputs) and, for tasks 3 and 5, on the frozen 152 inputs. Same fixed
    split (seed 20260928), partition seeds 1-3, same hyperparameters and tuning rules as
    the original runs: best round chosen on validation AUROC; TreeAvg eta chosen from
    {0.03, 0.05, 0.10, 0.20, 0.30} by mean validation AUROC over the 3 seeds (rule of
    scripts/psyche_d_treeavg_audit_extend_s5_lr_grid.py). Centralized LightGBM (frozen
    DART recipe, 70% train) and centralized XGBoost (s4 recipe, early stopping 20 on
    validation, max 150 rounds) refitted on the same inputs.
T3  Seed-aligned paired differences: TreeAvg minus local-only XGBoost and LightGBM, and
    FedAvg minus local-only MLP, where "local-only" = mean AUROC of the 5 client models
    computed inside each bootstrap draw.
T4  Non-IID: 5 clients = quintiles of screener birth year (no region variable exists),
    frozen 152 inputs, the 3 methods, TreeAvg at the eta selected in T2 for 152 inputs.
T5  Per-round validation AUROC, upload and download bytes, cumulative MB; budget 2.16 MB.
Bootstrap: 606 test participants, 2000 draws, numpy default_rng(42), shared by every model.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

import p2_fl_core as F
from p2_fl_core import C, ROOT, OUT, Boot, REPRO
from p2_fl_noniid import age_partition, describe
from psyche_d_fedavg_vs_fedtree_step4_fed_xgb_bagging import XGB_BASE_PARAMS

CACHE = ROOT / "research_datasets" / "revision_cache"
ETA_GRID = [0.03, 0.05, 0.10, 0.20, 0.30]
BUDGET = 2.16
SEEDS = C.PARTITION_SEEDS
import os  # noqa: E402
F.VARY_SEED = os.environ.get("FL_VARY_SEED") == "1"
TAG = "_seedvary" if F.VARY_SEED else ""


def o(name):
    stem, ext = name.rsplit(".", 1)
    return OUT / f"{stem}{TAG}.{ext}"


def set_cols(D, cols):
    D.fc = list(cols)
    D.X_val_raw = D.val_df[D.fc].to_numpy(dtype=np.float64)
    D.X_test_raw = D.test_df[D.fc].to_numpy(dtype=np.float64)


def centralized(D):
    ytr = D.yc(D.train_df)
    Xtr = D.train_df[D.fc].to_numpy(dtype=np.float64)
    imp = SimpleImputer(strategy="mean", keep_empty_features=True)
    clf = REPRO.phase2_dart_factory()
    clf.fit(imp.fit_transform(Xtr), ytr, sample_weight=compute_sample_weight("balanced", ytr))
    lgb_test = clf.predict_proba(imp.transform(D.X_test_raw))[:, 1]
    p = dict(XGB_BASE_PARAMS, scale_pos_weight=int((ytr == 0).sum()) / max(int((ytr == 1).sum()), 1))
    dval = xgb.DMatrix(D.X_val_raw, label=D.y_val, missing=np.nan)
    b = xgb.train(p, xgb.DMatrix(Xtr, label=ytr, missing=np.nan), num_boost_round=150,
                  evals=[(xgb.DMatrix(Xtr, label=ytr, missing=np.nan), "train"), (dval, "val")],
                  early_stopping_rounds=20, verbose_eval=False)
    xgb_test = b.predict(xgb.DMatrix(D.X_test_raw, missing=np.nan), iteration_range=(0, b.best_iteration + 1))
    return {"Centralized LightGBM": lgb_test, "Centralized XGBoost": xgb_test}


def main():
    t0 = time.time()
    fmap = pd.read_csv(OUT / "P0_4_psyche_d_feature_groups.csv").set_index("feature")["group"]
    D0 = F.Data()
    all_cols = list(D0.fc)
    P = ["phq9_score_start", "phq9_cat_start"]
    psl = [c for c in all_cols if fmap[c] in ("screener", "lifestyle_medication")] + P
    psl = [c for c in all_cols if c in set(psl)]  # keep frame order
    assert len(psl) == 41
    SETS = {"frozen 152": all_cols, "P+S+L 41": psl}
    B = Boot(D0.test_users, D0.y_test)
    sites = {s: C.assign_clients(D0.cache["train_participants"], k=C.N_CLIENTS, seed=s) for s in SEEDS}
    age_sites, age_ranges = age_partition(D0)

    preds, rounds, best_round, eta_rows, local = {}, [], {}, [], {}
    for sname, cols in SETS.items():
        D = F.Data()
        set_cols(D, cols)
        fa = F.run_fedavg_batch([sites[s] for s in SEEDS] + [age_sites] * (sname == "frozen 152"), local_only=True, cols=cols)
        grid = {}
        for i, s in enumerate(SEEDS):
            cdfs = D.clients(sites[s])
            runs = {"FedAvg": fa[i], "FedXgbBagging": F.run_fedxgb(D, cdfs)}
            for eta in ETA_GRID:
                grid[(eta, s)] = F.run_treeavg(D, cdfs, eta=eta)
            for m, r in runs.items():
                preds[(sname, "IID", m, s)] = r["test_pred"]
                best_round[(sname, "IID", m, s)] = r["best_round"]
                rounds += [{"input_set": sname, "partition": "IID", "method": m, "partition_seed": s, **h} for h in r["rounds"]]
            lt = F.local_only_trees(D, cdfs)
            local[(sname, s)] = {"MLP": fa[i]["local_only_test_preds"], "XGBoost": lt["xgboost"], "LightGBM": lt["lightgbm"]}
            print(f"{sname} seed {s} ({time.time() - t0:.0f}s)", flush=True)
        mean_val = {eta: float(np.mean([grid[(eta, s)]["best_val_auroc"] for s in SEEDS])) for eta in ETA_GRID}
        sel = max(ETA_GRID, key=lambda e: mean_val[e])
        for eta in ETA_GRID:
            eta_rows.append({"input_set": sname, "eta": eta, "mean_val_auroc": mean_val[eta], "selected": eta == sel,
                             "best_rounds": "/".join(str(grid[(eta, s)]["best_round"]) for s in SEEDS),
                             "test_auroc_mean": float(np.mean([roc_auc_score(D.y_test, grid[(eta, s)]["test_pred"]) for s in SEEDS]))})
        for s in SEEDS:
            r = grid[(sel, s)]
            preds[(sname, "IID", "TreeAvg", s)] = r["test_pred"]
            best_round[(sname, "IID", "TreeAvg", s)] = r["best_round"]
            rounds += [{"input_set": sname, "partition": "IID", "method": "TreeAvg", "partition_seed": s, **h} for h in r["rounds"]]
        for m, p in centralized(D).items():
            preds[(sname, "central", m, 0)] = p
        if sname == "frozen 152":
            cdfs = D.clients(age_sites)
            runs = {"FedAvg": fa[len(SEEDS)], "FedXgbBagging": F.run_fedxgb(D, cdfs), "TreeAvg": F.run_treeavg(D, cdfs, eta=sel)}
            for m, r in runs.items():
                preds[(sname, "age quintile", m, 0)] = r["test_pred"]
                best_round[(sname, "age quintile", m, 0)] = r["best_round"]
                rounds += [{"input_set": sname, "partition": "age quintile", "method": m, "partition_seed": 0, **h} for h in r["rounds"]]
            age_desc = describe(D, age_sites)
            for row in age_desc:
                row["birthyear_range"] = age_ranges[row["client"]]
        print(f"{sname} done, eta {sel} ({time.time() - t0:.0f}s)", flush=True)

    # reproduction checks against saved frozen-input predictions
    S_ = lambda f: json.loads((ROOT / "scripts" / f).read_text())  # noqa: E731
    chk = []
    old = {"FedAvg": ("psyche_d_fedavg_vs_fedtree_step23_results.json", "fedavg_test_predictions"),
           "FedXgbBagging": ("psyche_d_fedavg_vs_fedtree_step4_results.json", "fed_xgb_bagging_test_predictions")}
    for m, (f, k) in old.items():
        d = S_(f)
        for s in SEEDS:
            chk.append({"item": f"{m} seed {s}", "max_abs_diff": float(np.max(np.abs(np.asarray(d["per_seed_results"][s - 1][k]) - preds[("frozen 152", "IID", m, s)])))})
    g5 = S_("psyche_d_treeavg_audit_extend_s5_results.json")["grid_results_by_eta_then_seed"]["0.05"]
    for s in SEEDS:
        chk.append({"item": f"TreeAvg eta 0.05 seed {s}", "max_abs_diff": float(np.max(np.abs(np.asarray(g5[str(s)]["test_predictions"]) - preds[("frozen 152", "IID", "TreeAvg", s)])))})
    chk.append({"item": "Centralized LightGBM", "max_abs_diff": float(np.max(np.abs(np.asarray(S_("psyche_d_fedavg_vs_fedtree_step1_results.json")["lightgbm_dart_same_split_test_predictions"]) - preds[("frozen 152", "central", "Centralized LightGBM", 0)])))})
    chk.append({"item": "Centralized XGBoost", "max_abs_diff": float(np.max(np.abs(np.asarray(S_("psyche_d_treeavg_audit_extend_s4_results.json")["centralized_xgboost_test_predictions"]) - preds[("frozen 152", "central", "Centralized XGBoost", 0)])))})
    pd.DataFrame(chk).to_csv(o("T2_reproduction_check.csv"), index=False)

    np.savez_compressed(CACHE / f"T2_T5_fl_test_predictions{TAG}.npz", users=D0.test_users, y=D0.y_test,
                        **{"|".join(map(str, k)): v for k, v in preds.items()},
                        **{f"{sn}|IID|local {fam}|{s}|client{c}": p for (sn, s), d in local.items() for fam, pl in d.items() for c, p in enumerate(pl)})

    # ---------------- metrics
    dist, rows = {}, []
    for k, p in preds.items():
        met, dist[k] = B.metric(p)
        rows.append({"input_set": k[0], "partition": k[1], "method": k[2], "partition_seed": k[3] or "n/a",
                     "best_round": best_round.get(k, ""), **met})
    per = pd.DataFrame(rows)
    per.to_csv(o("T2_T4_per_seed_auroc.csv"), index=False)
    summ = []
    for sname in SETS:
        for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
            a = np.array([roc_auc_score(D0.y_test, preds[(sname, "IID", m, s)]) for s in SEEDS])
            summ.append({"input_set": sname, "method": m, "auroc_seed1": a[0], "auroc_seed2": a[1], "auroc_seed3": a[2],
                         "auroc_mean": a.mean(), "auroc_sd_ddof1": a.std(ddof=1)})
        for m in ["Centralized LightGBM", "Centralized XGBoost"]:
            summ.append({"input_set": sname, "method": m, "auroc_mean": roc_auc_score(D0.y_test, preds[(sname, "central", m, 0)])})
    pd.DataFrame(summ).to_csv(o("T2_seed_mean_sd.csv"), index=False)

    def pooled(name, sname, a_keys, b_keys, a_pts, b_pts, task):
        ds = [dist[a] - (b if isinstance(b, np.ndarray) else dist[b]) for a, b in zip(a_keys, b_keys)]
        out = []
        for s, d, pa, pb in zip(SEEDS, ds, a_pts, b_pts):
            ci = Boot.ci(d)
            out.append({"task": task, "input_set": sname, "comparison": name, "partition_seed": s, "delta": pa - pb, "ci": ci,
                        "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)})
        ci = Boot.ci(np.mean(ds, axis=0))
        out.append({"task": task, "input_set": sname, "comparison": name, "partition_seed": "pooled",
                    "delta": float(np.mean(np.array(a_pts) - np.array(b_pts))), "ci": ci, "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)})
        return out

    auc = lambda p: float(roc_auc_score(D0.y_test, p))  # noqa: E731
    PR = []
    for sname in SETS:
        for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
            for c in ["Centralized LightGBM", "Centralized XGBoost"]:
                ck = (sname, "central", c, 0)
                PR += pooled(f"{m} - {c}", sname, [(sname, "IID", m, s) for s in SEEDS], [ck] * 3,
                             [auc(preds[(sname, "IID", m, s)]) for s in SEEDS], [auc(preds[ck])] * 3, "T2")
        for c in ["FedAvg", "FedXgbBagging"]:
            PR += pooled(f"TreeAvg - {c}", sname, [(sname, "IID", "TreeAvg", s) for s in SEEDS], [(sname, "IID", c, s) for s in SEEDS],
                         [auc(preds[(sname, "IID", "TreeAvg", s)]) for s in SEEDS], [auc(preds[(sname, "IID", c, s)]) for s in SEEDS], "T2")
    # T2 extra: P+S+L minus frozen inputs for each method (same seeds)
    for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
        PR += pooled(f"{m}: P+S+L 41 - frozen 152", "both", [("P+S+L 41", "IID", m, s) for s in SEEDS], [("frozen 152", "IID", m, s) for s in SEEDS],
                     [auc(preds[("P+S+L 41", "IID", m, s)]) for s in SEEDS], [auc(preds[("frozen 152", "IID", m, s)]) for s in SEEDS], "T2")

    # ---------------- T3 benefit of federation vs local-only (mean of 5 clients per draw)
    lrows = []
    for (sname, s), fams in local.items():
        for fam, pl in fams.items():
            ldist = np.mean([B.dist(p) for p in pl], axis=0)
            dist[(sname, "local", fam, s)] = ldist
            lpt = float(np.mean([auc(p) for p in pl]))
            lrows.append({"input_set": sname, "family": fam, "partition_seed": s, "mean_client_auroc": lpt,
                          "client_aurocs": [auc(p) for p in pl], "sd_ddof1": float(np.std([auc(p) for p in pl], ddof=1)),
                          "mean_client_auroc_ci": Boot.ci(ldist)})
    ldf = pd.DataFrame(lrows)
    ldf.to_csv(o("T3_local_only.csv"), index=False)
    lp = {(r["input_set"], r["family"], r["partition_seed"]): r["mean_client_auroc"] for r in lrows}
    for sname in SETS:
        for fed, fam in [("TreeAvg", "XGBoost"), ("TreeAvg", "LightGBM"), ("FedAvg", "MLP")]:
            PR += pooled(f"{fed} - local-only {fam} (mean of 5 clients)", sname, [(sname, "IID", fed, s) for s in SEEDS],
                         [dist[(sname, "local", fam, s)] for s in SEEDS], [auc(preds[(sname, "IID", fed, s)]) for s in SEEDS],
                         [lp[(sname, fam, s)] for s in SEEDS], "T3")

    # ---------------- T4 non-IID (single partition)
    for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
        a = ("frozen 152", "age quintile", m, 0)
        for c in ["Centralized LightGBM", "Centralized XGBoost"]:
            b = ("frozen 152", "central", c, 0)
            ci = Boot.ci(dist[a] - dist[b])
            PR.append({"task": "T4", "input_set": "frozen 152 (age quintile)", "comparison": f"{m} - {c}", "partition_seed": "age",
                       "delta": auc(preds[a]) - auc(preds[b]), "ci": ci, "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)})
        b = ("frozen 152", "IID", m, 1)
        ci = Boot.ci(dist[a] - np.mean([dist[("frozen 152", "IID", m, s)] for s in SEEDS], axis=0))
        PR.append({"task": "T4", "input_set": "frozen 152 (age quintile)", "comparison": f"{m}: age quintile - IID (mean of 3 seeds)", "partition_seed": "age",
                   "delta": auc(preds[a]) - float(np.mean([auc(preds[("frozen 152", "IID", m, s)]) for s in SEEDS])), "ci": ci,
                   "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)})
    for c in ["FedAvg", "FedXgbBagging"]:
        a, b = ("frozen 152", "age quintile", "TreeAvg", 0), ("frozen 152", "age quintile", c, 0)
        ci = Boot.ci(dist[a] - dist[b])
        PR.append({"task": "T4", "input_set": "frozen 152 (age quintile)", "comparison": f"TreeAvg - {c}", "partition_seed": "age",
                   "delta": auc(preds[a]) - auc(preds[b]), "ci": ci, "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)})
    pd.DataFrame(PR).to_csv(o("T2_T3_T4_paired_differences.csv"), index=False)
    pd.DataFrame(age_desc).to_csv(o("T4_age_quintile_clients.csv"), index=False)
    pd.DataFrame(eta_rows).to_csv(o("T2_treeavg_eta_selection.csv"), index=False)

    # ---------------- T5 communication
    rdf = pd.DataFrame(rounds).sort_values(["input_set", "partition", "method", "partition_seed", "round"])
    grp = ["input_set", "partition", "method", "partition_seed"]
    rdf["round_mb"] = (rdf.upload_bytes + rdf.download_bytes) / 1e6
    rdf["cum_upload_mb"] = rdf.groupby(grp).upload_bytes.cumsum() / 1e6
    rdf["cum_download_mb"] = rdf.groupby(grp).download_bytes.cumsum() / 1e6
    rdf["cum_total_mb"] = rdf.cum_upload_mb + rdf.cum_download_mb
    rdf.to_csv(o("T5_per_round_communication.csv"), index=False)
    crow = []
    for k, g in rdf.groupby(grp):
        within = g[g.cum_total_mb <= BUDGET + 1e-12]
        br = best_round[(k[0], k[1], k[2], k[3])]
        crow.append({**dict(zip(grp, k)), "rounds_run": int(g["round"].max()), "selected_round": br,
                     "upload_bytes_per_round_mean": float(g.upload_bytes.mean()),
                     "download_bytes_per_round_first": int(g.download_bytes.iloc[0]), "download_bytes_per_round_last": int(g.download_bytes.iloc[-1]),
                     "total_upload_mb": g.upload_bytes.sum() / 1e6, "total_download_mb": g.download_bytes.sum() / 1e6, "total_mb": g.round_mb.sum(),
                     "mb_to_selected_round": float(g[g["round"] <= br].round_mb.sum()),
                     "rounds_within_2.16MB": len(within),
                     "val_auroc_last_round_within_budget": float(within.val_auroc.iloc[-1]) if len(within) else None,
                     "best_val_auroc_within_budget": float(within.val_auroc.max()) if len(within) else None,
                     "best_val_auroc_full_run": float(g.val_auroc.max())})
    cdf = pd.DataFrame(crow)
    cdf.to_csv(o("T5_communication_summary.csv"), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"FedAvg": "#2a6fdb", "FedXgbBagging": "#d9822b", "TreeAvg": "#1a9e5c"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, sname in zip(axes, SETS):
        for m, col in colors.items():
            for s in SEEDS:
                g = rdf[(rdf.input_set == sname) & (rdf.partition == "IID") & (rdf.method == m) & (rdf.partition_seed == s)]
                ax.plot(g.cum_total_mb, g.val_auroc, color=col, lw=1.6 if s == 1 else 0.8, alpha=1 if s == 1 else 0.45,
                        label=m if s == 1 else None, marker="o" if s == 1 else None, ms=2.5)
        ax.axvline(BUDGET, color="0.4", ls="--", lw=0.9)
        ax.set_xscale("log")
        ax.set_title(f"Inputs: {sname}", fontsize=10)
        ax.set_xlabel("Cumulative communication, MB (log scale)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Validation AUROC")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.text(0.01, 0.01, "Bold: partition seed 1; faint: seeds 2 and 3. 5 IID clients. Dashed line: 2.16 MB.", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(o("T5_communication_curves.png"), dpi=300)

    (o("T2_T5_meta.json")).write_text(json.dumps({
        "seeds": {"split": C.SPLIT_SEED, "partition": SEEDS, "bootstrap": F.BOOT_SEED, "n_boot": F.N_BOOT},
        "input_sets": {k: v for k, v in SETS.items()}, "eta_grid": ETA_GRID, "budget_mb": BUDGET,
        "vary_seed": F.VARY_SEED, "runtime_sec": round(time.time() - t0)}, indent=1))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(chk).to_string()); print(pd.DataFrame(eta_rows).to_string()); print(pd.DataFrame(summ).round(4).to_string())
    for r in PR:
        print(f"{r['task']} {r['input_set'][:22]:22s} {r['comparison'][:55]:55s} {str(r['partition_seed']):7s} {r['delta']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]")
    print(cdf.round(4).to_string()); print(pd.DataFrame(age_desc).to_string())


if __name__ == "__main__":
    main()
