#!/usr/bin/env python3
# PSYCHE-D dataset (Zenodo 5085146), CC BY-NC 4.0. Aggregates only in results/.
"""JMIR revision Phase 2, task 2.5: non-IID client partitions of the same
training pool (2825 participants), same fixed validation/test sets, same 3
methods (FedAvg, FedXgbBagging, TreeAvg) and hyperparameters as the IID runs.

Partitions (whole participants only):
  age5       5 clients = quintiles of screener birthyear (static per participant,
             no missing values in the training pool); deterministic, 1 partition.
  dirichlet  label skew: participant label = 1 if any of the participant's
             training periods is a worsening; for each label, client proportions
             ~ Dirichlet(alpha=0.5 x 5), participants of that label shuffled and
             cut at the cumulative proportions. rng seeds 1, 2, 3.
Metrics as 2.1: test AUROC/AUPRC with participant-level CIs (2000 resamples,
rng 42, shared with the IID runs), per partition and mean/SD over Dirichlet seeds.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import p2_fl_core as F
from p2_fl_core import C, OUT, Boot

DIRICHLET_SEEDS = [1, 2, 3]
ALPHA = 0.5


def age_partition(D):
    by = D.train_df.groupby("user_id")["birthyear"].first()
    q = pd.qcut(by.rank(method="first"), 5, labels=False)
    return [set(by.index[q == k]) for k in range(5)], [f"{int(by[q == k].min())}-{int(by[q == k].max())}" for k in range(5)]


def dirichlet_partition(D, seed):
    lab = D.train_df.assign(yy=D.y.loc[D.train_df.index].to_numpy()).groupby("user_id")["yy"].max()
    rng = np.random.default_rng(seed)
    sites = [set() for _ in range(5)]
    for c in (0, 1):
        users = np.array(sorted(lab.index[lab == c]))
        rng.shuffle(users)
        p = rng.dirichlet([ALPHA] * 5)
        cuts = (np.cumsum(p)[:-1] * len(users)).astype(int)
        for k, part in enumerate(np.split(users, cuts)):
            sites[k] |= set(part)
    return sites


def describe(D, sites):
    out = []
    for k, s in enumerate(sites):
        d = D.train_df[D.train_df["user_id"].isin(s)]
        yc = D.yc(d)
        out.append({"client": k, "n_participants": len(s), "n_rows": len(d), "n_pos_rows": int(yc.sum()),
                    "pos_rate": round(float(yc.mean()), 4) if len(yc) else None, "one_class": bool(len(np.unique(yc)) < 2)})
    return out


def main():
    D = F.Data()
    B = Boot(D.test_users, D.y_test)
    parts = {}
    s_age, age_ranges = age_partition(D)
    parts["age quintile"] = s_age
    for s in DIRICHLET_SEEDS:
        parts[f"Dirichlet(0.5) seed {s}"] = dirichlet_partition(D, s)
    desc = []
    for name, sites in parts.items():
        assert set().union(*sites) == D.cache["train_participants"] and sum(len(s) for s in sites) == len(D.cache["train_participants"])
        for row in describe(D, sites):
            desc.append({"partition": name, **row, "birthyear_range": age_ranges[row["client"]] if name == "age quintile" else ""})
    pd.DataFrame(desc).to_csv(OUT / "P2_5_noniid_client_composition.csv", index=False)
    usable = {n: s for n, s in parts.items() if all(len(x) > 0 for x in s)}
    skipped = sorted(set(parts) - set(usable))
    fedavg = dict(zip(usable, F.run_fedavg_batch(list(usable.values()), local_only=False)))
    rows, dists = [], {}
    for name, sites in usable.items():
        cdfs = D.clients(sites)
        runs = {"FedAvg": fedavg[name], "FedXgbBagging": F.run_fedxgb(D, cdfs), "TreeAvg": F.run_treeavg(D, cdfs)}
        for m, r in runs.items():
            met, dists[(m, name)] = B.metric(r["test_pred"])
            rows.append({"partition": name, "method": m, "best_round": r["best_round"], "auroc": round(met["auroc"], 6),
                         "auroc_ci": "%.6f-%.6f" % tuple(met["auroc_ci"]), "auprc": round(met["auprc"], 6),
                         "auprc_ci": "%.6f-%.6f" % tuple(met["auprc_ci"])})
        print(f"{name} done", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "P2_5_noniid_auroc.csv", index=False)
    summ = []
    dnames = [n for n in usable if n.startswith("Dirichlet")]
    for m in ["FedAvg", "FedXgbBagging", "TreeAvg"]:
        a = t[(t.method == m) & t.partition.isin(dnames)].auroc.to_numpy()
        summ.append({"method": m, "dirichlet_n_partitions": len(a), "auroc_mean": round(a.mean(), 4), "auroc_sd_ddof1": round(a.std(ddof=1), 4) if len(a) > 1 else None,
                     "age_quintile_auroc": float(t[(t.method == m) & (t.partition == "age quintile")].auroc.iloc[0])})
    # paired TreeAvg minus comparators within each non-IID partition (same participant draws)
    pr = []
    for name in usable:
        for c in ["FedAvg", "FedXgbBagging"]:
            dd = dists[("TreeAvg", name)] - dists[(c, name)]
            ci = Boot.ci(dd)
            pt = t[(t.partition == name) & (t.method == "TreeAvg")].auroc.iloc[0] - t[(t.partition == name) & (t.method == c)].auroc.iloc[0]
            pr.append({"partition": name, "comparison": f"TreeAvg - {c}", "delta_auroc": round(pt, 6), "ci": "%.6f to %.6f" % tuple(ci),
                       "ci_excludes_0": bool(ci[0] > 0 or ci[1] < 0)})
    pd.DataFrame(summ).to_csv(OUT / "P2_5_noniid_summary.csv", index=False)
    pd.DataFrame(pr).to_csv(OUT / "P2_5_noniid_paired.csv", index=False)
    (OUT / "P2_5_noniid_meta.json").write_text(json.dumps({"dirichlet_seeds": DIRICHLET_SEEDS, "alpha": ALPHA, "bootstrap_seed": F.BOOT_SEED,
                                                           "n_boot": F.N_BOOT, "skipped_partitions_empty_client": skipped}, indent=2))
    print(pd.DataFrame(desc).to_string()); print(t.to_string()); print(pd.DataFrame(summ).to_string()); print(pd.DataFrame(pr).to_string())


if __name__ == "__main__":
    main()
