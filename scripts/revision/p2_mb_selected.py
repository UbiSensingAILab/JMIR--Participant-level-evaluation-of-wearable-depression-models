#!/usr/bin/env python3
"""JMIR revision 2.4 add-on: cumulative MB up to each method's validation-selected
round, from P2_4_per_round_communication.csv and P2_1_per_seed_auroc.csv
(both written by p2_fl_rerun.py). Output: P2_4_mb_at_selected_round.csv."""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "results" / "revision"
r = pd.read_csv(OUT / "P2_4_per_round_communication.csv")
t = pd.read_csv(OUT / "P2_1_per_seed_auroc.csv")
rows = []
for _, x in t[t.method.isin(["FedAvg", "FedXgbBagging", "TreeAvg"])].iterrows():
    g = r[(r.method == x.method) & (r.partition_seed == int(x.partition_seed))]
    row = g[g["round"] == int(x.best_round)].iloc[0]
    rows.append({"method": x.method, "partition_seed": int(x.partition_seed), "selected_round": int(x.best_round),
                 "cum_upload_mb_at_selected": round(row.cum_upload_mb, 6), "cum_download_mb_at_selected": round(row.cum_download_mb, 6),
                 "cum_total_mb_at_selected": round(row.cum_total_mb, 6), "total_mb_full_run": round(g.round_mb.sum(), 6)})
pd.DataFrame(rows).to_csv(OUT / "P2_4_mb_at_selected_round.csv", index=False)
print(pd.DataFrame(rows).groupby("method")[["cum_total_mb_at_selected", "total_mb_full_run"]].mean())
