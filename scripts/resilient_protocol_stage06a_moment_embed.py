#!/usr/bin/env python3
# RESILIENT dataset (Zenodo 16755408): IRAS 321104, independent of REB25-1587_MOD1. Research data.
"""Protocol Stage 6 (branch R), part a: frozen MOMENT-1-small embeddings per participant
and per modality, cached for the modelling scripts. Run with .venv311.

Reuses UNMODIFIED from train_resilient_multimodal_binary_phq9.py: build_channel_tensors
(hourly HR mean / hourly Steps sum on [phq_date, +512 h), own-mean gap fill; nightly
Sleep_physio x4 + Sleep_state x4 left-aligned with input_mask = 0 on padding) and
embed_all_groups (MomentRiskPipeline frozen backbone, task "embedding", 512-d output).
Coverage rules from that script (HR/Steps >= 50 % hourly bins, >= 1 real night) are kept;
participants failing them get NaN embeddings (imputed train-fold-only downstream).

Note on branch R: the protocol suggests one 512-sample window per channel and mean+SD
pooling over windows. Here the label is one baseline PHQ-9 and the bounded window is
exactly one 512-hour context, so there is exactly one window per participant per modality
and pooling is the identity. Writes scripts/resilient_protocol_moment_embeds.pkl.
"""
from __future__ import annotations

import pickle
import resource
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import train_resilient_multimodal_binary_phq9 as B  # noqa: E402

OUT = SCRIPT_DIR / "resilient_protocol_moment_embeds.pkl"


def main():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    demo = B.load_demographics()
    uids = B.get_participant_uids()
    ct = B.build_channel_tensors(uids, demo, external_zscore=False)
    uids = [u for u in uids if u in ct["tabular"]]
    pipe = B.MomentRiskPipeline.from_pretrained()
    trainable_backbone = sum(p.numel() for p in pipe.parameters() if p.requires_grad) if hasattr(pipe, "parameters") else None
    emb = B.embed_all_groups(pipe, uids, ct)
    out = {"uids": uids,
           "hr": np.stack([emb["hr"][u] for u in uids]), "steps": np.stack([emb["steps"][u] for u in uids]),
           "sleep_physio": np.stack([emb["sleep_physio"][u] for u in uids]),
           "sleep_state": np.stack([emb["sleep_state"][u] for u in uids]),
           "coverage": {k: {u: bool(v[u]) for u in uids} for k, v in emb["coverage"].items()},
           "note": "frozen MOMENT-1-small, task embedding, 512-d per modality, window [phq_date, +512 h)"}
    OUT.write_bytes(pickle.dumps(out))
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS reports bytes
    print({k: (np.isnan(out[k]).any(axis=1).sum() if isinstance(out[k], np.ndarray) else None) for k in ["hr", "steps", "sleep_physio", "sleep_state"]},
          "trainable_params_reported:", trainable_backbone, f"peak RSS {rss_gb:.2f} GB")


if __name__ == "__main__":
    main()
