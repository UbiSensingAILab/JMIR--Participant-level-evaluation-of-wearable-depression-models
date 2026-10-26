#!/usr/bin/env python3
"""Small, FedAvg-compatible MLP for the psyche-d-fedavg-vs-fedtree session.
NEW module -- no broad architecture hunting (per task instruction): ONE fixed,
sensible architecture used identically for the centralized arm (Step 1) and
the FedAvg arm (Steps 2/3), as required.

Architecture: 152 -> 64 -> 32 -> 1, ReLU, dropout(0.2), sigmoid output.
~11.7K parameters -- small enough that transmitting the full parameter vector
every round (literal FedAvg, not just adapter weights) is trivially cheap.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

INPUT_DIM = 152
HIDDEN1 = 64
HIDDEN2 = 32
DROPOUT = 0.2
SEED = 20260928  # common-init seed for the FedAvg global model (Step 3)


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, HIDDEN1), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN1, HIDDEN2), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def new_model(seed: int = SEED) -> SmallMLP:
    """Common randomly-initialized model -- used as the FedAvg round-0 global
    model distributed to all clients (never a centrally-pretrained model,
    per the task's explicit instruction)."""
    torch.manual_seed(seed)
    return SmallMLP()


def get_state(model: nn.Module) -> dict:
    return copy.deepcopy(model.state_dict())


def set_state(model: nn.Module, state: dict) -> None:
    model.load_state_dict(state)


def fedavg_aggregate(client_states: list[dict], client_n: list[int]) -> dict:
    """Standard FedAvg: weighted average of client state_dicts, weighted by
    each client's local sample count (McMahan et al. 2017, Algorithm 1)."""
    total_n = sum(client_n)
    agg = copy.deepcopy(client_states[0])
    for key in agg:
        agg[key] = torch.zeros_like(agg[key], dtype=torch.float32)
        for state, n in zip(client_states, client_n):
            agg[key] += state[key].float() * (n / total_n)
    return agg


def train_local(
    model: nn.Module, X: np.ndarray, y: np.ndarray, epochs: int, lr: float = 1e-3,
    batch_size: int = 64, seed: int = 0, class_weight: bool = True,
) -> nn.Module:
    """Trains `model` IN PLACE for `epochs` local epochs on (X, y). Used both
    for centralized training (Step 1, epochs=many) and per-client local
    training within one FedAvg round (Steps 2/3, epochs=local_epochs)."""
    g = torch.Generator().manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    if class_weight:
        pos = float((y == 1).sum())
        neg = float((y == 0).sum())
        pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    else:
        pos_weight = None
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(Xt)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            opt.zero_grad()
            out = model(Xt[idx])
            loss = loss_fn(out, yt[idx])
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    logits = model(Xt)
    return torch.sigmoid(logits).numpy()
