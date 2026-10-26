# REB25-1587_MOD1 — research data module; no PII in logs
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mood_monitoring_agent.model import MomentRiskPipeline

DEFAULT_EMBED_CHUNK_SIZE = 64


@torch.no_grad()
def embed_all_windows(
    pipeline: MomentRiskPipeline,
    windows: torch.Tensor,
    chunk_size: int = DEFAULT_EMBED_CHUNK_SIZE,
    input_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Precompute frozen-MOMENT embeddings once for every window (RESEARCH Pattern 1).

    windows: [n_windows, 1, 512] float32. Returns a detached [n_windows, 512] tensor.
    Never call this inside a per-epoch training loop — the backbone is frozen and
    deterministic, so embeddings are computed exactly once and cached (DEPR-03).

    input_masks: optional [n_windows, 512] tensor of 1s (real) / 0s (padding) —
    lets a caller mark zero-padded positions in a short (< 512 real rows)
    window as invalid so MOMENT ignores them, instead of every position
    being treated as real. Added for train_hrv_phq9_moment_no_drop.py, which
    pads short participants to 512 rather than excluding them outright.
    Default None preserves the prior all-ones behavior for every existing
    caller exactly.
    """
    n_windows = windows.shape[0]
    seq_len = windows.shape[-1]
    device = pipeline.device
    backbone = pipeline.backbone

    if not hasattr(backbone, "embed"):
        raise RuntimeError("Backbone has no .embed(); load MOMENT with task_name=\"embedding\"")
    if input_masks is not None:
        assert input_masks.shape == (n_windows, seq_len), (
            f"input_masks shape {tuple(input_masks.shape)} != ({n_windows}, {seq_len})"
        )

    chunks: list[torch.Tensor] = []
    for start in range(0, n_windows, chunk_size):
        end = min(start + chunk_size, n_windows)
        batch = windows[start:end].to(device)
        if input_masks is not None:
            input_mask = input_masks[start:end].to(device)
        else:
            input_mask = torch.ones(batch.shape[0], seq_len, device=device)
        out = backbone.embed(x_enc=batch, input_mask=input_mask)
        embeddings = out.embeddings if hasattr(out, "embeddings") else out
        chunks.append(embeddings.detach())

    result = torch.cat(chunks, dim=0)
    assert result.shape == (n_windows, result.shape[-1]), (
        f"Expected [{n_windows}, d_model], got {tuple(result.shape)}"
    )
    assert not result.requires_grad, "Embedding cache must be detached (requires_grad False)"
    return result


def compute_pos_weight(train_labels: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Compute BCEWithLogitsLoss pos_weight from realized WINDOW-level label counts.

    Never derive this from subject-level counts (23/32) — RESEARCH Pitfall 1 shows
    the window-level class ratio diverges materially from the subject-level ratio
    because control subjects were recorded systematically longer.
    """
    labels = train_labels if isinstance(train_labels, torch.Tensor) else torch.as_tensor(train_labels)
    n_neg = int((labels == 0).sum().item())
    n_pos = int((labels == 1).sum().item())
    if n_pos == 0:
        raise ValueError(
            "compute_pos_weight: no positive (condition) windows in the training "
            "labels — cannot compute a meaningful pos_weight. Check the split."
        )
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def train_adapter(
    pipeline: MomentRiskPipeline,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-3,
) -> list[float]:
    """Fit only MomentAdapter + DepressionRiskHead on cached embeddings (DEPR-04).

    The backbone MUST remain frozen throughout: this function asserts every
    backbone parameter has requires_grad=False before building the optimizer and
    never sets any backbone parameter to trainable. Loss is a numerically-stable
    weighted logit-space BCE (RESEARCH Pitfall 3) computed via
    `pipeline.risk_head.net[:-1]` to bypass the head's trailing Sigmoid.
    """
    for param in pipeline.backbone.parameters():
        if param.requires_grad:
            raise RuntimeError(
                "train_adapter: a backbone parameter has requires_grad=True — "
                "refusing to train. The frozen-backbone invariant (DEPR-04) must "
                "hold before optimizing (never set backbone params trainable)."
            )
    pipeline.backbone.eval()
    pipeline.adapter.train()
    pipeline.risk_head.train()

    trainable_params = list(pipeline.adapter.parameters()) + list(pipeline.risk_head.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    labels = train_labels if isinstance(train_labels, torch.Tensor) else torch.as_tensor(train_labels)
    labels = labels.to(train_embeddings.device).float()
    pos_weight = compute_pos_weight(labels).to(train_embeddings.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    loss_history: list[float] = []
    for _ in range(epochs):
        optimizer.zero_grad()
        adapted = pipeline.adapter(train_embeddings)
        logits = pipeline.risk_head.net[:-1](adapted).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    return loss_history


def save_weights_json(pipeline: MomentRiskPipeline, output_path: str | Path) -> None:
    """Serialize adapter+head weights via the existing JSON FL schema (DEPR-08).

    NEVER use a pytorch-checkpoint or Python-object-serialization save path for
    this checkpoint (threat T-06-02) — always round-trip through
    get_adapter_weights() -> json.dump, matching the on-disk schema
    {"adapter": {...}, "risk_head": {...}} already consumed by
    fl_aggregated_weights.json / load_adapter_weights().
    """
    weights = pipeline.get_adapter_weights()
    path = Path(output_path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(weights, fh)


def verify_roundtrip(pipeline: MomentRiskPipeline, sample_embeddings: torch.Tensor) -> None:
    """Assert the JSON round-trip reproduces an identical forward-pass output.

    Captures adapter+head output on sample_embeddings, serializes via
    json.dumps(get_adapter_weights())/json.loads, reloads via
    load_adapter_weights(), and asserts torch.allclose pre/post.
    """
    pipeline.adapter.eval()
    pipeline.risk_head.eval()
    with torch.no_grad():
        pre_output = pipeline.risk_head(pipeline.adapter(sample_embeddings)).clone()

    serialized = json.loads(json.dumps(pipeline.get_adapter_weights()))
    pipeline.load_adapter_weights(serialized)

    with torch.no_grad():
        post_output = pipeline.risk_head(pipeline.adapter(sample_embeddings))

    if not torch.allclose(pre_output, post_output):
        raise AssertionError(
            "verify_roundtrip: adapter+head forward-pass output changed after the "
            "JSON round-trip through get_adapter_weights()/load_adapter_weights() — "
            "the weight checkpoint is not lossless (DEPR-08)."
        )
