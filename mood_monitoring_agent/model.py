# REB25-1587_MOD1 — research data module; no PII in logs
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from momentfm.models.moment import MOMENTPipeline
from collections.abc import Sequence

DEFAULT_MODEL_NAME = "AutonLab/MOMENT-1-small"
DEFAULT_BOTTLENECK = 16
DEFAULT_DEVICE = "cpu"


def _to_tensor(value: object, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return torch.tensor(value, dtype=torch.float32, device=device)


def _serialize_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, list]:
    return {key: tensor.detach().cpu().numpy().tolist() for key, tensor in state_dict.items()}


def _deserialize_state_dict(state_dict: dict[str, list], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in state_dict.items()}


class MomentAdapter(nn.Module):
    """Lightweight bottleneck adapter for MOMENT embeddings."""

    def __init__(self, d_model: int = 512, bottleneck: int = DEFAULT_BOTTLENECK) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck, d_model)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.act(self.down(x)))


class DepressionRiskHead(nn.Module):
    """Depression risk head that maps adapted embeddings to a probability."""

    def __init__(self, in_dim: int = 512, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MomentRiskPipeline(nn.Module):
    """Wraps a frozen MOMENT backbone with a trainable adapter and depression risk head."""

    def __init__(
        self,
        backbone: MOMENTPipeline,
        adapter: MomentAdapter | None = None,
        risk_head: DepressionRiskHead | None = None,
        device: str | torch.device = DEFAULT_DEVICE,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.backbone = backbone.to(self.device)
        self.adapter = adapter or MomentAdapter()
        self.risk_head = risk_head or DepressionRiskHead()
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = DEFAULT_MODEL_NAME,
        task_name: str = "embedding",
        device: str | torch.device = DEFAULT_DEVICE,
        bottleneck: int = DEFAULT_BOTTLENECK,
        **model_kwargs,
    ) -> "MomentRiskPipeline":
        device = torch.device(device)
        backbone = MOMENTPipeline.from_pretrained(
            pretrained_model_name_or_path,
            task_name=task_name,
            **model_kwargs,
        )
        backbone.init()
        backbone = backbone.to(device)
        adapter = MomentAdapter(bottleneck=bottleneck)
        risk_head = DepressionRiskHead()
        return cls(backbone=backbone, adapter=adapter, risk_head=risk_head, device=device)

    def _ensure_tensor(self, value: object, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if value is None:
            raise ValueError("Expected a tensor-like value, got None")
        if isinstance(value, torch.Tensor):
            return value.to(self.device).to(dtype)
        tensor = torch.tensor(value, dtype=dtype, device=self.device)
        return tensor

    def embed(
        self,
        *,
        x_enc: object,
        input_mask: object | None = None,
    ) -> torch.Tensor:
        """Run only the frozen backbone, returning its embedding (no adapter/risk head).

        Exposed separately from forward() so callers that need to compare
        multiple adapter/risk-head weight states against the SAME input (e.g.
        a before/after LoRA or before/after FL-round demo) can pay the
        backbone's cost once instead of once per comparison -- the backbone
        is frozen and doesn't change between those comparisons.
        """
        x_enc = self._ensure_tensor(x_enc, dtype=torch.float32)
        input_mask_tensor = None
        if input_mask is not None:
            input_mask_tensor = self._ensure_tensor(input_mask, dtype=torch.float32)
        with torch.no_grad():
            # Prefer a dedicated embed() helper if available on the backbone
            if hasattr(self.backbone, "embed"):
                embeddings = self.backbone.embed(x_enc=x_enc, input_mask=input_mask_tensor)
                # Some backbone embed() helpers return a TimeseriesOutputs-like object
                if hasattr(embeddings, "embeddings"):
                    embeddings = embeddings.embeddings
            else:
                output = self.backbone(x_enc=x_enc, input_mask=input_mask_tensor)
                embeddings = getattr(output, "embeddings", None)
                if embeddings is None:
                    embeddings = getattr(output, "output", None)
            if embeddings is None:
                raise RuntimeError("Backbone did not return embeddings")
        return embeddings

    def forward(
        self,
        *,
        x_enc: object,
        input_mask: object | None = None,
    ) -> torch.Tensor:
        embeddings = self.embed(x_enc=x_enc, input_mask=input_mask)
        adapted = self.adapter(embeddings)
        return self.risk_head(adapted)

    def get_adapter_weights(self) -> dict[str, dict[str, list]]:
        return {
            "adapter": _serialize_state_dict(self.adapter.state_dict()),
            "risk_head": _serialize_state_dict(self.risk_head.state_dict()),
        }

    def load_adapter_weights(self, weights: dict[str, dict[str, list]]) -> None:
        self.adapter.load_state_dict(_deserialize_state_dict(weights["adapter"], self.device))
        self.risk_head.load_state_dict(_deserialize_state_dict(weights["risk_head"], self.device))


def compute_z_score(value: float, mean: float, std: float) -> float:
    safe_std = max(abs(std), 1e-6)
    return (value - mean) / safe_std


def compute_mood_score(value: float, rolling_mean_7d: float, rolling_std_7d: float) -> float:
    z_score = compute_z_score(value, rolling_mean_7d, rolling_std_7d)
    mood_score = 50.0 + z_score * 15.0
    return float(np.clip(mood_score, 0.0, 100.0))


def compute_alert_flag(z_score: float) -> bool:
    return z_score < -1.5


def compute_trend(mood_scores: Sequence[float], timestamps: Sequence[object]) -> str:
    if len(mood_scores) != len(timestamps) or len(mood_scores) == 0:
        return "insufficient_data"

    timestamps_secs = np.array(
        [float(ts.timestamp()) if hasattr(ts, "timestamp") else float(ts) for ts in timestamps],
        dtype=np.float64,
    )
    mood_scores_arr = np.asarray(mood_scores, dtype=np.float32)
    sort_idx = np.argsort(timestamps_secs)
    timestamps_secs = timestamps_secs[sort_idx]
    mood_scores_arr = mood_scores_arr[sort_idx]

    end = timestamps_secs[-1]
    current_start = end - 86400.0
    previous_start = current_start - 86400.0

    current_mask = timestamps_secs > current_start
    previous_mask = (timestamps_secs > previous_start) & (timestamps_secs <= current_start)

    current_scores = mood_scores_arr[current_mask]
    previous_scores = mood_scores_arr[previous_mask]

    if current_scores.size == 0 or previous_scores.size == 0:
        return "insufficient_data"

    current_med = float(np.median(current_scores))
    previous_med = float(np.median(previous_scores))
    diff = current_med - previous_med

    if abs(diff) < 1.0:
        return "stable"
    return "improving" if diff > 0.0 else "declining"
