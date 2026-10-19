"""AE-native adapter used by the local MNIST stream smoke run."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class FullyConnectedAE(nn.Module):
    """Agreed small architecture: 784-256-64-256-784."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 64), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 784), nn.Sigmoid())

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x.reshape(x.shape[0], 784))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode(x))


def _flat_float(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    tensor = torch.as_tensor(x)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.reshape(tensor.shape[0], 784).float()
    if tensor.numel() and float(tensor.max()) > 1.0:
        tensor = tensor / 255.0
    return tensor


class AEAdapter:
    """Minimal route adapter; labels are intentionally absent from training."""

    name = "ae_fc_784_256_64_256_784"

    def __init__(self, seed: int = 1, learning_rate: float = 1e-3, device: str = "cpu") -> None:
        torch.manual_seed(seed)
        self.seed = seed
        self.device = torch.device(device)
        self.model = FullyConnectedAE().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

    def train_stream_batch(self, x_unlabelled: torch.Tensor | np.ndarray, *, epochs: int = 1, batch_size: int = 64, stage_seed: int = 0) -> list[dict[str, Any]]:
        """Update on pixel tensors only and return compact optimizer history."""
        x_stage = _flat_float(x_unlabelled).to(self.device)
        history = []
        for epoch in range(epochs):
            self.model.train()
            generator = torch.Generator().manual_seed(self.seed + stage_seed + epoch)
            permutation = torch.randperm(len(x_stage), generator=generator)
            losses = []
            for start in range(0, len(permutation), batch_size):
                x = x_stage[permutation[start : start + batch_size]]
                self.optimizer.zero_grad(set_to_none=True)
                loss = F.mse_loss(self.model(x), x)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss.detach()))
            history.append({
                "epoch": epoch + 1,
                "samples": len(x_stage),
                "optimizer_steps": len(losses),
                "mean_batch_mse": float(np.mean(losses)),
            })
        return history

    def encode(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            return self.model.encode(_flat_float(x).to(self.device))

    def native_metrics(self, x_old: torch.Tensor | np.ndarray, labels_for_eval: Any = None, baseline_latent: np.ndarray | None = None) -> dict[str, np.ndarray | float]:
        """Return AE-native metrics; ``labels_for_eval`` is never used for inference."""
        del labels_for_eval
        x = _flat_float(x_old).to(self.device)
        self.model.eval()
        with torch.inference_mode():
            latent = self.model.encode(x)
            reconstruction_mse = F.mse_loss(self.model.decoder(latent), x, reduction="none").mean(dim=1)
        latent_np = latent.cpu().numpy()
        error_np = reconstruction_mse.cpu().numpy()
        result: dict[str, np.ndarray | float] = {
            "latent": latent_np,
            "reconstruction_mse": error_np,
            "mean_reconstruction_mse": float(error_np.mean()),
        }
        if baseline_latent is not None:
            cosine = np.sum(baseline_latent * latent_np, axis=1) / (
                np.linalg.norm(baseline_latent, axis=1) * np.linalg.norm(latent_np, axis=1) + 1e-8
            )
            drift = 1.0 - cosine
            result["latent_drift_1_minus_cosine"] = drift
            result["latent_l2_drift"] = np.linalg.norm(latent_np - baseline_latent, axis=1)
        return result

    def event_metadata(self) -> dict[str, Any]:
        return {
            "route": self.name,
            "architecture": "784-256-64-256-784",
            "training_labels_entered_model": False,
            "ctm_attached": False,
        }
