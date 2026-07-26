from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.ctm import ContinuousThoughtMachine


@dataclass
class ProbeResult:
    predictions: np.ndarray
    certainties: np.ndarray
    sync_out: np.ndarray
    sync_action: np.ndarray
    queries: np.ndarray
    pre_activations: np.ndarray
    post_activations: np.ndarray
    attention: np.ndarray
    feature_shape: tuple[int, int]

    @property
    def probabilities(self) -> np.ndarray:
        logits = self.predictions - self.predictions.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return exp / exp.sum(axis=1, keepdims=True)


def _get_arg(args: Any, current: str, legacy: str | None = None, default: Any = None) -> Any:
    if hasattr(args, current):
        return getattr(args, current)
    if legacy and hasattr(args, legacy):
        return getattr(args, legacy)
    return default


def load_checkpoint_model(checkpoint_path: str | Path, device: str) -> tuple[ContinuousThoughtMachine, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint["args"]

    backbone_type = _get_arg(args, "backbone_type")
    if backbone_type is None:
        scales = _get_arg(args, "resnet_feature_scales", default=[4])
        backbone_type = f"{_get_arg(args, 'resnet_type')}-{scales[-1]}"

    neuron_select_type = _get_arg(args, "neuron_select_type", default="first-last")
    model = ContinuousThoughtMachine(
        iterations=_get_arg(args, "iterations"),
        d_model=_get_arg(args, "d_model"),
        d_input=_get_arg(args, "d_input"),
        heads=_get_arg(args, "heads"),
        n_synch_out=_get_arg(args, "n_synch_out"),
        n_synch_action=_get_arg(args, "n_synch_action"),
        synapse_depth=_get_arg(args, "synapse_depth"),
        memory_length=_get_arg(args, "memory_length"),
        deep_nlms=_get_arg(args, "deep_memory", "deep_nlms", True),
        memory_hidden_dims=_get_arg(args, "memory_hidden_dims"),
        do_layernorm_nlm=_get_arg(args, "do_normalisation", "do_layernorm_nlm", False),
        backbone_type=backbone_type,
        positional_embedding_type=_get_arg(args, "positional_embedding_type", default="none"),
        out_dims=_get_arg(args, "out_dims"),
        prediction_reshaper=[-1],
        dropout=0,
        dropout_nlm=0,
        neuron_select_type=neuron_select_type,
        n_random_pairing_self=_get_arg(args, "n_random_pairing_self", default=0),
    ).to(device)
    load_result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch. Missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model.eval()
    return model, args


def load_hub_model(model_id: str, device: str, cache_dir: str | Path | None = None) -> ContinuousThoughtMachine:
    model = ContinuousThoughtMachine.from_pretrained(
        model_id,
        map_location=device,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    model.eval()
    return model


def run_probe(model: ContinuousThoughtMachine, inputs: torch.Tensor) -> ProbeResult:
    if inputs.shape[0] != 1:
        raise ValueError("The detailed probe currently requires batch_size=1.")
    with torch.inference_mode():
        outputs = model(inputs, track=True)
    predictions, certainties, syncs, pre, post, attention = outputs
    sync_out, sync_action = syncs
    sync_action_for_query = torch.from_numpy(np.asarray(sync_action[:, 0], dtype=np.float32)).to(inputs.device)
    queries = model.q_proj(sync_action_for_query).detach().cpu().numpy().astype(np.float32)
    return ProbeResult(
        predictions=predictions[0].detach().cpu().numpy().T.astype(np.float32),
        certainties=certainties[0, 1].detach().cpu().numpy().astype(np.float32),
        sync_out=np.asarray(sync_out[:, 0], dtype=np.float32),
        sync_action=np.asarray(sync_action[:, 0], dtype=np.float32),
        queries=queries,
        pre_activations=np.asarray(pre[:, 0], dtype=np.float32),
        post_activations=np.asarray(post[:, 0], dtype=np.float32),
        attention=np.asarray(attention[:, 0], dtype=np.float32),
        feature_shape=tuple(int(value) for value in model.kv_features.shape[-2:]),
    )


def paper_full_sync(post_activations: np.ndarray, tick: int) -> np.ndarray:
    """Return the unweighted cumulative activation Gram matrix at ``tick``.

    This is a useful full-neuron analogue of synchronization, but it is not a
    learned-decay CTM representation: the official model only learns decay
    parameters for the neuron pairs selected for S_action and S_out.
    """
    if tick < 1 or tick > post_activations.shape[0]:
        raise ValueError(f"tick must be in [1, {post_activations.shape[0]}], got {tick}")
    z = torch.from_numpy(post_activations[:tick]).float()
    return (z.T @ z).numpy()


def neuron_activity_statistics(post_activations: np.ndarray) -> dict[str, np.ndarray]:
    """Score continuous-valued neurons over the complete observed trajectory."""
    z = np.asarray(post_activations, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError("post_activations must have shape (ticks, neurons)")
    rms = np.sqrt(np.mean(np.square(z), axis=0))
    std = np.std(z, axis=0)
    per_tick_scale = np.maximum(np.max(np.abs(z), axis=1, keepdims=True), 1e-12)
    active_fraction = np.mean(np.abs(z) >= 0.1 * per_tick_scale, axis=0)
    return {
        "rms": rms.astype(np.float32),
        "std": std.astype(np.float32),
        "active_fraction": active_fraction.astype(np.float32),
    }


def select_active_neurons(post_activations: np.ndarray, top_k: int = 512) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Select a fixed, traceable neuron subset by trajectory RMS activity."""
    stats = neuron_activity_statistics(post_activations)
    n_neurons = post_activations.shape[1]
    top_k = min(max(int(top_k), 1), n_neurons)
    indices = np.argsort(stats["rms"], kind="stable")[-top_k:][::-1]
    return indices.astype(np.int64), stats


def active_full_sync_history(post_activations: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return exact cumulative Gram matrices for a fixed neuron subset."""
    z = np.asarray(post_activations, dtype=np.float32)[:, np.asarray(indices, dtype=np.int64)]
    products = z[:, :, None] * z[:, None, :]
    return np.cumsum(products, axis=0, dtype=np.float32)


def normalized_sync_matrix(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize a Gram-like matrix to signed pairwise cosine values."""
    diagonal = np.maximum(np.diag(matrix), 0.0)
    denominator = np.sqrt(diagonal[:, None] * diagonal[None, :])
    result = matrix / np.maximum(denominator, eps)
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def synchronization_pair_indices(model: ContinuousThoughtMachine, synch_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Map each exact S_action/S_out value back to its original neuron pair."""
    if synch_type not in {"action", "out"}:
        raise ValueError("synch_type must be 'action' or 'out'")
    left = getattr(model, f"{synch_type}_neuron_indices_left", None)
    right = getattr(model, f"{synch_type}_neuron_indices_right", None)
    if left is None or right is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    left = left.detach().cpu().numpy().astype(np.int64)
    right = right.detach().cpu().numpy().astype(np.int64)
    if model.neuron_select_type in ("first-last", "random"):
        rows, cols = np.triu_indices(len(left))
        return left[rows], right[cols]
    return left, right


def downsample_square_matrix(matrix: np.ndarray, size: int = 256) -> np.ndarray:
    """Block-average a square matrix for compact numeric storage and plotting."""
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("matrix must be square")
    if n % size:
        edges = np.linspace(0, n, size + 1, dtype=int)
        return np.asarray(
            [
                [matrix[edges[i] : edges[i + 1], edges[j] : edges[j + 1]].mean()
                 for j in range(size)]
                for i in range(size)
            ],
            dtype=np.float32,
        )
    block = n // size
    return matrix.reshape(size, block, size, block).mean(axis=(1, 3)).astype(np.float32)


def downsample_full_sync(post_activations: np.ndarray, tick: int, size: int = 256) -> np.ndarray:
    """Compute the exact block mean of S^t without first materializing the D x D matrix."""
    if tick < 1 or tick > post_activations.shape[0]:
        raise ValueError(f"tick must be in [1, {post_activations.shape[0]}], got {tick}")
    d_model = post_activations.shape[1]
    if d_model % size:
        return downsample_square_matrix(paper_full_sync(post_activations, tick), size)
    block = d_model // size
    grouped = post_activations[:tick].reshape(tick, size, block).mean(axis=2)
    return (grouped.T @ grouped).astype(np.float32)


def triangular_representation(vector: np.ndarray, n_synch: int) -> np.ndarray:
    expected = n_synch * (n_synch + 1) // 2
    if vector.size != expected:
        raise ValueError(f"Expected {expected} values for n_synch={n_synch}, got {vector.size}")
    result = np.full((n_synch, n_synch), np.nan, dtype=np.float32)
    rows, cols = np.triu_indices(n_synch)
    result[rows, cols] = vector
    return result


def checkpoint_metadata(model: ContinuousThoughtMachine, args: Any) -> dict[str, Any]:
    keys = [
        "iterations",
        "d_model",
        "d_input",
        "heads",
        "n_synch_out",
        "n_synch_action",
        "synapse_depth",
        "memory_length",
        "memory_hidden_dims",
        "neuron_select_type",
        "n_random_pairing_self",
        "backbone_type",
        "out_dims",
    ]
    metadata = {key: getattr(model, key, getattr(args, key, None)) for key in keys}
    metadata["sync_representation_size_out"] = model.synch_representation_size_out
    metadata["sync_representation_size_action"] = model.synch_representation_size_action
    return metadata
