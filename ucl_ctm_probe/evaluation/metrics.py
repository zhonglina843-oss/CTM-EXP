"""Descriptive fixed-sample drift metrics; no training or causal claims."""

from __future__ import annotations

from typing import Any

import numpy as np


def _matrix(name: str, value: Any, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim or any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be non-empty with {ndim} dimensions; got {array.shape}")
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite real numeric values")
    return array.astype(np.float64, copy=False)


def _cosine_distance(left: np.ndarray, right: np.ndarray, name: str) -> np.ndarray:
    if left.shape != right.shape:
        raise ValueError(f"{name} shapes must match; got {left.shape} and {right.shape}")
    left_norm = np.linalg.norm(left, axis=-1)
    right_norm = np.linalg.norm(right, axis=-1)
    if np.any(left_norm == 0) or np.any(right_norm == 0):
        raise ValueError(f"{name} cosine distance is undefined for zero-norm vectors")
    cosine = np.sum(left * right, axis=-1) / (left_norm * right_norm)
    return 1.0 - np.clip(cosine, -1.0, 1.0)


def _summary(per_sample: np.ndarray) -> dict[str, Any]:
    return {
        "mean": float(np.mean(per_sample)),
        "median": float(np.median(per_sample)),
        "per_sample": per_sample.tolist(),
    }


def feature_drift(baseline: Any, current: Any) -> dict[str, Any]:
    """Per-sample ``1 - cosine`` between aligned ``[B, D]`` features."""

    left = _matrix("baseline features", baseline, 2)
    right = _matrix("current features", current, 2)
    if left.shape != right.shape:
        raise ValueError(f"feature shapes must match; got {left.shape} and {right.shape}")
    return _summary(_cosine_distance(left, right, "feature drift"))


def _cosine_similarity_matrix(features: np.ndarray, name: str) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} cosine similarity is undefined for zero-norm feature rows")
    normalized = features / norms
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def similarity_structure_drift(
    baseline: Any, current: Any, k: int = 10
) -> dict[str, Any]:
    """Compare fixed-set cosine relations and stable top-k neighbourhoods.

    Returns upper-triangle mean absolute similarity change, its Pearson
    correlation when defined, and mean per-sample k-neighbour overlap.
    """

    left = _matrix("baseline features", baseline, 2)
    right = _matrix("current features", current, 2)
    if left.shape != right.shape:
        raise ValueError(f"feature shapes must match; got {left.shape} and {right.shape}")
    count = left.shape[0]
    if count < 2:
        raise ValueError("similarity-structure drift requires at least 2 fixed samples")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer")

    sim0 = _cosine_similarity_matrix(left, "baseline features")
    sim1 = _cosine_similarity_matrix(right, "current features")
    upper = np.triu_indices(count, k=1)
    tri0, tri1 = sim0[upper], sim1[upper]
    correlation: float | None = None
    if tri0.size >= 2 and np.std(tri0) > 0 and np.std(tri1) > 0:
        correlation = float(np.corrcoef(tri0, tri1)[0, 1])

    neighbour_count = min(int(k), count - 1)
    np.fill_diagonal(sim0, -np.inf)
    np.fill_diagonal(sim1, -np.inf)
    # Stable sorting gives deterministic tie handling by original sample order.
    neighbours0 = np.argsort(-sim0, axis=1, kind="stable")[:, :neighbour_count]
    neighbours1 = np.argsort(-sim1, axis=1, kind="stable")[:, :neighbour_count]
    overlap = np.array(
        [len(set(a.tolist()).intersection(b.tolist())) / neighbour_count
         for a, b in zip(neighbours0, neighbours1)],
        dtype=np.float64,
    )
    return {
        "upper_triangle_mean_absolute_change": float(np.mean(np.abs(tri0 - tri1))),
        "upper_triangle_pearson_correlation": correlation,
        "upper_triangle_one_minus_correlation": (
            None if correlation is None else 1.0 - correlation
        ),
        "knn_k": neighbour_count,
        "knn_overlap_mean": float(np.mean(overlap)),
        "knn_overlap_per_sample": overlap.tolist(),
        "sample_count": count,
    }


def _normalized_tick_concat(history: np.ndarray, name: str) -> np.ndarray:
    """Unit-normalize each [tick,sample,dim] vector before tick concatenation."""

    norms = np.linalg.norm(history, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError(f"{name} contains a zero-norm tick vector")
    normalized = history / norms
    # [T,B,D] -> [B,T*D], retaining every selected tick in its stored order.
    return normalized.transpose(1, 0, 2).reshape(history.shape[1], -1)


def trajectory_drift(baseline: Any, current: Any, tick_ids: Any) -> dict[str, Any]:
    """Compare ``[T, B, D]`` activity at matched ticks.

    Primary summary: cosine drift after unit-normalizing each selected tick and
    concatenating ticks per sample. Sensitivity summary: mean same-tick cosine
    drift, with the full checkpoint-tick curve retained.
    """

    left = _matrix("baseline z_history", baseline, 3)
    right = _matrix("current z_history", current, 3)
    if left.shape != right.shape:
        raise ValueError(f"z_history shapes must match; got {left.shape} and {right.shape}")
    ticks = np.asarray(tick_ids)
    if ticks.ndim != 1 or ticks.size != left.shape[0] or ticks.dtype.kind not in "iu":
        raise ValueError(f"tick_ids must be integer [T] for z_history; got {ticks.shape}")
    if np.any(ticks <= 0) or np.any(np.diff(ticks) <= 0):
        raise ValueError("tick_ids must be strictly increasing positive integers")

    per_tick_per_sample = _cosine_distance(left, right, "trajectory drift")
    pooled0 = _normalized_tick_concat(left, "baseline z_history")
    pooled1 = _normalized_tick_concat(right, "current z_history")
    pooled_per_sample = _cosine_distance(pooled0, pooled1, "pooled trajectory drift")
    return {
        "pooling": "unit-normalize each selected tick, concatenate, then cosine distance",
        "mean": float(np.mean(pooled_per_sample)),
        "per_sample": pooled_per_sample.tolist(),
        "mean_over_ticks_sensitivity": float(np.mean(per_tick_per_sample)),
        "per_tick_mean_sensitivity": np.mean(per_tick_per_sample, axis=1).tolist(),
        "tick_ids": ticks.tolist(),
    }


def synchronization_drift(
    baseline: Any, current: Any, tick_ids: Any
) -> dict[str, Any]:
    """Compare actual sync tensor outputs without changing their stored shape.

    Vector representation: same-tick cosine distance on ``[T,B,D]``.
    Matrix representation: per-sample, per-tick baseline-relative normalized
    Frobenius distance on ``[T,B,R,C]``, ``||A-B||_F / max(||A||_F, eps)``.
    """

    left = np.asarray(baseline)
    right = np.asarray(current)
    if left.ndim < 3 or any(size == 0 for size in left.shape):
        raise ValueError(f"baseline sync tensor must be non-empty [T,B,...], got {left.shape}")
    if right.ndim < 3 or any(size == 0 for size in right.shape):
        raise ValueError(f"current sync tensor must be non-empty [T,B,...], got {right.shape}")
    if left.shape != right.shape:
        raise ValueError(f"sync_history shapes must match; got {left.shape} and {right.shape}")
    if left.dtype.kind not in "fiu" or not np.all(np.isfinite(left)):
        raise ValueError("baseline sync tensor must contain finite real numeric values")
    if right.dtype.kind not in "fiu" or not np.all(np.isfinite(right)):
        raise ValueError("current sync tensor must contain finite real numeric values")
    ticks = np.asarray(tick_ids)
    if ticks.ndim != 1 or ticks.size != left.shape[0] or ticks.dtype.kind not in "iu":
        raise ValueError(f"tick_ids must be integer [T] for sync_history; got {ticks.shape}")
    if np.any(ticks <= 0) or np.any(np.diff(ticks) <= 0):
        raise ValueError("tick_ids must be strictly increasing positive integers")

    if left.ndim == 3:
        distances = _cosine_distance(
            left.astype(np.float64, copy=False),
            right.astype(np.float64, copy=False),
            "synchronization-vector drift",
        )
        metric = "same-tick cosine distance; vector [T,B,D]"
        representation = "vector"
    else:
        left64 = left.astype(np.float64, copy=False)
        right64 = right.astype(np.float64, copy=False)
        trailing_axes = tuple(range(2, left.ndim))
        delta_norm = np.linalg.norm(left64 - right64, axis=trailing_axes)
        baseline_norm = np.linalg.norm(left64, axis=trailing_axes)
        distances = delta_norm / np.maximum(baseline_norm, np.finfo(np.float64).eps)
        representation = "matrix" if left.ndim == 4 else "tensor"
        shape_label = "matrix [T,B,R,C]" if representation == "matrix" else "tensor [T,B,...]"
        metric = f"baseline-relative normalized Frobenius distance; {shape_label}"
    return {
        "representation": representation,
        "metric": metric,
        "mean": float(np.mean(distances)),
        "per_tick_mean": np.mean(distances, axis=1).tolist(),
        "per_sample_mean": np.mean(distances, axis=0).tolist(),
        "tick_ids": ticks.tolist(),
        "sync_shape": list(left.shape),
    }


def dynamic_separability(z_history: Any, group_ids: Any) -> dict[str, Any]:
    """Offline-only within-minus-between cosine similarity of CTM trajectories.

    Group IDs are accepted here solely for evaluation; they must never enter a
    model, adapter, probe, or training path.
    """

    history = _matrix("z_history", z_history, 3)
    groups = np.asarray(group_ids)
    if groups.ndim != 1 or groups.shape[0] != history.shape[1]:
        raise ValueError(
            f"group_ids must have shape [{history.shape[1]}], got {groups.shape}"
        )
    if len(np.unique(groups)) < 2:
        raise ValueError("dynamic separability requires at least two distinct groups")
    trajectories = _normalized_tick_concat(history, "z_history")
    similarities = _cosine_similarity_matrix(trajectories, "trajectories")
    upper = np.triu_indices(history.shape[1], k=1)
    same = groups[upper[0]] == groups[upper[1]]
    if not np.any(same) or np.all(same):
        raise ValueError("dynamic separability needs both within-group and between-group pairs")
    within = float(np.mean(similarities[upper][same]))
    between = float(np.mean(similarities[upper][~same]))
    return {
        "within_group_similarity": within,
        "between_group_similarity": between,
        "separability": within - between,
        "sample_count": int(groups.shape[0]),
        "group_count": int(len(np.unique(groups))),
        "label_use": "offline evaluation only",
    }


def compare_checkpoints(baseline: Any, current: Any, k: int = 10) -> dict[str, Any]:
    """Validate checkpoint alignment, then compute four drift summaries."""

    for field in ("route", "seed"):
        if baseline.manifest.get(field) != current.manifest.get(field):
            raise ValueError(
                f"checkpoint comparison requires matching {field}; got "
                f"{baseline.manifest.get(field)!r} and {current.manifest.get(field)!r}"
            )
    if not np.array_equal(baseline.probe_ids, current.probe_ids):
        raise ValueError("fixed probe IDs/order differ between checkpoints")
    if not np.array_equal(baseline.tick_ids, current.tick_ids):
        raise ValueError("CTM tick IDs differ between checkpoints")
    for name in ("features", "z_history", "sync_history", "action_history"):
        if getattr(baseline, name).shape != getattr(current, name).shape:
            raise ValueError(
                f"{name} shape changed across checkpoints: "
                f"{getattr(baseline, name).shape} vs {getattr(current, name).shape}"
            )

    feature = feature_drift(baseline.features, current.features)
    structure = similarity_structure_drift(baseline.features, current.features, k=k)
    trajectory = trajectory_drift(baseline.z_history, current.z_history, baseline.tick_ids)
    sync = synchronization_drift(
        baseline.sync_history, current.sync_history, baseline.tick_ids
    )
    action = synchronization_drift(
        baseline.action_history, current.action_history, baseline.tick_ids
    )
    return {
        "route": baseline.manifest["route"],
        "seed": int(baseline.manifest["seed"]),
        "baseline_checkpoint": baseline.manifest["checkpoint_id"],
        "checkpoint": current.manifest["checkpoint_id"],
        "probe_count": baseline.sample_count,
        "tick_ids": baseline.tick_ids.tolist(),
        "sync_representation": baseline.sync_representation,
        "feature_drift_mean": feature["mean"],
        "feature_drift_per_sample": feature["per_sample"],
        "similarity_triangle_mae": structure["upper_triangle_mean_absolute_change"],
        "similarity_triangle_correlation": structure["upper_triangle_pearson_correlation"],
        "knn_k": structure["knn_k"],
        "knn_overlap_mean": structure["knn_overlap_mean"],
        "knn_overlap_per_sample": structure["knn_overlap_per_sample"],
        "trajectory_drift_primary_mean": trajectory["mean"],
        "trajectory_drift_sensitivity_mean": trajectory["mean_over_ticks_sensitivity"],
        "trajectory_drift_per_tick": trajectory["per_tick_mean_sensitivity"],
        "synchronization_drift_mean": sync["mean"],
        "synchronization_drift_per_tick": sync["per_tick_mean"],
        "synchronization_metric": sync["metric"],
        "synchronization_shape": sync["sync_shape"],
        "action_representation": baseline.action_representation,
        "action_drift_mean": action["mean"],
        "action_drift_per_tick": action["per_tick_mean"],
        "action_metric": action["metric"],
        "action_shape": action["sync_shape"],
    }
