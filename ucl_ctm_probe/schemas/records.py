"""Strict, small schema for one UCL x CTM checkpoint record.

The schema follows ``docs/UCL_CTM_Probe_Implementation_Plan.md``.  In
particular, ``sync_history`` is kept as either a vector tensor
``[T, B, D]`` or a per-sample matrix tensor ``[T, B, R, C]``.  The validator
never flattens the latter and never calls a vector a matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class RecordValidationError(ValueError):
    """Raised when a checkpoint record violates the immutable contract."""


@dataclass(frozen=True)
class CheckpointRecord:
    """Loaded checkpoint arrays and manifest metadata.

    Shapes are validated at load time:

    * ``probe_ids``: ``[B]``
    * ``features``: ``[B, D]``
    * ``tick_ids``: ``[T]``
    * ``z_history``: ``[T, B, Dz]``
    * ``sync_history``: primary CTM ``out`` tensor, shape ``[T, B, ...]``
    * ``action_history``: distinct CTM ``action`` tensor, shape ``[T, B, ...]``
    * ``post_state``: ``[B, Dpost]``
    """

    path: Path
    manifest: Mapping[str, Any]
    probe_ids: np.ndarray
    features: np.ndarray
    native_metrics: Mapping[str, Any]
    tick_ids: np.ndarray
    z_history: np.ndarray
    sync_history: np.ndarray
    action_history: np.ndarray
    post_state: np.ndarray
    sync_representation: str
    action_representation: str

    @property
    def sample_count(self) -> int:
        return int(self.probe_ids.shape[0])

    @property
    def tick_count(self) -> int:
        return int(self.tick_ids.shape[0])


def _fail(source: str, message: str) -> None:
    raise RecordValidationError(f"{source}: {message}")


def _require_numeric(array: np.ndarray, source: str, name: str) -> None:
    if array.dtype.kind not in "fiu":
        _fail(source, f"{name} must have a real numeric dtype, got {array.dtype}")
    if not np.all(np.isfinite(array)):
        _fail(source, f"{name} contains NaN or infinite values")


def _require_shape(array: np.ndarray, source: str, name: str, ndim: int) -> None:
    if array.ndim != ndim:
        _fail(source, f"{name} must be {ndim}-D, got shape {array.shape}")
    if any(size == 0 for size in array.shape):
        _fail(source, f"{name} must be non-empty, got shape {array.shape}")


def validate_manifest(manifest: Mapping[str, Any], source: str = "manifest") -> None:
    """Validate required manifest fields and their basic types.

    Shape-specific optional fields are checked by :func:`validate_arrays`.
    """

    if not isinstance(manifest, Mapping):
        _fail(source, "must be a JSON object")
    required = (
        "schema_version",
        "route",
        "seed",
        "checkpoint_id",
        "tick_ids",
        "sync_representation",
        "action_representation",
        "sync_sources",
        "array_shapes",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        _fail(source, f"missing required field(s): {', '.join(missing)}")
    if not isinstance(manifest["route"], str) or not manifest["route"].strip():
        _fail(source, "route must be a non-empty string")
    if manifest["schema_version"] != "1.0.0":
        _fail(source, "schema_version must be '1.0.0'")
    seed = manifest["seed"]
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        _fail(source, "seed must be an integer")
    if seed < 0:
        _fail(source, "seed must be non-negative")
    checkpoint_id = manifest["checkpoint_id"]
    if isinstance(checkpoint_id, bool) or not isinstance(
        checkpoint_id, (str, int, np.integer)
    ):
        _fail(source, "checkpoint_id must be a string or integer")
    if isinstance(checkpoint_id, str) and not checkpoint_id.strip():
        _fail(source, "checkpoint_id must not be empty")

    for field in ("sync_representation", "action_representation"):
        if manifest[field] not in {"vector", "matrix", "tensor"}:
            _fail(source, f"{field} must be 'vector', 'matrix', or 'tensor'")
    if manifest["sync_sources"] != {"sync_history": "out", "action_history": "action"}:
        _fail(source, "sync_sources must map sync_history -> out and action_history -> action")
    ticks = manifest["tick_ids"]
    if not isinstance(ticks, list) or not ticks:
        _fail(source, "tick_ids must be a non-empty JSON array")
    if any(isinstance(tick, bool) or not isinstance(tick, int) or tick <= 0 for tick in ticks):
        _fail(source, "tick_ids must contain positive integers")
    if any(right <= left for left, right in zip(ticks, ticks[1:])):
        _fail(source, "tick_ids must be strictly increasing")
    shapes = manifest["array_shapes"]
    shape_names = {
        "probe_ids", "features", "tick_ids", "z_history", "sync_history",
        "action_history", "post_state"
    }
    if not isinstance(shapes, Mapping) or set(shapes) != shape_names:
        _fail(source, f"array_shapes must contain exactly: {', '.join(sorted(shape_names))}")
    for name, shape in shapes.items():
        if not isinstance(shape, list) or not shape:
            _fail(source, f"array_shapes.{name} must be a non-empty dimension list")
        if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
            _fail(source, f"array_shapes.{name} must contain positive integers")


def validate_arrays(
    manifest: Mapping[str, Any],
    probe_ids: np.ndarray,
    features: np.ndarray,
    native_metrics: Mapping[str, Any],
    tick_ids: np.ndarray,
    z_history: np.ndarray,
    sync_history: np.ndarray,
    action_history: np.ndarray,
    post_state: np.ndarray,
    source: str = "record",
) -> tuple[str, str]:
    """Validate arrays and return out/action synchronization representations."""

    validate_manifest(manifest, f"{source}/manifest")
    if not isinstance(native_metrics, Mapping):
        _fail(source, "native_metrics must be a JSON object")

    _require_shape(probe_ids, source, "probe_ids", 1)
    if probe_ids.dtype.kind not in "iuSU":
        _fail(source, f"probe_ids must be numeric or text, got {probe_ids.dtype}")
    if len(np.unique(probe_ids)) != probe_ids.shape[0]:
        _fail(source, "probe_ids must be unique and preserve fixed-set order")

    _require_shape(features, source, "features", 2)
    _require_numeric(features, source, "features")
    _require_shape(tick_ids, source, "tick_ids", 1)
    if tick_ids.dtype.kind not in "iu":
        _fail(source, f"tick_ids must have an integer dtype, got {tick_ids.dtype}")
    if np.any(tick_ids <= 0) or np.any(np.diff(tick_ids) <= 0):
        _fail(source, "tick_ids must be strictly increasing positive integers")

    _require_shape(z_history, source, "z_history", 3)
    _require_numeric(z_history, source, "z_history")
    _require_numeric(post_state, source, "post_state")
    _require_shape(post_state, source, "post_state", 2)
    _require_numeric(sync_history, source, "sync_history")
    if sync_history.ndim < 3:
        _fail(
            source,
            "sync_history must have at least 3 dimensions [T,B,...], "
            f"got shape {sync_history.shape}",
        )
    if any(size == 0 for size in sync_history.shape):
        _fail(source, f"sync_history must be non-empty, got {sync_history.shape}")
    _require_numeric(action_history, source, "action_history")
    if action_history.ndim < 3 or any(size == 0 for size in action_history.shape):
        _fail(source, f"action_history must be non-empty [T,B,...], got {action_history.shape}")

    def representation(array: np.ndarray) -> str:
        return "vector" if array.ndim == 3 else "matrix" if array.ndim == 4 else "tensor"

    sync_representation = representation(sync_history)
    action_representation = representation(action_history)

    batch = probe_ids.shape[0]
    ticks = tick_ids.shape[0]
    if features.shape[0] != batch:
        _fail(source, f"features batch {features.shape[0]} != probe_ids {batch}")
    if z_history.shape[:2] != (ticks, batch):
        _fail(
            source,
            f"z_history leading shape {z_history.shape[:2]} != (ticks,batch) "
            f"({ticks},{batch})",
        )
    if sync_history.shape[:2] != (ticks, batch):
        _fail(
            source,
            f"sync_history leading shape {sync_history.shape[:2]} != "
            f"(ticks,batch) ({ticks},{batch})",
        )
    if action_history.shape[:2] != (ticks, batch):
        _fail(
            source,
            f"action_history leading shape {action_history.shape[:2]} != "
            f"(ticks,batch) ({ticks},{batch})",
        )
    if post_state.shape[0] != batch:
        _fail(source, f"post_state batch {post_state.shape[0]} != probe_ids {batch}")

    declared_sync = manifest["sync_representation"]
    if declared_sync != sync_representation:
        _fail(
            source,
            f"declared sync_representation={declared_sync!r} but actual "
            f"shape {sync_history.shape} is {sync_representation}",
        )
    if manifest["action_representation"] != action_representation:
        _fail(
            source,
            f"declared action_representation={manifest['action_representation']!r} "
            f"but actual shape {action_history.shape} is {action_representation}",
        )

    if manifest["tick_ids"] != tick_ids.tolist():
        _fail(source, "manifest tick_ids do not match ctm_trace.npz")
    actual_shapes = {
        "probe_ids": list(probe_ids.shape),
        "features": list(features.shape),
        "tick_ids": list(tick_ids.shape),
        "z_history": list(z_history.shape),
        "sync_history": list(sync_history.shape),
        "action_history": list(action_history.shape),
        "post_state": list(post_state.shape),
    }
    if manifest["array_shapes"] != actual_shapes:
        mismatches = [
            f"{name}: declared {manifest['array_shapes'].get(name)} != actual {shape}"
            for name, shape in actual_shapes.items()
            if manifest["array_shapes"].get(name) != shape
        ]
        _fail(source, "manifest array_shapes mismatch (" + "; ".join(mismatches) + ")")

    return sync_representation, action_representation
