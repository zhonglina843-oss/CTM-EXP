"""Disk reader and alignment checks for checkpoint records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np

from ucl_ctm_probe.schemas.records import (
    CheckpointRecord,
    RecordValidationError,
    validate_arrays,
)


REQUIRED_FILES = (
    "manifest.json",
    "probe_ids.npy",
    "features.npy",
    "native_metrics.json",
    "ctm_trace.npz",
)


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordValidationError(f"{path}: cannot read JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise RecordValidationError(f"{path}: expected a JSON object")
    return value


def _load_npy(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except RecordValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise RecordValidationError(f"{path}: cannot read NumPy array: {exc}") from exc


def load_checkpoint(path: str | Path) -> CheckpointRecord:
    """Load one ``records/.../checkpoint_*`` directory.

    Malformed or misaligned checkpoint files fail before metrics are computed.
    """

    root = Path(path)
    if not root.is_dir():
        raise RecordValidationError(f"{root}: checkpoint directory does not exist")
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if missing:
        raise RecordValidationError(f"{root}: missing file(s): {', '.join(missing)}")

    manifest = _load_json(root / "manifest.json")
    probe_ids = _load_npy(root / "probe_ids.npy")
    features = _load_npy(root / "features.npy")
    native_metrics = _load_json(root / "native_metrics.json")
    try:
        with np.load(root / "ctm_trace.npz", allow_pickle=False) as trace_file:
            required_trace = ("tick_ids", "z_history", "sync_history", "post_state")
            missing_trace = [name for name in required_trace if name not in trace_file.files]
            if missing_trace:
                raise RecordValidationError(
                    f"{root / 'ctm_trace.npz'}: missing array(s): {', '.join(missing_trace)}"
                )
            tick_ids = trace_file["tick_ids"]
            z_history = trace_file["z_history"]
            sync_history = trace_file["sync_history"]
            if "action_history" not in trace_file.files:
                raise RecordValidationError(
                    f"{root / 'ctm_trace.npz'}: missing array 'action_history'; "
                    "CTM out and action outputs must remain separate"
                )
            action_history = trace_file["action_history"]
            post_state = trace_file["post_state"]
    except RecordValidationError:
        raise
    except (OSError, ValueError) as exc:
        raise RecordValidationError(f"{root / 'ctm_trace.npz'}: cannot read trace: {exc}") from exc

    sync_representation, action_representation = validate_arrays(
        manifest,
        probe_ids,
        features,
        native_metrics,
        tick_ids,
        z_history,
        sync_history,
        action_history,
        post_state,
        source=str(root),
    )

    return CheckpointRecord(
        path=root,
        manifest=manifest,
        probe_ids=probe_ids,
        features=features,
        native_metrics=native_metrics,
        tick_ids=tick_ids,
        z_history=z_history,
        sync_history=sync_history,
        action_history=action_history,
        post_state=post_state,
        sync_representation=sync_representation,
        action_representation=action_representation,
    )


def iter_checkpoint_paths(records_root: str | Path) -> Iterable[Path]:
    """Yield checkpoint directories containing a manifest in stable path order."""

    root = Path(records_root)
    if not root.is_dir():
        raise RecordValidationError(f"{root}: records root does not exist")
    yield from sorted((path.parent for path in root.rglob("manifest.json")), key=str)


def load_records(records_root: str | Path) -> list[CheckpointRecord]:
    """Load and validate every checkpoint below a records root."""

    records = [load_checkpoint(path) for path in iter_checkpoint_paths(records_root)]
    if not records:
        raise RecordValidationError(f"{records_root}: no manifest.json files found")
    return records
