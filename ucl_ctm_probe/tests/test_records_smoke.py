"""Synthetic record smoke test; contains no model training or real results.

Run from the repository root:

    python3 -m unittest ucl_ctm_probe.tests.test_records_smoke -v

The passing end-to-end test writes small synthetic JSON/CSV summaries and a
PNG figure under ``artifacts/A5/``. These outputs validate the pipeline only.
"""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ucl_ctm_probe.evaluation.metrics import (
    compare_checkpoints,
    dynamic_separability,
    feature_drift,
    similarity_structure_drift,
    synchronization_drift,
    trajectory_drift,
)
from ucl_ctm_probe.evaluation.records import load_checkpoint
from ucl_ctm_probe.schemas.records import RecordValidationError


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "A5"
TICK_IDS = np.array([1, 2, 4, 8, 12, 16, 20], dtype=np.int64)


def _make_record(path: Path, checkpoint_id: str, shift: float = 0.0) -> None:
    rng = np.random.default_rng(20261019)
    ids = np.arange(100, 106, dtype=np.int64)
    features0 = rng.normal(size=(6, 5))
    tick_count = len(TICK_IDS)
    z0 = rng.normal(size=(tick_count, 6, 4))
    sync0 = rng.normal(size=(tick_count, 6, 3))  # A vector representation, not a matrix.
    action0 = rng.normal(size=(tick_count, 6, 2))  # Retained separately from CTM out.
    features = features0 + shift * rng.normal(size=features0.shape)
    z_history = z0 + shift * rng.normal(size=z0.shape)
    sync_history = sync0 + shift * rng.normal(size=sync0.shape)
    action_history = action0 + shift * rng.normal(size=action0.shape)
    post_state = rng.normal(size=(6, 7))

    path.mkdir(parents=True)
    np.save(path / "probe_ids.npy", ids)
    np.save(path / "features.npy", features)
    np.savez_compressed(
        path / "ctm_trace.npz",
        tick_ids=TICK_IDS,
        z_history=z_history,
        sync_history=sync_history,
        action_history=action_history,
        post_state=post_state,
    )
    shapes = {
        "probe_ids": list(ids.shape),
        "features": list(features.shape),
        "tick_ids": list(TICK_IDS.shape),
        "z_history": list(z_history.shape),
        "sync_history": list(sync_history.shape),
        "action_history": list(action_history.shape),
        "post_state": list(post_state.shape),
    }
    manifest = {
        "schema_version": "1.0.0",
        "route": "ae",
        "seed": 0,
        "checkpoint_id": checkpoint_id,
        "tick_ids": TICK_IDS.tolist(),
        "sync_representation": "vector",
        "action_representation": "vector",
        "sync_sources": {"sync_history": "out", "action_history": "action"},
        "array_shapes": shapes,
        "source": {"kind": "synthetic smoke test"},
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (path / "native_metrics.json").write_text(
        json.dumps({"synthetic_smoke": True}, indent=2) + "\n", encoding="utf-8"
    )


def _write_summary(result: dict) -> tuple[Path, Path, Path]:
    table_dir = ARTIFACT_ROOT / "tables"
    figure_dir = ARTIFACT_ROOT / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    json_path = table_dir / "synthetic_smoke_metrics.json"
    csv_path = table_dir / "synthetic_smoke_metrics.csv"
    figure_path = figure_dir / "synthetic_smoke_metrics.png"

    json_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = [
        ("feature_drift_mean", result["feature_drift_mean"], "mean per-sample 1-cosine"),
        ("similarity_triangle_mae", result["similarity_triangle_mae"], "upper-triangle mean absolute change"),
        ("knn_overlap_mean", result["knn_overlap_mean"], "mean overlap; higher is more stable"),
        ("trajectory_drift_primary_mean", result["trajectory_drift_primary_mean"], "unit-normalized ticks concatenated"),
        ("synchronization_drift_mean", result["synchronization_drift_mean"], result["synchronization_metric"]),
        ("action_drift_mean", result["action_drift_mean"], result["action_metric"]),
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("metric", "synthetic_value", "definition"))
        writer.writerows(rows)
        for tick, value in zip(TICK_IDS, result["trajectory_drift_per_tick"]):
            writer.writerow((f"trajectory_drift_tick_{tick}", value, "same-tick mean 1-cosine"))
        for tick, value in zip(TICK_IDS, result["synchronization_drift_per_tick"]):
            writer.writerow((f"synchronization_drift_tick_{tick}", value, result["synchronization_metric"]))
        for tick, value in zip(TICK_IDS, result["action_drift_per_tick"]):
            writer.writerow((f"action_drift_tick_{tick}", value, result["action_metric"]))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Feature\ncosine drift", "Similarity\ntriangle MAE", "Trajectory\ncosine drift", "CTM out\nvector drift", "CTM action\nvector drift"]
    values = [
        result["feature_drift_mean"],
        result["similarity_triangle_mae"],
        result["trajectory_drift_primary_mean"],
        result["synchronization_drift_mean"],
        result["action_drift_mean"],
    ]
    fig, (ax, shape_ax) = plt.subplots(
        1, 2, figsize=(12, 5.6), gridspec_kw={"width_ratios": [1.0, 1.15]}
    )
    bars = ax.bar(labels, values, color=["#4C78A8", "#72B7B2", "#F58518", "#B279A2", "#54A24B"])
    ax.set_title("Synthetic drift outputs")
    ax.set_ylabel("Metric value (scales are metric-specific)")
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    shape_ax.axis("off")
    shape_ax.set_title("Record → validation → aligned comparison")
    boxes = [
        (0.05, 0.72, "Checkpoint A / B\nIDs [B] • features [B,D]"),
        (0.05, 0.43, "CTM trace\nz [T,B,Dz] • out [T,B,...] • action [T,B,...]"),
        (0.58, 0.57, "Schema checks\nIDs, tick order, exact shapes"),
        (0.58, 0.25, "Four drift views + both outputs\nfeature • relation • trajectory • out • action"),
    ]
    for x, y, text in boxes:
        shape_ax.text(x, y, text, transform=shape_ax.transAxes, ha="left", va="center", fontsize=9,
                      bbox={"boxstyle": "round,pad=0.55", "facecolor": "#F2F4F7", "edgecolor": "#65758B"})
    arrow = {"arrowstyle": "->", "color": "#65758B", "lw": 1.3}
    shape_ax.annotate("", xy=(0.56, 0.64), xytext=(0.42, 0.73), xycoords="axes fraction", arrowprops=arrow)
    shape_ax.annotate("", xy=(0.56, 0.36), xytext=(0.42, 0.49), xycoords="axes fraction", arrowprops=arrow)
    shape_ax.annotate("same fixed IDs + ticks", xy=(0.66, 0.46), xytext=(0.67, 0.53), xycoords="axes fraction", ha="left", fontsize=8, arrowprops=arrow)
    fig.suptitle("SYNTHETIC CONTRACT SMOKE — NOT EXPERIMENT DATA", color="#9B2226", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return json_path, csv_path, figure_path


class RecordSmokeTests(unittest.TestCase):
    def test_end_to_end_synthetic_record_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ucl-ctm-smoke-") as temp:
            root = Path(temp)
            base_path = root / "records/ae/seed_0/checkpoint_000"
            next_path = root / "records/ae/seed_0/checkpoint_001"
            _make_record(base_path, "000", shift=0.0)
            _make_record(next_path, "001", shift=0.12)
            base = load_checkpoint(base_path)
            current = load_checkpoint(next_path)
            self.assertEqual(base.sync_representation, "vector")
            self.assertEqual(base.sync_history.shape, (7, 6, 3))
            self.assertEqual(base.action_history.shape, (7, 6, 2))
            metrics = compare_checkpoints(base, current, k=2)
            self.assertEqual(metrics["probe_count"], 6)
            self.assertEqual(metrics["synchronization_shape"], [7, 6, 3])
            self.assertEqual(metrics["action_shape"], [7, 6, 2])
            for name in (
                "feature_drift_mean",
                "similarity_triangle_mae",
                "trajectory_drift_primary_mean",
                "synchronization_drift_mean",
                "action_drift_mean",
            ):
                self.assertTrue(np.isfinite(metrics[name]), name)

            json_path, csv_path, figure_path = _write_summary(metrics)
            for output in (json_path, csv_path, figure_path):
                self.assertTrue(output.is_file(), output)
                self.assertGreater(output.stat().st_size, 0)

    def test_metrics_reject_empty_or_wrong_rank_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty"):
            feature_drift(np.empty((0, 4)), np.empty((0, 4)))
        with self.assertRaisesRegex(ValueError, "2 dimensions"):
            feature_drift(np.ones(4), np.ones(4))

    def test_reader_rejects_shape_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ucl-ctm-bad-shape-") as temp:
            path = Path(temp) / "checkpoint_000"
            _make_record(path, "000")
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["array_shapes"]["features"] = [6, 999]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RecordValidationError, "array_shapes mismatch"):
                load_checkpoint(path)

    def test_dynamic_separability_is_offline_only(self) -> None:
        rng = np.random.default_rng(17)
        trajectories = rng.normal(size=(3, 6, 5))
        result = dynamic_separability(trajectories, np.array([0, 0, 0, 1, 1, 1]))
        self.assertEqual(result["label_use"], "offline evaluation only")
        self.assertTrue(np.isfinite(result["separability"]))

    def test_sync_matrix_metric_preserves_real_shape(self) -> None:
        rng = np.random.default_rng(19)
        baseline = rng.normal(size=(2, 4, 3, 3))
        current = baseline + 0.05
        result = synchronization_drift(baseline, current, np.array([1, 2]))
        self.assertEqual(result["representation"], "matrix")
        self.assertEqual(result["sync_shape"], [2, 4, 3, 3])
        self.assertIn("Frobenius", result["metric"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
