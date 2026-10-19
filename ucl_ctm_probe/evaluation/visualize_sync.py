"""Plot CTM synchronisation matrices reconstructed from first-last upper triangles.

No model forward pass or GPU is used. Inputs are saved checkpoint records and
the MNIST test labels needed to group the fixed probe set offline.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import struct

import numpy as np

from ucl_ctm_probe.evaluation.metrics import synchronization_drift
from ucl_ctm_probe.evaluation.records import load_checkpoint


CHANNELS = ("sync_history", "action_history")
CHANNEL_LABELS = {"sync_history": "out (neurons 0-31)", "action_history": "action (neurons 96-127)"}


def read_idx_labels(mnist_root: Path) -> np.ndarray:
    raw = mnist_root / "MNIST" / "raw"
    path = raw / "t10k-labels-idx1-ubyte"
    opener = open
    if not path.is_file():
        path = path.with_suffix(path.suffix + ".gz")
        opener = gzip.open
    if not path.is_file():
        raise FileNotFoundError(f"MNIST test-label file not found under {raw}")
    with opener(path, "rb") as stream:
        magic, count = struct.unpack(">II", stream.read(8))
        if magic != 2049:
            raise ValueError(f"unexpected IDX label magic {magic} in {path}")
        labels = np.frombuffer(stream.read(), dtype=np.uint8)
    if len(labels) != count:
        raise ValueError(f"IDX label count mismatch: header={count}, data={len(labels)}")
    return labels


def vector_to_symmetric(vector: np.ndarray) -> np.ndarray:
    """Reconstruct a first-last symmetric matrix from its stored upper triangle."""
    vector = np.asarray(vector, dtype=np.float64)
    n = int((np.sqrt(8 * vector.size + 1) - 1) / 2)
    if n * (n + 1) // 2 != vector.size:
        raise ValueError(f"length {vector.size} is not a triangular-number upper triangle")
    i, j = np.triu_indices(n)
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[i, j] = vector
    matrix[j, i] = vector
    return matrix


def save_figure(fig, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{stem}.svg", bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


def load_seed(records_root: Path, seed: int):
    seed_root = records_root / f"seed_{seed}"
    paths = sorted(seed_root.glob("checkpoint_*"))
    records = [load_checkpoint(path) for path in paths]
    if len(records) != 3:
        raise ValueError(f"seed {seed}: expected 3 checkpoints under {seed_root}, found {len(records)}")
    return records


def check_probe_labels(records, labels: np.ndarray) -> np.ndarray:
    ids = records[0].probe_ids.astype(np.int64)
    if np.any(ids < 0) or np.any(ids >= len(labels)):
        raise ValueError("probe IDs exceed MNIST test-label range")
    for record in records[1:]:
        if not np.array_equal(ids, record.probe_ids):
            raise ValueError("probe IDs/order differ across checkpoints")
    for record in records:
        if not np.array_equal(record.tick_ids, records[0].tick_ids):
            raise ValueError("tick IDs differ across checkpoints")
    return labels[ids]


def plot_sample_before_after(records, labels, sample_label, sample_offset, output):
    import matplotlib.pyplot as plt

    matching = np.flatnonzero(labels == sample_label)
    if sample_offset >= len(matching):
        raise ValueError(f"no probe offset {sample_offset} for label {sample_label}")
    sample = int(matching[sample_offset])
    before, after = records[0], records[-1]
    tick_index = len(before.tick_ids) - 1
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 6.2))
    for row, channel in enumerate(CHANNELS):
        m0 = vector_to_symmetric(getattr(before, channel)[tick_index, sample])
        m1 = vector_to_symmetric(getattr(after, channel)[tick_index, sample])
        delta = m1 - m0
        vmax = float(np.percentile(np.concatenate([m0.ravel(), m1.ravel()]), 99.5))
        dmax = float(np.percentile(np.abs(delta), 99.5)) or 1e-12
        for col, (matrix, title, cmap, vmin, vmax_panel) in enumerate((
            (m0, f"Initial · tick {before.tick_ids[tick_index]}", "viridis", 0, vmax),
            (m1, f"After class 4 · tick {after.tick_ids[tick_index]}", "viridis", 0, vmax),
            (delta, "After - initial", "RdBu_r", -dmax, dmax),
        )):
            ax = axes[row, col]
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax_panel, interpolation="nearest")
            ax.set_title(title)
            ax.set_xlabel("selected neuron index")
            ax.set_ylabel(CHANNEL_LABELS[channel])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle(f"Fixed old MNIST probe | label {sample_label}, sample offset {sample_offset}")
    fig.text(0.5, 0.01, "Symmetric reconstruction from 528 first-last upper-triangle sync statistics; not a correlation matrix.", ha="center", fontsize=8)
    save_figure(fig, output, "sync_sample_initial_after4_delta")


def plot_sample_tick_differences(records, labels, sample_label, sample_offset, output):
    import matplotlib.pyplot as plt

    matching = np.flatnonzero(labels == sample_label)
    if sample_offset >= len(matching):
        raise ValueError(f"no probe offset {sample_offset} for label {sample_label}")
    sample = int(matching[sample_offset])
    before, after = records[0], records[-1]
    fig, axes = plt.subplots(2, len(before.tick_ids), figsize=(16, 5.8), squeeze=False)
    for row, channel in enumerate(CHANNELS):
        deltas = [vector_to_symmetric(getattr(after, channel)[i, sample]) -
                  vector_to_symmetric(getattr(before, channel)[i, sample])
                  for i in range(len(before.tick_ids))]
        limit = float(np.percentile(np.abs(np.stack(deltas)), 99.5)) or 1e-12
        for col, delta in enumerate(deltas):
            ax = axes[row, col]
            im = ax.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_title(f"tick {before.tick_ids[col]}")
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(CHANNEL_LABELS[channel])
        fig.colorbar(im, ax=axes[row].tolist(), fraction=0.018, pad=0.01)
    fig.suptitle(f"Sync change across CTM ticks | fixed old label {sample_label}, offset {sample_offset}")
    fig.text(0.5, 0.01, "Each panel is after-class-4 minus initial; one shared diverging range per output.", ha="center", fontsize=8)
    save_figure(fig, output, "sync_sample_delta_by_tick")


def plot_class_mean_matrices(records, labels, output):
    import matplotlib.pyplot as plt

    before, after = records[0], records[-1]
    fig, axes = plt.subplots(6, 3, figsize=(8.5, 15.5))
    for channel_index, channel in enumerate(CHANNELS):
        positive_handle = None
        difference_handle = None
        class_matrices = {}
        all_values = []
        for label in (0, 1, 2):
            indices = np.flatnonzero(labels == label)
            for stage_name, record in (("initial", before), ("after4", after)):
                vectors = getattr(record, channel)[-1, indices]
                matrix = vector_to_symmetric(vectors.mean(axis=0))
                class_matrices[label, stage_name] = matrix
                all_values.append(matrix.ravel())
        vmax = float(np.percentile(np.concatenate(all_values), 99.5))
        deltas = [class_matrices[label, "after4"] - class_matrices[label, "initial"] for label in (0, 1, 2)]
        dmax = float(np.percentile(np.abs(np.stack(deltas)), 99.5)) or 1e-12
        for class_index, label in enumerate((0, 1, 2)):
            row = channel_index * 3 + class_index
            mats = (
                class_matrices[label, "initial"],
                class_matrices[label, "after4"],
                deltas[class_index],
            )
            for col, matrix in enumerate(mats):
                ax = axes[row, col]
                if col < 2:
                    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
                    positive_handle = im
                else:
                    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-dmax, vmax=dmax, interpolation="nearest")
                    difference_handle = im
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(("Initial mean", "After class 4 mean", "Difference")[col])
                if col == 0:
                    ax.set_ylabel(f"{CHANNEL_LABELS[channel]}\nMNIST {label}")
        rows = axes[channel_index * 3:(channel_index + 1) * 3]
        fig.colorbar(positive_handle, ax=rows[:, :2].ravel().tolist(), fraction=0.025, pad=0.02, label="sync statistic")
        fig.colorbar(difference_handle, ax=rows[:, 2].ravel().tolist(), fraction=0.05, pad=0.04, label="change")
    fig.suptitle("Class-mean CTM sync statistics for fixed old probes", y=0.995)
    fig.text(0.5, 0.002, "Rows average the 200 fixed old test samples in each class. Upper-triangle vectors are reconstructed symmetrically.", ha="center", fontsize=8)
    save_figure(fig, output, "sync_class_mean_initial_after4_delta")


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, 1e-12)) @ (vectors / np.maximum(norms, 1e-12)).T


def plot_class_pattern_similarity(records, labels, output):
    import matplotlib.pyplot as plt

    before, after = records[0], records[-1]
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 6.2))
    for row, channel in enumerate(CHANNELS):
        means = {}
        for stage_name, record in (("initial", before), ("after4", after)):
            means[stage_name] = np.stack([
                getattr(record, channel)[-1, np.flatnonzero(labels == label)].mean(axis=0)
                for label in (0, 1, 2)
            ])
        sims0 = cosine_matrix(means["initial"])
        sims1 = cosine_matrix(means["after4"])
        delta = sims1 - sims0
        dmax = float(np.max(np.abs(delta))) or 1e-12
        for col, (matrix, title, cmap, vmin, vmax) in enumerate((
            (sims0, "Initial", "viridis", 0, 1),
            (sims1, "After class 4", "viridis", 0, 1),
            (delta, "After - initial", "RdBu_r", -dmax, dmax),
        )):
            ax = axes[row, col]
            im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(3), ["0", "1", "2"])
            ax.set_yticks(range(3), ["0", "1", "2"])
            ax.set_xlabel("MNIST class")
            ax.set_ylabel(CHANNEL_LABELS[channel])
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Cosine similarity between class-mean sync vectors")
    save_figure(fig, output, "sync_class_pattern_similarity")


def plot_sync_vs_retention(records_by_seed, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    colors = {1: "#4477AA", 2: "#CC6677", 3: "#228833"}
    source_rows = []
    for seed, records in records_by_seed.items():
        baseline = records[0]
        base_mse = float(baseline.native_metrics["mean_reconstruction_mse"])
        for record, marker in zip(records[1:], ("o", "s")):
            current_mse = float(record.native_metrics["mean_reconstruction_mse"])
            out = synchronization_drift(baseline.sync_history, record.sync_history, baseline.tick_ids)
            action = synchronization_drift(baseline.action_history, record.action_history, baseline.tick_ids)
            row = {
                "seed": seed,
                "checkpoint": record.manifest["checkpoint_id"],
                "delta_old_reconstruction_mse": current_mse - base_mse,
                "sync_out_drift": out["mean"],
                "sync_action_drift": action["mean"],
            }
            source_rows.append(row)
            for ax, xkey, title in (
                (axes[0], "sync_out_drift", "out sync drift"),
                (axes[1], "sync_action_drift", "action sync drift"),
            ):
                ax.scatter(row[xkey], row["delta_old_reconstruction_mse"], color=colors[seed], marker=marker, s=55)
                ax.annotate(f"s{seed} {record.manifest['checkpoint_id'].split('_')[0]}",
                            (row[xkey], row["delta_old_reconstruction_mse"]), xytext=(4, 4),
                            textcoords="offset points", fontsize=6)
                ax.set_title(title)
                ax.set_xlabel("sync drift from initial checkpoint")
                ax.set_ylabel("old-test reconstruction MSE increase")
                ax.axhline(0, color="#999999", linewidth=0.7)
                ax.axvline(0, color="#999999", linewidth=0.7)
                ax.grid(alpha=0.2)
    fig.suptitle("Descriptive seed/checkpoint association (n=3 seeds; no fitted trend)")
    save_figure(fig, output, "sync_drift_vs_old_reconstruction")
    return source_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-root", type=Path, required=True, help=".../records/ae")
    parser.add_argument("--mnist-root", type=Path, required=True, help="MNIST root containing MNIST/raw")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sample-label", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--selection", choices=("first-last",), required=True,
                        help="Only first-last has a verified upper-triangle-to-neuron mapping here.")
    args = parser.parse_args()

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = read_idx_labels(args.mnist_root)
    records_by_seed = {seed: load_seed(args.records_root, seed) for seed in args.seeds}
    primary_seed = args.seeds[0]
    primary = records_by_seed[primary_seed]
    probe_labels = check_probe_labels(primary, labels)
    for seed, records in records_by_seed.items():
        seed_labels = check_probe_labels(records, labels)
        counts = [int(np.sum(seed_labels == label)) for label in (0, 1, 2)]
        if counts != [200, 200, 200]:
            raise ValueError(f"seed {seed}: expected 200 probe examples per old class, got {counts}")
        for record in records:
            if record.sync_history.shape[-1] != 32 * 33 // 2 or record.action_history.shape[-1] != 32 * 33 // 2:
                raise ValueError("verified first-last mapping expects 528 values (32 selected neurons) per sync output")

    plot_sample_before_after(primary, probe_labels, args.sample_label, args.sample_offset, args.output_dir)
    plot_sample_tick_differences(primary, probe_labels, args.sample_label, args.sample_offset, args.output_dir)
    plot_class_mean_matrices(primary, probe_labels, args.output_dir)
    plot_class_pattern_similarity(primary, probe_labels, args.output_dir)
    scatter_rows = plot_sync_vs_retention(records_by_seed, args.output_dir)
    with (args.output_dir / "sync_retention_points.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scatter_rows[0]))
        writer.writeheader()
        writer.writerows(scatter_rows)
    (args.output_dir / "plot_manifest.json").write_text(json.dumps({
        "kind": "exploratory visualization of saved CTM traces; no model inference performed",
        "seeds": args.seeds,
        "sample_label": args.sample_label,
        "sample_offset_within_label": args.sample_offset,
        "matrix_mapping": "first-last upper triangle, mirrored symmetrically; 32 selected neurons per output",
        "out_global_neurons": [0, 31],
        "action_global_neurons": [96, 127],
        "tick_ids": primary[0].tick_ids.tolist(),
        "interpretation_limit": "sync change alone is not forgetting or causality",
        "figures": [
            "sync_sample_initial_after4_delta.png",
            "sync_sample_delta_by_tick.png",
            "sync_class_mean_initial_after4_delta.png",
            "sync_class_pattern_similarity.png",
            "sync_drift_vs_old_reconstruction.png",
        ],
    }, indent=2) + "\n")
    plt.close("all")
    print(json.dumps({"figures": sorted(path.name for path in args.output_dir.glob("*.png"))}, indent=2))


if __name__ == "__main__":
    main()
