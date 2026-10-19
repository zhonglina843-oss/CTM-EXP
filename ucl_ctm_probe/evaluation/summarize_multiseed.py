"""Aggregate the AE+CTM seed summaries into reproducible CSV/JSON/PNG outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics


METRICS = (
    "reconstruction_mse",
    "feature_drift_mean",
    "similarity_triangle_mae",
    "trajectory_drift_mean",
    "sync_out_drift_mean",
    "sync_action_drift_mean",
)
CHECKPOINTS = ("012_initial", "3_update", "4_update")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    per_seed = []
    calibration = {}
    for seed in (1, 2, 3):
        summary = json.loads((args.input_dir / f"summary_seed{seed:02d}.json").read_text())
        calibration[str(seed)] = summary["probe_calibration_mse"]
        by_checkpoint = {row["checkpoint"]: row for row in summary["checkpoints"]}
        drift_rows = {
            row["checkpoint"]: row
            for row in csv.DictReader((args.input_dir / f"drift_summary_seed{seed:02d}.csv").open())
        }
        for checkpoint in CHECKPOINTS:
            native = by_checkpoint[checkpoint]
            drift = drift_rows.get(checkpoint, {})
            per_seed.append({
                "seed": seed,
                "checkpoint": checkpoint,
                "reconstruction_mse": float(native["mean_reconstruction_mse"]),
                "feature_drift_mean": float(drift.get("feature_drift_mean", 0.0)),
                "similarity_triangle_mae": float(drift.get("similarity_triangle_mae", 0.0)),
                "trajectory_drift_mean": float(drift.get("trajectory_drift_mean", 0.0)),
                "sync_out_drift_mean": float(drift.get("sync_out_drift_mean", 0.0)),
                "sync_action_drift_mean": float(drift.get("sync_action_drift_mean", 0.0)),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = ["seed", "checkpoint", *METRICS]
    with (args.output_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_seed)

    aggregate = []
    for checkpoint in CHECKPOINTS:
        subset = [row for row in per_seed if row["checkpoint"] == checkpoint]
        result = {"checkpoint": checkpoint, "n_seeds": len(subset)}
        for metric in METRICS:
            values = [row[metric] for row in subset]
            result[f"{metric}_mean"] = statistics.mean(values)
            result[f"{metric}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate.append(result)
    with (args.output_dir / "aggregate.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)

    (args.output_dir / "aggregate.json").write_text(json.dumps({
        "kind": "three-seed AE+CTM feasibility run; not a final scientific result",
        "seeds": [1, 2, 3],
        "calibration_loss_by_seed": calibration,
        "aggregate": aggregate,
    }, indent=2) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    x = np.arange(len(CHECKPOINTS))
    for ax, metric in zip(axes.flat, METRICS):
        values = np.array([[row[metric] for row in per_seed if row["checkpoint"] == cp] for cp in CHECKPOINTS])
        for seed_idx in range(values.shape[1]):
            ax.plot(x, values[:, seed_idx], color="#9aa4b2", alpha=0.6, linewidth=1.3, marker=".")
        means = values.mean(axis=1)
        sd = values.std(axis=1, ddof=1)
        ax.errorbar(x, means, yerr=sd, color="#245c89", marker="o", capsize=4, linewidth=2)
        ax.set_title(metric.replace("_", " "))
        ax.set_xticks(x, CHECKPOINTS, rotation=15)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("AE + calibrated frozen CTM | 3-seed server feasibility run", color="#8f2525")
    fig.tight_layout()
    fig.savefig(args.output_dir / "checkpoint_panel.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
