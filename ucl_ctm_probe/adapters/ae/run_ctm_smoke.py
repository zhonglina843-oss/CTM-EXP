"""Run the existing AE MNIST smoke, then record frozen CTM traces per checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import csv
import subprocess

import numpy as np
import torch
from torchvision.datasets import MNIST

ROOT = Path(__file__).resolve().parents[3]
CTM_REPO = ROOT.parent / "ctm-official"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CTM_REPO))

from ucl_ctm_probe.adapters.ae.adapter import AEAdapter  # noqa: E402
from ucl_ctm_probe.evaluation.metrics import compare_checkpoints  # noqa: E402
from ucl_ctm_probe.evaluation.records import load_records  # noqa: E402
from ucl_ctm_probe.probe.ctm_trace import (  # noqa: E402
    DEFAULT_TICK_IDS,
    calibrate_ctm_reconstructor,
    initialize_feature_ctm,
    trace_ctm,
)


def write_checkpoint(
    output: Path,
    checkpoint_id: str,
    probe_ids: np.ndarray,
    features: np.ndarray,
    native_metrics: dict,
    trace: dict,
    seed: int,
    ctm_hash: str,
    adapter_hash: str,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "probe_ids.npy", probe_ids)
    np.save(output / "features.npy", features.astype(np.float32))
    native = {
        "mean_reconstruction_mse": float(native_metrics["mean_reconstruction_mse"]),
        "reconstruction_mse": native_metrics["reconstruction_mse"].tolist(),
    }
    (output / "native_metrics.json").write_text(json.dumps(native, indent=2) + "\n")

    arrays = {
        "tick_ids": trace["tick_ids"],
        "z_history": trace["z_history"],
        "sync_history": trace["sync_history"]["out"],
        "action_history": trace["sync_history"]["action"],
        "post_state": trace["post_state"],
    }
    np.savez_compressed(output / "ctm_trace.npz", **arrays)
    sync_representation = "vector" if arrays["sync_history"].ndim == 3 else "matrix"
    action_representation = "vector" if arrays["action_history"].ndim == 3 else "matrix"
    manifest = {
        "schema_version": "1.0.0",
        "route": "ae",
        "seed": int(seed),
        "checkpoint_id": checkpoint_id,
        "tick_ids": arrays["tick_ids"].astype(int).tolist(),
        "sync_representation": sync_representation,
        "action_representation": action_representation,
        "sync_sources": {"sync_history": "out", "action_history": "action"},
        "array_shapes": {
            "probe_ids": list(probe_ids.shape),
            "features": list(features.shape),
            **{name: list(value.shape) for name, value in arrays.items()},
        },
        "source": {
            "ctm_repo": str(CTM_REPO),
            "ctm_commit": subprocess.check_output(
                ["git", "-C", str(CTM_REPO), "rev-parse", "HEAD"], text=True
            ).strip(),
            "ctm_checkpoint_sha256": ctm_hash,
            "adapter_sha256": adapter_hash,
            "probe_type": "randomly initialized frozen CTM; engineering smoke only",
        },
        "probe": {"input_dim": int(features.shape[1]), "tick_ids": arrays["tick_ids"].astype(int).tolist()},
        "preprocessing": {"ae_input": "flattened MNIST image scaled to [0,1]", "probe_ids_fixed": True},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/AE_CTM"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=128)
    parser.add_argument("--old-probe-size", type=int, default=30)
    parser.add_argument("--initial-epochs", type=int, default=1)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--probe-calibration-epochs", type=int, default=5)
    args = parser.parse_args()

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train = MNIST(str(args.mnist_root), train=True, download=False)
    test = MNIST(str(args.mnist_root), train=False, download=False)
    train_labels = train.targets.numpy()
    test_labels = test.targets.numpy()
    rng = np.random.default_rng(args.seed)
    stream_ids = {
        label: rng.choice(np.flatnonzero(train_labels == label), args.per_class, replace=False)
        for label in range(5)
    }
    probe_labels = (0, 1, 2)
    quotas = {label: args.old_probe_size // 3 + int(i < args.old_probe_size % 3)
              for i, label in enumerate(probe_labels)}
    probe_ids = np.concatenate([
        rng.choice(np.flatnonzero(test_labels == label), quotas[label], replace=False)
        for label in probe_labels
    ]).astype(np.int64)
    old_images = test.data[torch.from_numpy(probe_ids)]

    device = torch.device(args.device)
    ae = AEAdapter(seed=args.seed, device=str(device))
    fixed_adapter, ctm = initialize_feature_ctm(seed=args.seed, device=device)
    adapter_hash = hashlib.sha256(b"".join(
        tensor.detach().cpu().numpy().tobytes() for tensor in fixed_adapter.state_dict().values()
    )).hexdigest()
    ctm_hash = None
    calibration_losses = []

    checkpoints = []
    stages = [
        ("012_initial", (0, 1, 2), args.initial_epochs),
        ("3_update", (3,), args.update_epochs),
        ("4_update", (4,), args.update_epochs),
    ]
    for stage_idx, (name, labels, epochs) in enumerate(stages):
        ids = np.concatenate([stream_ids[label] for label in labels])
        images = train.data[torch.from_numpy(ids)].reshape(-1, 784).float().div(255.0)
        ae.train_stream_batch(images, epochs=epochs, stage_seed=stage_idx * 100)
        if stage_idx == 0:
            initial_features = ae.encode(images)
            calibration_losses = calibrate_ctm_reconstructor(
                ctm,
                fixed_adapter,
                initial_features,
                epochs=args.probe_calibration_epochs,
                batch_size=64,
                seed=args.seed,
            )
            ctm_hash = hashlib.sha256(b"".join(
                tensor.detach().cpu().numpy().tobytes() for tensor in ctm.state_dict().values()
            )).hexdigest()
        native = ae.native_metrics(old_images)
        features = native["latent"]
        with torch.inference_mode():
            probe_input = fixed_adapter(torch.from_numpy(features).float().to(device)).reshape(-1, 64, 1, 1)
        trace = trace_ctm(ctm, probe_input, DEFAULT_TICK_IDS)
        record = args.output_dir / "records" / "ae" / f"seed_{args.seed}" / f"checkpoint_{stage_idx:03d}"
        write_checkpoint(record, name, probe_ids, features, native, trace, args.seed, ctm_hash, adapter_hash)
        checkpoints.append({
            "checkpoint": name,
            "record": str(record),
            "mean_reconstruction_mse": float(native["mean_reconstruction_mse"]),
            "ctm_z_shape": list(trace["z_history"].shape),
            "sync_out_shape": list(trace["sync_history"]["out"].shape),
            "sync_action_shape": list(trace["sync_history"]["action"].shape),
        })

    run_key = f"seed{args.seed:02d}"
    (args.output_dir / f"summary_{run_key}.json").write_text(json.dumps({
        "kind": "engineering smoke only; CTM calibrated on initial unlabelled AE train features then frozen",
        "seed": args.seed,
        "probe_calibration_epochs": args.probe_calibration_epochs,
        "probe_calibration_mse": calibration_losses,
        "probe_frozen_after_initial_stage": True,
        "checkpoints": checkpoints,
    }, indent=2) + "\n")

    records = load_records(args.output_dir / "records" / "ae" / f"seed_{args.seed}")
    comparisons = [compare_checkpoints(records[0], record, k=min(5, len(probe_ids) - 1))
                   for record in records[1:]]
    rows = []
    for record, comparison in zip(records[1:], comparisons):
        rows.append({
            "checkpoint": record.manifest["checkpoint_id"],
            "feature_drift_mean": comparison["feature_drift_mean"],
            "similarity_triangle_mae": comparison["similarity_triangle_mae"],
            "trajectory_drift_mean": comparison["trajectory_drift_primary_mean"],
            "sync_out_drift_mean": comparison["synchronization_drift_mean"],
            "sync_action_drift_mean": comparison["action_drift_mean"],
            "reconstruction_mse": float(record.native_metrics["mean_reconstruction_mse"]),
        })
    (args.output_dir / f"drift_summary_{run_key}.json").write_text(
        json.dumps({
            "kind": "engineering smoke only; calibrated frozen CTM",
            "calibration_mse": calibration_losses,
            "comparisons": comparisons,
        }, indent=2) + "\n"
    )
    with (args.output_dir / f"drift_summary_{run_key}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    from PIL import Image, ImageDraw

    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1320, 620), "white")
    draw = ImageDraw.Draw(image)
    panels = [
        ("Native reconstruction MSE", "reconstruction_mse", "#3572A5"),
        ("Feature cosine drift", "feature_drift_mean", "#C44E52"),
        ("CTM trajectory drift", "trajectory_drift_mean", "#E18D2D"),
        ("CTM sync out / action drift", None, "#4A8F65"),
    ]
    for index, (title, key, colour) in enumerate(panels):
        left = 65 + (index % 2) * 640
        top = 95 + (index // 2) * 280
        right, bottom = left + 545, top + 205
        draw.text((left, top - 34), title, fill="black")
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        if key is None:
            series = [
                ("out", [row["sync_out_drift_mean"] for row in rows], "#4A8F65"),
                ("action", [row["sync_action_drift_mean"] for row in rows], "#8163A8"),
            ]
        else:
            series = [("", [row[key] for row in rows], colour)]
        all_values = [value for _, values, _ in series for value in values]
        high = max(all_values, default=1.0)
        scale = high or 1.0
        for label, values, line_colour in series:
            points = []
            for i, value in enumerate(values):
                x = left + 35 + i * (right - left - 70) / max(1, len(values) - 1)
                y = bottom - 22 - value / scale * (bottom - top - 45)
                points.append((x, y))
                draw.text((int(x - 18), bottom + 9), rows[i]["checkpoint"], fill="black")
                draw.text((int(x - 20), int(y - 19)), f"{value:.3f}", fill=line_colour)
            if len(points) > 1:
                draw.line(points, fill=line_colour, width=4)
            for x, y in points:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=line_colour)
            if label:
                draw.text((right - 120, top - 30 + (0 if label == "out" else 18)), label, fill=line_colour)
    draw.text((300, 595), "ONE-SEED ENGINEERING SMOKE | INITIAL-FEATURE CALIBRATED CTM | NOT FINAL EVIDENCE", fill="#9B2226")
    image.save(figure_dir / f"ae_ctm_seed{args.seed:02d}_checkpoint_panel.png")
    print(json.dumps(checkpoints, indent=2))


if __name__ == "__main__":
    main()
