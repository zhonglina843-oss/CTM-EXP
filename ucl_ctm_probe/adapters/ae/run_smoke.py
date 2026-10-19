"""Small, offline MNIST class-stream smoke run for the AE-native route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.datasets import MNIST
from adapter import AEAdapter


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_panel(path: Path, checkpoints: list[dict], seed: int) -> None:
    """Dependency-light PNG so the smoke run works in a headless desktop Python."""
    image = Image.new("RGB", (1200, 520), "white")
    draw = ImageDraw.Draw(image)
    panels = [
        ("Fixed old-probe reconstruction", "Mean per-sample MSE", "mean_reconstruction_mse", "#3572A5"),
        ("Latent drift from initial checkpoint", "Mean 1 - cosine", "mean_latent_drift_1_minus_cosine", "#C44E52"),
    ]
    labels = [row["checkpoint"] for row in checkpoints]
    for panel_index, (title, ylabel, key, colour) in enumerate(panels):
        left = 70 + 570 * panel_index
        top, right, bottom = 95, left + 500, 410
        draw.text((left, 30), title, fill="black")
        draw.text((left, 55), ylabel, fill="black")
        draw.line((left, top, left, bottom), fill="black", width=2)
        draw.line((left, bottom, right, bottom), fill="black", width=2)
        values = [float(row[key]) for row in checkpoints]
        high = max(values)
        low = min(values)
        span = high - low or 1.0
        points = []
        for i, value in enumerate(values):
            x = left + 30 + i * (right - left - 60) / max(1, len(values) - 1)
            y = bottom - 25 - (value - low) / span * (bottom - top - 50)
            points.append((x, y))
            draw.text((int(x - 22), bottom + 12), labels[i], fill="black")
            draw.text((int(x - 20), int(y - 20)), f"{value:.4f}", fill=colour)
        if len(points) > 1:
            draw.line(points, fill=colour, width=4)
        for x, y in points:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=colour)
    draw.text((420, 475), f"AE-only MNIST stream smoke (seed {seed})", fill="black")
    image.save(path)


def choose_fixed_probe(labels: np.ndarray, seed: int, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    classes = (0, 1, 2)
    quotas = {label: n // len(classes) + int(i < n % len(classes)) for i, label in enumerate(classes)}
    ids = []
    for label in classes:
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < quotas[label]:
            raise ValueError(f"test split has too few examples of class {label}")
        ids.extend(rng.choice(candidates, size=quotas[label], replace=False).tolist())
    ids = np.asarray(ids, dtype=np.int64)
    return ids, labels[ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-root", type=Path, required=True, help="Existing torchvision MNIST root; download is disabled.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/A2"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--per-class", type=int, default=256, help="Training examples per stream class.")
    parser.add_argument("--old-probe-size", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--initial-epochs", type=int, default=2)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    # download=False is deliberate: this run only uses data already on disk.
    train = MNIST(root=str(args.mnist_root), train=True, download=False)
    test = MNIST(root=str(args.mnist_root), train=False, download=False)
    train_images = train.data
    train_labels = train.targets.numpy()
    test_images = test.data
    test_labels = test.targets.numpy()
    rng = np.random.default_rng(args.seed)

    stream_ids: dict[int, np.ndarray] = {}
    for label in range(5):
        candidates = np.flatnonzero(train_labels == label)
        if len(candidates) < args.per_class:
            raise ValueError(f"train split has too few examples of class {label}")
        stream_ids[label] = rng.choice(candidates, size=args.per_class, replace=False)

    probe_ids, probe_labels = choose_fixed_probe(test_labels, args.seed, args.old_probe_size)
    fixed_probe = test_images[torch.as_tensor(probe_ids, dtype=torch.long)]
    adapter = AEAdapter(seed=args.seed, learning_rate=args.learning_rate)
    model = adapter.model

    run_dir = args.output_dir
    run_key = f"ae_seed{args.seed:02d}"
    tables_dir = run_dir / "tables"
    figures_dir = run_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_rows: list[dict] = []
    sample_rows: list[dict] = []
    training_rows: list[dict] = []
    baseline_z = None
    stream_seen = 0
    total_steps = 0

    stages = [("012_initial", (0, 1, 2), args.initial_epochs), ("3_update", (3,), args.update_epochs), ("4_update", (4,), args.update_epochs)]
    for stage_idx, (stage, labels, epochs) in enumerate(stages):
        # Labels end here: they select the controlled stream; only images enter the optimizer.
        stage_ids = np.concatenate([stream_ids[label] for label in labels])
        x_stage = train_images[torch.as_tensor(stage_ids, dtype=torch.long)].reshape(-1, 784).float().div(255.0)
        history = adapter.train_stream_batch(
            x_stage, epochs=epochs, batch_size=args.batch_size, stage_seed=100 * stage_idx
        )
        for row in history:
            total_steps += row["optimizer_steps"]
            training_rows.append({"stage": stage, **row})
        stream_seen += len(x_stage)
        native = adapter.native_metrics(fixed_probe)
        current_z = native["latent"]
        errors = native["reconstruction_mse"]
        if baseline_z is None:
            baseline_z = current_z.copy()
        cosine = np.sum(baseline_z * current_z, axis=1) / (
            np.linalg.norm(baseline_z, axis=1) * np.linalg.norm(current_z, axis=1) + 1e-8
        )
        drift = 1.0 - cosine
        latent_l2 = np.linalg.norm(current_z - baseline_z, axis=1)
        checkpoint_rows.append({
            "checkpoint": stage,
            "stream_samples_seen": stream_seen,
            "optimizer_steps_total": total_steps,
            "mean_reconstruction_mse": float(errors.mean()),
            "std_reconstruction_mse": float(errors.std(ddof=0)),
            "mean_latent_drift_1_minus_cosine": float(drift.mean()),
            "std_latent_drift_1_minus_cosine": float(drift.std(ddof=0)),
            "mean_latent_l2_drift": float(latent_l2.mean()),
        })
        for i, (sample_id, label, error, sample_drift, sample_l2) in enumerate(
            zip(probe_ids, probe_labels, errors, drift, latent_l2)
        ):
            sample_rows.append({
                "checkpoint": stage,
                "probe_order": i,
                "test_sample_id": int(sample_id),
                "offline_eval_label": int(label),
                "reconstruction_mse": float(error),
                "latent_drift_1_minus_cosine_from_initial": float(sample_drift),
                "latent_l2_drift_from_initial": float(sample_l2),
            })

    write_csv(tables_dir / f"{run_key}_checkpoints.csv", checkpoint_rows)
    write_csv(tables_dir / f"{run_key}_per_sample.csv", sample_rows)
    write_csv(tables_dir / f"{run_key}_training.csv", training_rows)

    # Re-evaluate the unchanged final checkpoint twice; this is the native-AE
    # analogue of the deterministic probe check before CTM is attached.
    repeat_a = adapter.native_metrics(fixed_probe)
    repeat_b = adapter.native_metrics(fixed_probe)
    determinism = {
        "final_checkpoint": "4_update",
        "max_abs_latent_difference": float(np.max(np.abs(repeat_a["latent"] - repeat_b["latent"]))),
        "max_abs_reconstruction_difference": float(
            np.max(np.abs(repeat_a["reconstruction_mse"] - repeat_b["reconstruction_mse"]))
        ),
    }
    (tables_dir / f"{run_key}_summary.json").write_text(json.dumps({
        "route": adapter.name,
        "ctm_attached": False,
        "seed": args.seed,
        "stream": ["0/1/2 initial", "3 update", "4 update"],
        "checkpoints": checkpoint_rows,
        "determinism": determinism,
    }, indent=2), encoding="utf-8")

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{run_key}_manifest.json").write_text(json.dumps({
        "route": "AE-native MNIST smoke; no CTM attached",
        "seed": args.seed,
        "architecture": "784-256-64-256-784",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "loss": "per-example mean squared reconstruction error",
        "device": "cpu",
        "stream_order": ["0/1/2 initial", "3 update", "4 update"],
        "per_class_training_examples": args.per_class,
        "initial_epochs": args.initial_epochs,
        "update_epochs": args.update_epochs,
        "batch_size": args.batch_size,
        "fixed_old_probe_size": args.old_probe_size,
        "fixed_old_probe_split": "MNIST test; labels only used for balanced selection/offline report",
        "training_labels_entered_model": False,
        "mnist_root": str(args.mnist_root),
        "mnist_train_images_sha256": sha256(args.mnist_root / "MNIST" / "raw" / "train-images-idx3-ubyte"),
        "mnist_train_labels_sha256": sha256(args.mnist_root / "MNIST" / "raw" / "train-labels-idx1-ubyte"),
        "mnist_test_images_sha256": sha256(args.mnist_root / "MNIST" / "raw" / "t10k-images-idx3-ubyte"),
        "mnist_test_labels_sha256": sha256(args.mnist_root / "MNIST" / "raw" / "t10k-labels-idx1-ubyte"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torchvision": __import__("torchvision").__version__,
        "deterministic_algorithms": True,
        "device_cuda_available": torch.cuda.is_available(),
    }, indent=2), encoding="utf-8")

    draw_panel(figures_dir / f"{run_key}_checkpoint_panel.png", checkpoint_rows, args.seed)

    print(json.dumps({"checkpoints": checkpoint_rows, "steps": total_steps}, indent=2))


if __name__ == "__main__":
    main()
