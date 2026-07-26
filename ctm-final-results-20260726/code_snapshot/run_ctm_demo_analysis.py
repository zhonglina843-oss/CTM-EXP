from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from ctm_analysis.convergence import compute_convergence_metrics
from ctm_analysis.probe import (
    active_full_sync_history,
    checkpoint_metadata,
    downsample_full_sync,
    load_checkpoint_model,
    load_hub_model,
    paper_full_sync,
    run_probe,
    select_active_neurons,
    synchronization_pair_indices,
)
from ctm_analysis.visualize import (
    plot_active_sync,
    plot_query_overview,
    plot_sample_summary,
    plot_sync_overview,
    save_active_sync_gif,
    save_query_gif,
    save_sync_triptych_gif,
)
from tasks.image_classification.imagenet_classes import IMAGENET2012_CLASSES


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ticks(raw: str, maximum: int) -> list[int]:
    ticks = {int(value) for value in raw.split(",") if value.strip()}
    ticks.add(maximum)
    return sorted(tick for tick in ticks if 1 <= tick <= maximum)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_aggregate(rows: list[dict], output_path: Path) -> None:
    keys = [
        "label_stable_tick",
        "probability_js_tick",
        "action_step_plateau_tick",
        "out_step_plateau_tick",
        "full_sync_step_plateau_tick",
    ]
    labels = ["Label", "Probability", "S_action", "S_out", "S_full"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    data = [np.asarray([row[key] for row in rows]) for key in keys]
    axes[0].boxplot(data, tick_labels=labels, showmeans=True)
    axes[0].set_ylabel("Convergence tick")
    axes[0].set_title("Convergence definitions")
    for values, label in zip(data, labels):
        ordered = np.sort(values)
        axes[1].step(ordered, np.arange(1, len(values) + 1) / len(values), where="post", label=label)
    axes[1].set(xlabel="Convergence tick", ylabel="Fraction converged", ylim=(0, 1.02))
    axes[1].legend()
    axes[1].set_title("Empirical CDF")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--model-id")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--inference-iterations", type=int, default=50)
    parser.add_argument("--matrix-ticks", default="1,5,10,15,20,30,50")
    parser.add_argument("--save-full-final", action="store_true")
    parser.add_argument("--active-neurons", type=int, default=512)
    parser.add_argument("--active-gif-neurons", type=int, default=256)
    parser.add_argument("--no-active-gif", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.checkpoint:
        model, checkpoint_args = load_checkpoint_model(args.checkpoint, args.device)
    else:
        model = load_hub_model(args.model_id, args.device, args.model_cache_dir)
        checkpoint_args = argparse.Namespace()
    trained_iterations = model.iterations
    model.iterations = args.inference_iterations
    matrix_ticks = parse_ticks(args.matrix_ticks, args.inference_iterations)

    transform = transforms.Compose(
        [transforms.Resize(256), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )
    class_names = list(IMAGENET2012_CLASSES.values())
    image_paths = sorted(
        [path for path in args.input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )[: args.max_samples]
    if not image_paths:
        raise RuntimeError(f"No images found in {args.input_dir}")

    manifest = {
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": sha256(args.checkpoint) if args.checkpoint else None,
        "model_id": args.model_id,
        "trained_iterations": trained_iterations,
        "inference_iterations": args.inference_iterations,
        "matrix_ticks": matrix_ticks,
        "active_neuron_selection": {
            "score": "RMS post-activation over the complete observed tick trajectory",
            "numeric_top_k": args.active_neurons,
            "gif_top_k": args.active_gif_neurons,
            "fixed_across_ticks": True,
        },
        "model": checkpoint_metadata(model, checkpoint_args),
        "input_note": "Website video first-frame crops; MP4 compression may differ from raw ImageNet pixels.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_rows = []
    for image_path in image_paths:
        sample_id = image_path.stem
        sample_dir = args.output_dir / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        pil_image = Image.open(image_path).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(args.device)
        probe = run_probe(model, input_tensor)
        metrics, series = compute_convergence_metrics(
            probe.predictions,
            probe.certainties,
            probe.sync_action,
            probe.sync_out,
            probe.post_activations,
        )
        row = {"sample_id": sample_id, **metrics}
        summary_rows.append(row)

        np.savez_compressed(
            sample_dir / "traces.npz",
            predictions=probe.predictions,
            certainties=probe.certainties,
            sync_action=probe.sync_action,
            sync_out=probe.sync_out,
            queries=probe.queries,
            pre_activations=probe.pre_activations,
            post_activations=probe.post_activations,
            attention=probe.attention,
            **series,
        )
        np.save(sample_dir / "query_all_ticks.npy", probe.queries)

        action_left, action_right = synchronization_pair_indices(model, "action")
        out_left, out_right = synchronization_pair_indices(model, "out")
        np.savez_compressed(
            sample_dir / "sync_pair_indices.npz",
            action_left=action_left,
            action_right=action_right,
            out_left=out_left,
            out_right=out_right,
        )

        active_indices, activity = select_active_neurons(probe.post_activations, args.active_neurons)
        active_history = active_full_sync_history(probe.post_activations, active_indices)
        np.save(sample_dir / "active_neuron_indices.npy", active_indices)
        np.savez_compressed(sample_dir / "active_neuron_statistics.npz", **activity)
        np.savez_compressed(sample_dir / "S_full_active_all_ticks.npz", matrices=active_history)
        active_order = plot_active_sync(
            active_history,
            active_indices,
            sample_dir / "S_full_active_exact.png",
        )
        np.save(sample_dir / "active_neuron_cluster_order.npy", active_order)
        plot_query_overview(probe.queries, sample_dir / "query_overview.png")
        save_query_gif(probe.queries, sample_dir / "query.gif")
        if not args.no_active_gif:
            gif_count = min(args.active_gif_neurons, len(active_indices))
            gif_indices = active_indices[:gif_count]
            gif_history = active_full_sync_history(probe.post_activations, gif_indices)
            gif_order = plot_active_sync(
                gif_history,
                gif_indices,
                sample_dir / "S_full_active_gif_subset.png",
            )
            save_active_sync_gif(
                gif_history,
                gif_indices,
                sample_dir / "S_full_active.gif",
                gif_order,
            )
            save_sync_triptych_gif(
                probe.sync_action,
                probe.sync_out,
                gif_history,
                sample_dir / "S_action_S_out_S_full.gif",
                model.neuron_select_type,
                model.n_synch_action,
                model.n_synch_out,
                gif_order,
            )
        (sample_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

        final_small = downsample_full_sync(probe.post_activations, args.inference_iterations)
        np.save(sample_dir / "S_full_final_256x256.npy", final_small)
        if args.save_full_final:
            final_full = paper_full_sync(probe.post_activations, args.inference_iterations)
            np.save(sample_dir / "S_full_final_4096x4096.npy", final_full)
        for tick in matrix_ticks:
            matrix_small = downsample_full_sync(probe.post_activations, tick)
            np.save(sample_dir / f"S_full_tick_{tick:03d}_256x256.npy", matrix_small)

        image_array = np.asarray(pil_image)
        plot_sample_summary(
            image_array,
            probe.probabilities,
            probe.certainties,
            probe.attention,
            class_names,
            metrics,
            sample_dir / "result_summary.png",
            matrix_ticks,
            probe.feature_shape,
        )
        plot_sync_overview(
            probe.sync_action,
            probe.sync_out,
            final_small,
            series,
            model.neuron_select_type,
            model.n_synch_action,
            model.n_synch_out,
            sample_dir / "synchronization_overview.png",
        )

    save_csv(args.output_dir / "convergence_summary.csv", summary_rows)
    plot_aggregate(summary_rows, args.output_dir / "aggregate_convergence.png")


if __name__ == "__main__":
    main()
