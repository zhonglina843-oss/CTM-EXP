from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset

from ctm_analysis.convergence import compute_sync_convergence_metrics, first_sustained
from ctm_analysis.probe import (
    active_full_sync_history,
    load_hub_model,
    select_active_neurons,
    synchronization_pair_indices,
)
from ctm_analysis.visualize import active_neuron_order, plot_query_overview, save_query_gif, save_sync_triptych_gif
from tasks.mazes.analysis.run import has_solved_checker
from tasks.mazes.plotting import make_maze_gif


def collate(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, list[np.ndarray]]:
    arrays = [np.asarray(item["image"], dtype=np.float32) / 255.0 for item in batch]
    images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2)
    targets = torch.tensor(np.stack([item["solution_path"] for item in batch]), dtype=torch.long)
    return images, targets, arrays


def maze_metrics(labels: np.ndarray, target: np.ndarray, image: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    final = labels[-1]
    final_match = np.all(labels == final[None], axis=1)
    correct = labels == target[None]
    step_accuracy = correct.mean(axis=1)
    exact = correct.all(axis=1)
    solved = np.asarray([has_solved_checker(image, route, valid_only=True)[0] for route in labels])
    return {
        "final_route_stable_tick": first_sustained(final_match),
        "exact_route_tick": first_sustained(exact, missing=0),
        "navigation_solved_tick": first_sustained(solved, missing=0),
        "step_accuracy_99_tick": first_sustained(step_accuracy >= 0.99, missing=0),
        "final_step_accuracy": float(step_accuracy[-1]),
        "final_navigation_solved": int(solved[-1]),
    }, {
        "step_accuracy": step_accuracy.astype(np.float32),
        "navigation_solved": solved.astype(np.uint8),
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="SakanaAI/ctm-maze-large")
    parser.add_argument("--dataset-id", default="SakanaAI/mazes-large")
    parser.add_argument("--model-cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--detailed-samples", type=int, default=3)
    parser.add_argument("--active-neurons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset_id, split=f"test[:{args.samples}]")
    images, targets, image_arrays = collate([dataset[index] for index in range(len(dataset))])
    model = load_hub_model(args.model_id, args.device, args.model_cache_dir)
    model.eval()
    with torch.inference_mode():
        predictions, certainties, syncs, pre, post, attention = model(images.to(args.device), track=True)
    route_logits = predictions.reshape(predictions.shape[0], -1, 5, predictions.shape[-1])
    probabilities = torch.softmax(route_logits, dim=2).detach().cpu().numpy().transpose(0, 3, 1, 2)
    labels = probabilities.argmax(axis=3)
    certainty = certainties[:, 1].detach().cpu().numpy()
    sync_out, sync_action = (np.asarray(value) for value in syncs)
    pre, post, attention = np.asarray(pre), np.asarray(post), np.asarray(attention)
    targets_np = targets.numpy()
    with torch.inference_mode():
        query_input = torch.from_numpy(sync_action.reshape(-1, sync_action.shape[-1]).astype(np.float32)).to(args.device)
        queries = model.q_proj(query_input).detach().cpu().numpy().reshape(
            sync_action.shape[0], sync_action.shape[1], -1
        ).astype(np.float32)

    rows: list[dict] = []
    all_series = []
    for sample in range(len(dataset)):
        task_metrics, task_series = maze_metrics(labels[sample], targets_np[sample], image_arrays[sample])
        sync_metrics, sync_series = compute_sync_convergence_metrics(
            sync_action[:, sample], sync_out[:, sample], post[:, sample]
        )
        rows.append({"sample_id": sample, **task_metrics, **sync_metrics})
        all_series.append((task_series, sync_series))
    write_rows(args.output_dir / "convergence_summary.csv", rows)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "task": "maze-large",
        "model_id": args.model_id,
        "dataset_id": args.dataset_id,
        "samples": len(dataset),
        "iterations": int(model.iterations),
        "d_model": int(model.d_model),
        "neuron_select_type": model.neuron_select_type,
        "S_full_definition": "unweighted cumulative post-activation Gram",
        "query_definition": "q_proj(S_action) recomputed from tracked S_action with the trained query projection",
    }, indent=2), encoding="utf-8")

    stability = np.asarray([row["final_route_stable_tick"] for row in rows])
    chosen = np.unique([0, int(stability.argmin()), int(stability.argmax())])[: args.detailed_samples]
    action_left, action_right = synchronization_pair_indices(model, "action")
    out_left, out_right = synchronization_pair_indices(model, "out")
    attention_shape = model.kv_features.shape[-2:]
    reshaped_attention = attention.reshape(
        attention.shape[0], attention.shape[1], -1, attention_shape[0], attention_shape[1]
    )
    for sample in chosen:
        sample_dir = args.output_dir / "samples" / f"sample-{sample:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        active_indices, activity = select_active_neurons(post[:, sample], args.active_neurons)
        full_history = active_full_sync_history(post[:, sample], active_indices)
        order = active_neuron_order(full_history[-1])
        np.savez_compressed(
            sample_dir / "traces.npz",
            probabilities=probabilities[sample], certainty=certainty[sample],
            sync_action=sync_action[:, sample], sync_out=sync_out[:, sample],
            queries=queries[:, sample],
            pre_activations=pre[:, sample], post_activations=post[:, sample],
            attention=reshaped_attention[:, sample], target=targets_np[sample],
            image=image_arrays[sample], **all_series[sample][0], **all_series[sample][1],
        )
        np.savez_compressed(sample_dir / "S_full_active_all_ticks.npz", matrices=full_history)
        np.save(sample_dir / "query_all_ticks.npy", queries[:, sample])
        np.save(sample_dir / "active_neuron_indices.npy", active_indices)
        np.save(sample_dir / "active_neuron_cluster_order.npy", order)
        np.savez_compressed(sample_dir / "active_neuron_statistics.npz", **activity)
        np.savez_compressed(
            sample_dir / "sync_pair_indices.npz", action_left=action_left, action_right=action_right,
            out_left=out_left, out_right=out_right,
        )
        (sample_dir / "metrics.json").write_text(json.dumps(rows[sample], indent=2), encoding="utf-8")
        make_maze_gif(
            images[sample].numpy(),
            route_logits[sample].detach().cpu().numpy(),
            targets_np[sample],
            reshaped_attention[:, sample],
            str(sample_dir),
        )
        plot_query_overview(queries[:, sample], sample_dir / "query_overview.png")
        save_query_gif(queries[:, sample], sample_dir / "query.gif")
        save_sync_triptych_gif(
            sync_action[:, sample], sync_out[:, sample], full_history,
            sample_dir / "synchronization_triptych.gif", model.neuron_select_type,
            model.n_synch_action, model.n_synch_out, order,
        )

    ticks = np.arange(1, model.iterations + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(stability, bins=np.arange(0.5, model.iterations + 1.5), color="#287271")
    axes[0].set(xlabel="Final-route stability tick", ylabel="Samples")
    axes[1].plot(ticks, [np.mean(stability <= tick) for tick in ticks])
    axes[1].set(xlabel="Tick", ylabel="Fraction stable", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(args.output_dir / "aggregate_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
