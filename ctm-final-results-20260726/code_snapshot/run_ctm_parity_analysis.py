from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from ctm_analysis.convergence import compute_sync_convergence_metrics, first_sustained
from ctm_analysis.probe import (
    active_full_sync_history,
    select_active_neurons,
    synchronization_pair_indices,
)
from ctm_analysis.visualize import active_neuron_order, plot_query_overview, save_query_gif, save_sync_triptych_gif
from models.utils import get_model_args_from_checkpoint, load_checkpoint, reshape_predictions
from tasks.parity.utils import prepare_model


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parity_metrics(labels: np.ndarray, target: np.ndarray) -> tuple[dict, dict[str, np.ndarray]]:
    final = labels[-1]
    final_match = np.all(labels == final[None], axis=1)
    correct = labels == target[None]
    accuracy = correct.mean(axis=1)
    fully_correct = correct.all(axis=1)
    token_ticks = np.asarray(
        [first_sustained(labels[:, i] == final[i]) for i in range(labels.shape[1])], dtype=np.int32
    )
    return {
        "final_sequence_stable_tick": first_sustained(final_match),
        "full_correct_sustained_tick": first_sustained(fully_correct, missing=0),
        "accuracy_99_tick": first_sustained(accuracy >= 0.99, missing=0),
        "mean_token_stable_tick": float(token_ticks.mean()),
        "final_accuracy": float(accuracy[-1]),
    }, {"accuracy": accuracy.astype(np.float32), "token_stable_ticks": token_ticks}


def save_result_gif(
    input_bits: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    certainties: np.ndarray,
    output_path: Path,
) -> None:
    frames = []
    labels = probabilities.argmax(axis=1)
    for tick in range(len(probabilities)):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        axes[0].imshow(input_bits.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[0].set_title("Input bits")
        axes[1].imshow(target.reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Target cumulative parity")
        axes[2].imshow(probabilities[tick, 1].reshape(8, 8), cmap="viridis", vmin=0, vmax=1)
        axes[2].set_title("P(parity = 1)")
        axes[3].imshow((labels[tick] == target).reshape(8, 8), cmap="RdYlGn", vmin=0, vmax=1)
        axes[3].set_title(f"Correct: {(labels[tick] == target).mean():.1%}")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Parity | tick {tick + 1}/{len(probabilities)} | certainty {certainties[tick]:.3f}")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(output_path, frames, fps=10, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--detailed-samples", type=int, default=3)
    parser.add_argument("--active-neurons", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(args.checkpoint, args.device)
    model_args = get_model_args_from_checkpoint(checkpoint)
    model = prepare_model([model_args.parity_sequence_length, 2], model_args, args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    inputs = torch.randint(0, 2, (args.samples, model_args.parity_sequence_length), device=args.device).float()
    signed_inputs = inputs * 2 - 1
    targets = torch.cumsum((signed_inputs == -1).long(), dim=1) % 2
    with torch.inference_mode():
        predictions, certainties, syncs, pre, post, attention = model(signed_inputs, track=True)
    reshaped = reshape_predictions(predictions, [model_args.parity_sequence_length, 2])
    probabilities = torch.softmax(reshaped, dim=2).detach().cpu().numpy().transpose(0, 3, 2, 1)
    certainties_np = certainties[:, 1].detach().cpu().numpy()
    sync_out_all, sync_action_all = syncs
    sync_action_all = np.asarray(sync_action_all)
    sync_out_all = np.asarray(sync_out_all)
    post = np.asarray(post)
    with torch.inference_mode():
        query_input = torch.from_numpy(sync_action_all.reshape(-1, sync_action_all.shape[-1]).astype(np.float32)).to(args.device)
        queries_all = model.q_proj(query_input).detach().cpu().numpy().reshape(
            sync_action_all.shape[0], sync_action_all.shape[1], -1
        ).astype(np.float32)

    rows = []
    sample_series = []
    for sample in range(args.samples):
        labels = probabilities[sample].argmax(axis=1)
        task_metrics, task_series = parity_metrics(labels, targets[sample].cpu().numpy())
        sync_metrics, sync_series = compute_sync_convergence_metrics(
            sync_action_all[:, sample], sync_out_all[:, sample], post[:, sample]
        )
        row = {"sample_id": sample, **task_metrics, **sync_metrics}
        rows.append(row)
        sample_series.append((task_series, sync_series))
    save_rows(args.output_dir / "convergence_summary.csv", rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps({
            "task": "parity",
            "checkpoint": str(args.checkpoint),
            "samples": args.samples,
            "iterations": int(model_args.iterations),
            "sequence_length": int(model_args.parity_sequence_length),
            "neuron_select_type": model.neuron_select_type,
            "S_full_definition": "unweighted cumulative post-activation Gram",
            "query_definition": "q_proj(S_action) recomputed from tracked S_action with the trained query projection",
        }, indent=2), encoding="utf-8"
    )

    stability = np.asarray([row["final_sequence_stable_tick"] for row in rows])
    chosen = np.unique(np.asarray([0, int(stability.argmin()), int(stability.argmax())]))[: args.detailed_samples]
    action_left, action_right = synchronization_pair_indices(model, "action")
    out_left, out_right = synchronization_pair_indices(model, "out")
    for sample in chosen:
        sample_dir = args.output_dir / "samples" / f"sample-{sample:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        active_indices, activity = select_active_neurons(post[:, sample], args.active_neurons)
        full_history = active_full_sync_history(post[:, sample], active_indices)
        order = active_neuron_order(full_history[-1])
        np.savez_compressed(
            sample_dir / "traces.npz",
            probabilities=probabilities[sample],
            certainties=certainties_np[sample],
            sync_action=sync_action_all[:, sample],
            sync_out=sync_out_all[:, sample],
            queries=queries_all[:, sample],
            pre_activations=np.asarray(pre)[:, sample],
            post_activations=post[:, sample],
            attention=np.asarray(attention)[:, sample],
            target=targets[sample].cpu().numpy(),
            input=signed_inputs[sample].cpu().numpy(),
            **sample_series[sample][0],
            **sample_series[sample][1],
        )
        np.savez_compressed(sample_dir / "S_full_active_all_ticks.npz", matrices=full_history)
        np.save(sample_dir / "query_all_ticks.npy", queries_all[:, sample])
        np.save(sample_dir / "active_neuron_indices.npy", active_indices)
        np.save(sample_dir / "active_neuron_cluster_order.npy", order)
        np.savez_compressed(sample_dir / "active_neuron_statistics.npz", **activity)
        np.savez_compressed(
            sample_dir / "sync_pair_indices.npz",
            action_left=action_left, action_right=action_right, out_left=out_left, out_right=out_right,
        )
        (sample_dir / "metrics.json").write_text(json.dumps(rows[sample], indent=2), encoding="utf-8")
        save_result_gif(
            inputs[sample].cpu().numpy(), targets[sample].cpu().numpy(),
            probabilities[sample], certainties_np[sample], sample_dir / "result.gif",
        )
        plot_query_overview(queries_all[:, sample], sample_dir / "query_overview.png")
        save_query_gif(queries_all[:, sample], sample_dir / "query.gif")
        save_sync_triptych_gif(
            sync_action_all[:, sample], sync_out_all[:, sample], full_history,
            sample_dir / "synchronization_triptych.gif", model.neuron_select_type,
            model.n_synch_action, model.n_synch_out, order,
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(stability, bins=np.arange(0.5, model_args.iterations + 1.5), color="#287271")
    axes[0].set(xlabel="Final-sequence stability tick", ylabel="Samples")
    axes[1].plot(np.arange(1, model_args.iterations + 1), [
        np.mean(stability <= tick) for tick in range(1, model_args.iterations + 1)
    ])
    axes[1].set(xlabel="Tick", ylabel="Fraction stable", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(args.output_dir / "aggregate_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
