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
from ctm_analysis.visualize import active_neuron_order, save_sync_triptych_gif
from models.ctm_sort import ContinuousThoughtMachineSORT
from models.utils import get_model_args_from_checkpoint, load_checkpoint
from tasks.sort.utils import decode_predictions


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_model(model_args, device: str) -> ContinuousThoughtMachineSORT:
    model = ContinuousThoughtMachineSORT(
        iterations=model_args.iterations,
        d_model=model_args.d_model,
        d_input=model_args.out_dims - 1,
        heads=model_args.heads,
        n_synch_out=model_args.n_synch_out,
        n_synch_action=getattr(model_args, "n_synch_action", 0),
        synapse_depth=model_args.synapse_depth,
        memory_length=model_args.memory_length,
        deep_nlms=model_args.deep_memory,
        memory_hidden_dims=model_args.memory_hidden_dims,
        do_layernorm_nlm=model_args.do_normalisation,
        backbone_type="none",
        positional_embedding_type="none",
        out_dims=model_args.out_dims,
        prediction_reshaper=[-1],
        dropout=getattr(model_args, "dropout", 0.0),
        dropout_nlm=getattr(model_args, "dropout_nlm", None),
        neuron_select_type=model_args.neuron_select_type,
        n_random_pairing_self=getattr(model_args, "n_random_pairing_self", 0),
    ).to(device)
    return model


def padded_decodes_by_tick(predictions: torch.Tensor, target_length: int, blank_label: int) -> tuple[np.ndarray, np.ndarray]:
    decoded = np.full((predictions.shape[-1], target_length), blank_label, dtype=np.int64)
    lengths = np.zeros(predictions.shape[-1], dtype=np.int64)
    for tick in range(1, predictions.shape[-1] + 1):
        sequence = decode_predictions(predictions[:, :, :tick], blank_label=blank_label)[0].detach().cpu().numpy()
        lengths[tick - 1] = min(len(sequence), target_length)
        decoded[tick - 1, : lengths[tick - 1]] = sequence[:target_length]
    return decoded, lengths


def sort_metrics(decoded_by_tick: np.ndarray, decoded_lengths: np.ndarray, target: np.ndarray, blank_label: int) -> tuple[dict, dict]:
    target_length = len(target)
    exact = (decoded_lengths == target_length) & np.all(decoded_by_tick == target[None, :], axis=1)
    token_acc = (decoded_by_tick == target[None, :]).mean(axis=1)
    final = decoded_by_tick[-1]
    final_len = int(decoded_lengths[-1])
    final_match = (decoded_lengths == final_len) & np.all(decoded_by_tick == final[None, :], axis=1)
    return {
        "final_exact": int(exact[-1]),
        "final_token_accuracy": float(token_acc[-1]),
        "final_decoded_length": final_len,
        "first_exact_sustained_tick": first_sustained(exact, missing=0),
        "token_accuracy_99_tick": first_sustained(token_acc >= 0.99, missing=0),
        "final_decoded_stable_tick": first_sustained(final_match),
    }, {
        "exact_by_tick": exact.astype(np.bool_),
        "token_accuracy_by_tick": token_acc.astype(np.float32),
        "decoded_by_tick": decoded_by_tick.astype(np.int64),
        "decoded_lengths": decoded_lengths.astype(np.int64),
    }


def save_sort_result_gif(
    values: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    decoded_by_tick: np.ndarray,
    decoded_lengths: np.ndarray,
    token_accuracy: np.ndarray,
    output_path: Path,
) -> None:
    frames = []
    ticks = probabilities.shape[0]
    blank = probabilities.shape[1] - 1
    target_values = values[target]
    for tick in range(ticks):
        decoded = decoded_by_tick[tick, : decoded_lengths[tick]]
        decoded_values = values[decoded] if len(decoded) else np.empty(0)
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
        axes[0].bar(np.arange(len(values)), values, color="#8ecae6")
        axes[0].set_title("Input real values")
        axes[1].plot(np.arange(len(target_values)), target_values, marker="o", color="#2a9d8f")
        axes[1].set_title("Target sorted values")
        if len(decoded_values):
            axes[2].plot(np.arange(len(decoded_values)), decoded_values, marker="o", color="#e76f51")
        axes[2].set_xlim(-0.5, len(values) - 0.5)
        axes[2].set_title(f"CTC decoded prefix len={len(decoded)}")
        axes[3].plot(np.arange(1, ticks + 1), token_accuracy, color="#264653")
        axes[3].axvline(tick + 1, color="black", alpha=0.45)
        axes[3].set_ylim(0, 1.02)
        axes[3].set_title("Token accuracy vs target")
        for ax in axes:
            ax.grid(alpha=0.25)
        pred_tokens = " ".join(map(str, decoded[:10]))
        if len(decoded) > 10:
            pred_tokens += " ..."
        fig.suptitle(
            f"Sorting | tick {tick + 1}/{ticks} | token acc={token_accuracy[tick]:.2f} | "
            f"blank={blank} | decoded: {pred_tokens}"
        )
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(args.checkpoint, args.device)
    model_args = get_model_args_from_checkpoint(checkpoint)
    model = build_model(model_args, args.device)
    with torch.inference_mode():
        model(torch.zeros(1, model_args.N_to_sort, device=args.device))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    inputs = torch.randn(args.samples, model_args.N_to_sort, device=args.device)
    targets = torch.argsort(inputs, dim=1)
    with torch.inference_mode():
        predictions, certainties, sync_out_all, pre, post, _ = model(inputs, track=True)

    blank_label = predictions.shape[1] - 1
    probabilities = torch.softmax(predictions, dim=1).detach().cpu().numpy().transpose(0, 2, 1)
    certainties_np = certainties[:, 1].detach().cpu().numpy()
    sync_out_all = np.asarray(sync_out_all, dtype=np.float32)
    post = np.asarray(post, dtype=np.float32)
    pre = np.asarray(pre, dtype=np.float32)
    inputs_np = inputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()

    rows = []
    sample_series = []
    decoded_cache = []
    for sample in range(args.samples):
        decoded_by_tick, decoded_lengths = padded_decodes_by_tick(
            predictions[sample : sample + 1].detach(), model_args.N_to_sort, blank_label
        )
        task_metrics, task_series = sort_metrics(decoded_by_tick, decoded_lengths, targets_np[sample], blank_label)
        sync_metrics, sync_series = compute_sync_convergence_metrics(None, sync_out_all[:, sample], post[:, sample])
        rows.append({"sample_id": sample, **task_metrics, **sync_metrics})
        sample_series.append((task_series, sync_series))
        decoded_cache.append((decoded_by_tick, decoded_lengths))

    save_rows(args.output_dir / "convergence_summary.csv", rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "task": "sorting",
                "checkpoint": str(args.checkpoint),
                "samples": args.samples,
                "iterations": int(model_args.iterations),
                "N_to_sort": int(model_args.N_to_sort),
                "d_model": int(model_args.d_model),
                "neuron_select_type": model.neuron_select_type,
                "S_action": "not present in ContinuousThoughtMachineSORT (q_proj=None, n_synch_action=0)",
                "query": "not present in ContinuousThoughtMachineSORT",
                "S_full_definition": "unweighted cumulative post-activation Gram",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    exact_ticks = np.asarray([row["first_exact_sustained_tick"] for row in rows])
    token_ticks = np.asarray([row["token_accuracy_99_tick"] for row in rows])
    chosen_scores = np.where(exact_ticks > 0, exact_ticks, token_ticks)
    chosen = np.unique(np.asarray([0, int(chosen_scores.argmin()), int(chosen_scores.argmax())]))[: args.detailed_samples]
    out_left, out_right = synchronization_pair_indices(model, "out")
    for sample in chosen:
        sample_dir = args.output_dir / "samples" / f"sample-{sample:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        active_indices, activity = select_active_neurons(post[:, sample], args.active_neurons)
        full_history = active_full_sync_history(post[:, sample], active_indices)
        order = active_neuron_order(full_history[-1])
        decoded_by_tick, decoded_lengths = decoded_cache[sample]
        np.savez_compressed(
            sample_dir / "traces.npz",
            probabilities=probabilities[sample],
            certainties=certainties_np[sample],
            predictions=predictions[sample].detach().cpu().numpy(),
            sync_action=np.empty((model_args.iterations, 0), dtype=np.float32),
            sync_out=sync_out_all[:, sample],
            queries=np.empty((model_args.iterations, 0), dtype=np.float32),
            pre_activations=pre[:, sample],
            post_activations=post[:, sample],
            input_values=inputs_np[sample],
            target_indices=targets_np[sample],
            blank_label=np.asarray(blank_label),
            **sample_series[sample][0],
            **sample_series[sample][1],
        )
        np.savez_compressed(sample_dir / "S_full_active_all_ticks.npz", matrices=full_history)
        np.save(sample_dir / "active_neuron_indices.npy", active_indices)
        np.save(sample_dir / "active_neuron_cluster_order.npy", order)
        np.savez_compressed(sample_dir / "active_neuron_statistics.npz", **activity)
        np.savez_compressed(
            sample_dir / "sync_pair_indices.npz",
            action_left=np.empty(0, dtype=np.int64),
            action_right=np.empty(0, dtype=np.int64),
            out_left=out_left,
            out_right=out_right,
        )
        (sample_dir / "metrics.json").write_text(json.dumps(rows[sample], indent=2), encoding="utf-8")
        save_sort_result_gif(
            inputs_np[sample],
            targets_np[sample],
            probabilities[sample],
            decoded_by_tick,
            decoded_lengths,
            sample_series[sample][0]["token_accuracy_by_tick"],
            sample_dir / "result.gif",
        )
        save_sync_triptych_gif(
            None,
            sync_out_all[:, sample],
            full_history,
            sample_dir / "synchronization_triptych.gif",
            model.neuron_select_type,
            0,
            model.n_synch_out,
            order,
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    valid = exact_ticks[exact_ticks > 0]
    axes[0].hist(valid, bins=np.arange(0.5, model_args.iterations + 1.5), color="#287271")
    axes[0].set(xlabel="Exact decoded-list sustained tick", ylabel="Samples")
    axes[1].plot(np.arange(1, model_args.iterations + 1), [
        np.mean((exact_ticks > 0) & (exact_ticks <= tick)) for tick in range(1, model_args.iterations + 1)
    ], label="exact list")
    axes[1].plot(np.arange(1, model_args.iterations + 1), [
        np.mean((token_ticks > 0) & (token_ticks <= tick)) for tick in range(1, model_args.iterations + 1)
    ], label="token acc >= 99%")
    axes[1].set(xlabel="Tick", ylabel="Fraction converged", ylim=(0, 1.02))
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "aggregate_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
