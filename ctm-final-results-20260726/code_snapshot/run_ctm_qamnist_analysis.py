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
from models.utils import get_model_args_from_checkpoint, load_checkpoint
from tasks.qamnist.utils import get_dataset, prepare_model
from utils.samplers import QAMNISTSampler


def save_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ensure_qamnist_args(model_args) -> None:
    model_args.model_type = getattr(model_args, "model_type", "ctm")
    model_args.out_dims = getattr(model_args, "out_dims", 10)
    model_args.use_most_certain = getattr(model_args, "use_most_certain", True)


def qamnist_batch(model_args, samples: int, device: str):
    _, test_data, _, _, _ = get_dataset(
        model_args.q_num_images,
        model_args.q_num_images_delta,
        model_args.q_num_repeats_per_input,
        model_args.q_num_operations,
        model_args.q_num_operations_delta,
    )
    sampler = QAMNISTSampler(test_data, batch_size=samples)
    loader = torch.utils.data.DataLoader(test_data, num_workers=0, batch_sampler=sampler)
    x, question, question_readable, target = next(iter(loader))
    if isinstance(question, (list, tuple)):
        question = torch.stack(list(question), dim=1)
    z = question.unsqueeze(2) if question.ndim == 2 else question
    return x.to(device), z.to(device), list(question_readable), target.to(device)


def trace_qamnist_forward(model, x: torch.Tensor, z: torch.Tensor) -> dict[str, object]:
    B = x.size(0)
    device = x.device
    total_iterations_for_digits = x.size(1)
    total_iterations_for_question = z.size(1)
    total_iterations = total_iterations_for_digits + total_iterations_for_question + model.iterations_for_answering

    state_trace = model.start_trace.unsqueeze(0).expand(B, -1, -1)
    activated_state = model.start_activated_state.unsqueeze(0).expand(B, -1)

    predictions = torch.empty(B, model.out_dims, total_iterations, device=device, dtype=x.dtype)
    certainties = torch.empty(B, 2, total_iterations, device=device, dtype=x.dtype)

    decay_alpha_action, decay_beta_action = None, None
    model.decay_params_action.data = torch.clamp(model.decay_params_action, 0, 15)
    model.decay_params_out.data = torch.clamp(model.decay_params_out, 0, 15)
    r_action = torch.exp(-model.decay_params_action).unsqueeze(0).repeat(B, 1)
    r_out = torch.exp(-model.decay_params_out).unsqueeze(0).repeat(B, 1)
    _, decay_alpha_out, decay_beta_out = model.compute_synchronisation(
        activated_state, None, None, r_out, synch_type="out"
    )

    prev_input = None
    prev_kv = None
    sync_action_tracking = []
    sync_out_tracking = []
    query_tracking = []
    pre_activations_tracking = []
    post_activations_tracking = []
    attention_by_tick = []
    step_types = []

    for stepi in range(total_iterations):
        is_digit_step, is_question_step, is_answer_step = model.determine_step_type(
            total_iterations_for_digits, total_iterations_for_question, stepi
        )
        kv, prev_input = model.get_kv_for_step(
            total_iterations_for_digits, total_iterations_for_question, stepi, x, z, prev_input, prev_kv
        )
        prev_kv = kv

        synchronization_action, decay_alpha_action, decay_beta_action = model.compute_synchronisation(
            activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type="action"
        )
        q = model.q_proj(synchronization_action)
        attn_weights = None
        if is_digit_step:
            attn_out, attn_weights = model.attention(
                q.unsqueeze(1), kv, kv, average_attn_weights=False, need_weights=True
            )
            pre_synapse_input = torch.concatenate((attn_out.squeeze(1), activated_state), dim=-1)
        else:
            pre_synapse_input = torch.concatenate((kv.squeeze(1), activated_state), dim=-1)

        state = model.synapses(pre_synapse_input)
        state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
        activated_state = model.trace_processor(state_trace)

        synchronization_out, decay_alpha_out, decay_beta_out = model.compute_synchronisation(
            activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type="out"
        )
        current_prediction = model.output_projector(synchronization_out)
        current_certainty = model.compute_certainty(current_prediction)

        predictions[..., stepi] = current_prediction
        certainties[..., stepi] = current_certainty
        sync_action_tracking.append(synchronization_action.detach().cpu().numpy())
        sync_out_tracking.append(synchronization_out.detach().cpu().numpy())
        query_tracking.append(q.detach().cpu().numpy())
        pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
        post_activations_tracking.append(activated_state.detach().cpu().numpy())
        if attn_weights is None:
            attention_by_tick.append(None)
        else:
            attention_by_tick.append(attn_weights.detach().cpu().numpy())
        step_types.append("digit" if is_digit_step else ("question" if is_question_step else "answer"))

    return {
        "predictions": predictions.detach().cpu().numpy(),
        "certainties": certainties.detach().cpu().numpy(),
        "sync_action": np.asarray(sync_action_tracking),
        "sync_out": np.asarray(sync_out_tracking),
        "queries": np.asarray(query_tracking),
        "pre_activations": np.asarray(pre_activations_tracking),
        "post_activations": np.asarray(post_activations_tracking),
        "attention_by_tick": attention_by_tick,
        "step_types": np.asarray(step_types),
        "phase_boundaries": {
            "digits_end_tick": int(total_iterations_for_digits),
            "question_end_tick": int(total_iterations_for_digits + total_iterations_for_question),
            "total_ticks": int(total_iterations),
        },
    }


def qamnist_metrics(labels: np.ndarray, target: int, certainties: np.ndarray, answer_start: int) -> tuple[dict, dict]:
    correct = labels == target
    final_label = int(labels[-1])
    final_stable = labels == final_label
    answer_correct = correct.copy()
    answer_correct[:answer_start] = False
    answer_final_stable = final_stable.copy()
    answer_final_stable[:answer_start] = False
    most_certain_tick = int(np.argmax(certainties[:, 1]) + 1)
    return {
        "target": int(target),
        "final_label": final_label,
        "final_correct": int(final_label == target),
        "first_correct_sustained_tick": first_sustained(correct, missing=0),
        "answer_correct_sustained_tick": first_sustained(answer_correct, missing=0),
        "final_label_stable_tick": first_sustained(final_stable),
        "answer_final_label_stable_tick": first_sustained(answer_final_stable),
        "most_certain_tick": most_certain_tick,
        "most_certain_label": int(labels[most_certain_tick - 1]),
        "most_certain_correct": int(labels[most_certain_tick - 1] == target),
        "final_certainty": float(certainties[-1, 1]),
        "max_certainty": float(certainties[:, 1].max()),
    }, {
        "correct": correct.astype(np.bool_),
        "labels": labels.astype(np.int16),
        "prob_target": None,
    }


def save_qamnist_result_gif(
    images: np.ndarray,
    question: str,
    probabilities: np.ndarray,
    certainties: np.ndarray,
    target: int,
    step_types: np.ndarray,
    output_path: Path,
) -> None:
    frames = []
    logits_ticks = probabilities.shape[0]
    n_digits = images.shape[0]
    for tick in range(logits_ticks):
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
        digit_idx = min(tick, n_digits - 1)
        img = images[digit_idx, 0]
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title(f"Input digit view {digit_idx + 1}/{n_digits}")
        bars = axes[1].bar(np.arange(probabilities.shape[1]), probabilities[tick], color="#2a9d8f")
        bars[target].set_color("#e76f51")
        axes[1].set_ylim(0, 1)
        axes[1].set_title(f"P(answer) | pred={int(probabilities[tick].argmax())}")
        axes[2].plot(np.arange(1, logits_ticks + 1), certainties[:, 1], color="#264653")
        axes[2].axvline(tick + 1, color="black", alpha=0.5)
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Certainty")
        axes[3].axis("off")
        axes[3].text(
            0,
            0.95,
            f"tick {tick + 1}/{logits_ticks}\nphase: {step_types[tick]}\ntarget: {target}\n\n{question}",
            va="top",
            fontsize=9,
        )
        for ax in axes[:2]:
            ax.set_xticks([])
            ax.set_yticks([])
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
    ensure_qamnist_args(model_args)
    x, z, question_readable, targets = qamnist_batch(model_args, args.samples, args.device)
    model = prepare_model(model_args, args.device)
    with torch.inference_mode():
        model(x[:1], z[:1])
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    with torch.inference_mode():
        trace = trace_qamnist_forward(model, x, z)

    predictions = trace["predictions"]
    certainties = trace["certainties"]
    probabilities = torch.softmax(torch.from_numpy(predictions), dim=1).numpy().transpose(0, 2, 1)
    labels = probabilities.argmax(axis=2)
    sync_action_all = trace["sync_action"]
    sync_out_all = trace["sync_out"]
    queries_all = trace["queries"]
    post = trace["post_activations"]
    step_types = trace["step_types"]
    answer_start = int(trace["phase_boundaries"]["question_end_tick"])

    rows = []
    sample_series = []
    targets_np = targets.detach().cpu().numpy()
    for sample in range(args.samples):
        task_metrics, task_series = qamnist_metrics(
            labels[sample], int(targets_np[sample]), certainties[sample].transpose(1, 0), answer_start
        )
        task_series["prob_target"] = probabilities[sample, :, int(targets_np[sample])].astype(np.float32)
        sync_metrics, sync_series = compute_sync_convergence_metrics(
            sync_action_all[:, sample], sync_out_all[:, sample], post[:, sample]
        )
        row = {"sample_id": sample, **task_metrics, **sync_metrics}
        rows.append(row)
        sample_series.append((task_series, sync_series))

    save_rows(args.output_dir / "convergence_summary.csv", rows)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "task": "qamnist",
                "checkpoint": str(args.checkpoint),
                "samples": args.samples,
                "ticks": int(trace["phase_boundaries"]["total_ticks"]),
                "phase_boundaries": trace["phase_boundaries"],
                "d_model": int(model_args.d_model),
                "neuron_select_type": model.neuron_select_type,
                "S_full_definition": "unweighted cumulative post-activation Gram",
                "query_definition": "q_proj(S_action) recomputed from tracked S_action; consumed by attention on digit-observation ticks",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    stability = np.asarray([row["answer_correct_sustained_tick"] for row in rows])
    chosen = np.unique(np.asarray([0, int(np.argmax([r["final_correct"] for r in rows])), int(stability.argmax())]))
    chosen = chosen[: args.detailed_samples]
    action_left, action_right = synchronization_pair_indices(model, "action")
    out_left, out_right = synchronization_pair_indices(model, "out")
    x_np = x.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()
    certainties_tick = certainties.transpose(0, 2, 1)
    for sample in chosen:
        sample_dir = args.output_dir / "samples" / f"sample-{sample:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        active_indices, activity = select_active_neurons(post[:, sample], args.active_neurons)
        full_history = active_full_sync_history(post[:, sample], active_indices)
        order = active_neuron_order(full_history[-1])
        np.savez_compressed(
            sample_dir / "traces.npz",
            probabilities=probabilities[sample],
            certainties=certainties_tick[sample],
            predictions=predictions[sample],
            sync_action=sync_action_all[:, sample],
            sync_out=sync_out_all[:, sample],
            queries=queries_all[:, sample],
            pre_activations=trace["pre_activations"][:, sample],
            post_activations=post[:, sample],
            target=targets_np[sample],
            input_images=x_np[sample],
            question_tokens=z_np[sample],
            step_types=step_types,
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
            action_left=action_left,
            action_right=action_right,
            out_left=out_left,
            out_right=out_right,
        )
        (sample_dir / "question.txt").write_text(question_readable[sample], encoding="utf-8")
        (sample_dir / "metrics.json").write_text(json.dumps(rows[sample], indent=2), encoding="utf-8")
        save_qamnist_result_gif(
            x_np[sample],
            question_readable[sample],
            probabilities[sample],
            certainties_tick[sample],
            int(targets_np[sample]),
            step_types,
            sample_dir / "result.gif",
        )
        plot_query_overview(queries_all[:, sample], sample_dir / "query_overview.png")
        save_query_gif(queries_all[:, sample], sample_dir / "query.gif")
        save_sync_triptych_gif(
            sync_action_all[:, sample],
            sync_out_all[:, sample],
            full_history,
            sample_dir / "synchronization_triptych.gif",
            model.neuron_select_type,
            model.n_synch_action,
            model.n_synch_out,
            order,
        )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    valid = stability[stability > 0]
    axes[0].hist(valid, bins=np.arange(0.5, trace["phase_boundaries"]["total_ticks"] + 1.5), color="#287271")
    axes[0].axvline(answer_start, color="#e76f51", linestyle="--", label="answer starts")
    axes[0].set(xlabel="Correct-answer sustained tick", ylabel="Samples")
    axes[0].legend()
    axes[1].plot(np.arange(1, trace["phase_boundaries"]["total_ticks"] + 1), [
        np.mean((stability > 0) & (stability <= tick))
        for tick in range(1, trace["phase_boundaries"]["total_ticks"] + 1)
    ])
    axes[1].axvline(answer_start, color="#e76f51", linestyle="--")
    axes[1].set(xlabel="Tick", ylabel="Fraction correct and stable", ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(args.output_dir / "aggregate_convergence.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
