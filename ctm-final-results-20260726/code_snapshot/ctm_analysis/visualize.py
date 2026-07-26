from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .probe import normalized_sync_matrix, triangular_representation


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_sample_summary(
    image: np.ndarray,
    probabilities: np.ndarray,
    certainties: np.ndarray,
    attention: np.ndarray,
    class_names: list[str],
    metrics: dict[str, float | int],
    output_path: Path,
    selected_ticks: list[int],
    feature_shape: tuple[int, int],
) -> None:
    ticks = [tick for tick in selected_ticks if 1 <= tick <= probabilities.shape[0]]
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, max(len(ticks), 4), height_ratios=[1.05, 1])

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(image)
    ax.set_title("Input")
    ax.axis("off")

    h_feat, w_feat = feature_shape
    for col, tick in enumerate(ticks[1:], start=1):
        if col >= grid.ncols:
            break
        attn = attention[tick - 1].mean(axis=0).reshape(h_feat, w_feat)
        attn = F.interpolate(
            torch.from_numpy(attn)[None, None], size=image.shape[:2], mode="bilinear", align_corners=False
        )[0, 0].numpy()
        attn = (attn - attn.min()) / max(float(attn.max() - attn.min()), 1e-8)
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(image)
        ax.imshow(attn, cmap="viridis", alpha=0.55)
        ax.set_title(f"Attention, tick {tick}")
        ax.axis("off")

    ax_prob = fig.add_subplot(grid[1, : max(2, grid.ncols // 2)])
    final_top = np.argsort(probabilities[-1])[-5:][::-1]
    x = np.arange(1, probabilities.shape[0] + 1)
    for index in final_top:
        ax_prob.plot(x, probabilities[:, index], label=class_names[index][:28])
    ax_prob.axvline(metrics["label_stable_tick"], color="black", linestyle="--", linewidth=1.2)
    ax_prob.set(xlabel="Internal tick", ylabel="Probability", xlim=(1, probabilities.shape[0]))
    ax_prob.set_title("Final top-5 class probabilities")
    ax_prob.legend(fontsize=7, loc="best")

    ax_cert = fig.add_subplot(grid[1, max(2, grid.ncols // 2) :])
    ax_cert.plot(x, certainties, color="#202020", linewidth=2)
    ax_cert.axvline(metrics["certainty_plateau_tick"], color="#d1495b", linestyle="--")
    ax_cert.set(xlabel="Internal tick", ylabel="Certainty", xlim=(1, probabilities.shape[0]), ylim=(0, 1))
    ax_cert.set_title("Certainty trajectory")
    fig.suptitle(
        f"Final: {class_names[int(metrics['final_class'])]} | "
        f"label stable at tick {metrics['label_stable_tick']}"
    )
    fig.tight_layout()
    _save(fig, output_path)


def plot_sync_overview(
    sync_action: np.ndarray,
    sync_out: np.ndarray,
    full_sync_small: np.ndarray,
    series: dict[str, np.ndarray],
    neuron_select_type: str,
    n_synch_action: int,
    n_synch_out: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    final_action = sync_action[-1]
    final_out = sync_out[-1]

    if neuron_select_type in ("random", "first-last"):
        action_view = triangular_representation(final_action, n_synch_action)
        out_view = triangular_representation(final_out, n_synch_out)
        axes[0, 0].imshow(action_view, cmap="coolwarm", aspect="auto")
        axes[0, 1].imshow(out_view, cmap="coolwarm", aspect="auto")
    else:
        axes[0, 0].imshow(sync_action.T, cmap="coolwarm", aspect="auto")
        axes[0, 1].imshow(sync_out.T, cmap="coolwarm", aspect="auto")
        axes[0, 0].set_xlabel("Tick")
        axes[0, 1].set_xlabel("Tick")
    axes[0, 0].set_title("S_action (exact model representation)")
    axes[0, 1].set_title("S_out (exact model representation)")
    lim = np.quantile(np.abs(full_sync_small), 0.995)
    axes[0, 2].imshow(full_sync_small, cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[0, 2].set_title("S_full = Z Z^T (block-mean view)")

    x = np.arange(1, sync_action.shape[0] + 1)
    axes[1, 0].plot(x, series["action_cosine_to_final"], label="action")
    axes[1, 0].plot(x, series["out_cosine_to_final"], label="out")
    axes[1, 0].plot(x, series["full_sync_cosine_to_final"], label="full")
    axes[1, 0].set(xlabel="Tick", ylabel="Cosine to final", ylim=(0, 1.01))
    axes[1, 0].legend()

    axes[1, 1].plot(x, series["action_step_relative_l2"], label="action")
    axes[1, 1].plot(x, series["out_step_relative_l2"], label="out")
    axes[1, 1].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set(xlabel="Tick", ylabel="Step relative L2", ylim=(0, 1))
    axes[1, 1].legend()

    axes[1, 2].plot(x, series["full_sync_step_relative_frobenius"], color="#00798c")
    axes[1, 2].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[1, 2].set(xlabel="Tick", ylabel="Step relative Frobenius", ylim=(0, 1))
    axes[1, 2].set_title("Full synchronization step change")
    fig.tight_layout()
    _save(fig, output_path)


def active_neuron_order(final_matrix: np.ndarray) -> np.ndarray:
    """Return a deterministic correlation-cluster order for an active subset."""
    normalized = normalized_sync_matrix(final_matrix)
    distance = np.clip(1.0 - np.abs(normalized), 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    if len(distance) < 2:
        return np.arange(len(distance))
    return leaves_list(linkage(squareform(distance, checks=False), method="average"))


def plot_active_sync(
    history: np.ndarray,
    neuron_indices: np.ndarray,
    output_path: Path,
    order: np.ndarray | None = None,
) -> np.ndarray:
    """Plot an exact active-neuron submatrix in activity and cluster orders."""
    final = history[-1]
    order = active_neuron_order(final) if order is None else np.asarray(order)
    limit = max(float(np.quantile(np.abs(final), 0.995)), 1e-8)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(final, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    axes[0].set_title(f"Exact active S_full ({len(neuron_indices)} neurons)")
    clustered = final[np.ix_(order, order)]
    axes[1].imshow(clustered, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    axes[1].set_title("Same matrix, correlation-clustered")
    axes[2].imshow(
        normalized_sync_matrix(clustered), cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
    )
    axes[2].set_title("Normalized pairwise synchronization")
    for ax in axes:
        ax.set_xlabel("Selected neuron rank")
        ax.set_ylabel("Selected neuron rank")
    fig.suptitle("Rows/columns map to original IDs in active_neuron_indices.npy")
    fig.tight_layout()
    _save(fig, output_path)
    return order


def save_active_sync_gif(
    history: np.ndarray,
    neuron_indices: np.ndarray,
    output_path: Path,
    order: np.ndarray,
    fps: int = 10,
) -> None:
    """Animate a fixed active subset so apparent motion reflects only state change."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = history[:, order][:, :, order]
    limit = max(float(np.quantile(np.abs(ordered[-1]), 0.995)), 1e-8)
    frames = []
    for tick, matrix in enumerate(ordered, start=1):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
        axes[0].imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
        axes[0].set_title(f"Active S_full, tick {tick}")
        axes[1].imshow(
            normalized_sync_matrix(matrix), cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest"
        )
        axes[1].set_title("Normalized")
        for ax in axes:
            ax.set_xlabel("Clustered active-neuron rank")
            ax.set_ylabel("Clustered active-neuron rank")
        fig.suptitle(f"Fixed {len(neuron_indices)}-neuron subset")
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        frames.append(frame)
        plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)


def plot_query_overview(queries: np.ndarray, output_path: Path) -> None:
    """Plot the exact q_proj(S_action) query trajectory as a tick-by-feature heatmap."""
    limit = max(float(np.quantile(np.abs(queries), 0.995)), 1e-8)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].imshow(queries, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
    axes[0].set_title("Query trajectory: q_proj(S_action)")
    axes[0].set_xlabel("Query feature")
    axes[0].set_ylabel("Internal tick")
    axes[1].plot(np.arange(1, queries.shape[0] + 1), np.linalg.norm(queries, axis=1), color="#202020")
    axes[1].set_title("Query vector norm")
    axes[1].set_xlabel("Internal tick")
    axes[1].set_ylabel("L2 norm")
    fig.tight_layout()
    _save(fig, output_path)


def save_query_gif(queries: np.ndarray, output_path: Path, fps: int = 10) -> None:
    """Animate query vectors; square reshape is used when possible for compact visual inspection."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_features = queries.shape[1]
    side = int(np.sqrt(n_features))
    as_square = side * side == n_features
    limit = max(float(np.quantile(np.abs(queries), 0.995)), 1e-8)
    frames = []
    for tick, query in enumerate(queries, start=1):
        fig, ax = plt.subplots(figsize=(5.2, 4.8))
        if as_square:
            ax.imshow(query.reshape(side, side), cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.set_title(f"Query q_proj(S_action), tick {tick}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.imshow(query[None, :], cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto", interpolation="nearest")
            ax.set_title(f"Query q_proj(S_action), tick {tick}")
            ax.set_xlabel("Query feature")
            ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)


def _representation_view(vector: np.ndarray, neuron_select_type: str, n_synch: int) -> np.ndarray:
    if neuron_select_type in ("first-last", "random"):
        return triangular_representation(vector, n_synch)
    width = int(np.ceil(np.sqrt(vector.size)))
    result = np.full((width, width), np.nan, dtype=np.float32)
    result.flat[: vector.size] = vector
    return result


def save_sync_triptych_gif(
    sync_action: np.ndarray | None,
    sync_out: np.ndarray | None,
    full_history: np.ndarray,
    output_path: Path,
    neuron_select_type: str,
    n_synch_action: int,
    n_synch_out: int,
    full_order: np.ndarray | None = None,
    fps: int = 10,
) -> None:
    """Animate exact model representations beside an exact active S_full subset."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ticks = full_history.shape[0]
    full_order = np.arange(full_history.shape[1]) if full_order is None else np.asarray(full_order)
    ordered_full = full_history[:, full_order][:, :, full_order]
    action_limit = max(float(np.quantile(np.abs(sync_action), 0.995)), 1e-8) if sync_action is not None else 1
    out_limit = max(float(np.quantile(np.abs(sync_out), 0.995)), 1e-8) if sync_out is not None else 1
    full_limit = max(float(np.quantile(np.abs(ordered_full[-1]), 0.995)), 1e-8)
    frames = []
    for index in range(ticks):
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
        if sync_action is None:
            axes[0].text(0.5, 0.5, "Not used by this task", ha="center", va="center")
        else:
            axes[0].imshow(
                _representation_view(sync_action[index], neuron_select_type, n_synch_action),
                cmap="coolwarm", vmin=-action_limit, vmax=action_limit, interpolation="nearest",
            )
        axes[0].set_title("S_action (official representation)")
        if sync_out is None:
            axes[1].text(0.5, 0.5, "Not used by this task", ha="center", va="center")
        else:
            axes[1].imshow(
                _representation_view(sync_out[index], neuron_select_type, n_synch_out),
                cmap="coolwarm", vmin=-out_limit, vmax=out_limit, interpolation="nearest",
            )
        axes[1].set_title("S_out (official representation)")
        axes[2].imshow(
            ordered_full[index], cmap="coolwarm", vmin=-full_limit, vmax=full_limit, interpolation="nearest"
        )
        axes[2].set_title("S_full (fixed active-neuron subset)")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(f"Internal tick {index + 1} / {ticks}")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    imageio.mimsave(output_path, frames, fps=fps, loop=0)
