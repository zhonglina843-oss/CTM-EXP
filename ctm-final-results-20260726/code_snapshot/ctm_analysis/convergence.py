from __future__ import annotations

import numpy as np


def first_sustained(mask: np.ndarray, missing: int | None = None) -> int:
    """Return the first 1-based tick from which mask stays true."""
    suffix = np.logical_and.accumulate(mask[::-1])[::-1]
    hits = np.flatnonzero(suffix)
    if hits.size:
        return int(hits[0] + 1)
    return int(mask.size) if missing is None else int(missing)


def _first_sustained(mask: np.ndarray) -> int:
    return first_sustained(mask)


def _cosine_to_final(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    final = values[-1]
    numerator = values @ final
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(final)
    return numerator / np.maximum(denominator, eps)


def _relative_l2_to_final(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    final = values[-1]
    return np.linalg.norm(values - final, axis=1) / max(np.linalg.norm(final), eps)


def _step_relative_l2(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    result = np.full(values.shape[0], np.inf, dtype=np.float64)
    result[1:] = np.linalg.norm(np.diff(values, axis=0), axis=1) / np.maximum(
        np.linalg.norm(values[1:], axis=1), eps
    )
    return result


def _js_to_final(probabilities: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = probabilities[-1]
    m = 0.5 * (probabilities + q[None])
    p_term = np.sum(probabilities * (np.log(probabilities + eps) - np.log(m + eps)), axis=1)
    q_term = np.sum(q[None] * (np.log(q[None] + eps) - np.log(m + eps)), axis=1)
    return 0.5 * (p_term + q_term)


def full_sync_similarity(post_activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compare every raw S^t to S^T exactly without materializing D x D matrices."""
    z = np.asarray(post_activations, dtype=np.float64)
    gram = z @ z.T
    final_norm_sq = np.square(gram).sum()
    cosine = np.empty(z.shape[0], dtype=np.float64)
    relative_frobenius = np.empty(z.shape[0], dtype=np.float64)
    for tick in range(1, z.shape[0] + 1):
        current_norm_sq = np.square(gram[:tick, :tick]).sum()
        cross = np.square(gram[:tick, :]).sum()
        denom = np.sqrt(max(current_norm_sq * final_norm_sq, 1e-24))
        cosine[tick - 1] = cross / denom
        distance_sq = max(current_norm_sq + final_norm_sq - 2.0 * cross, 0.0)
        relative_frobenius[tick - 1] = np.sqrt(distance_sq / max(final_norm_sq, 1e-24))
    return cosine.astype(np.float32), relative_frobenius.astype(np.float32)


def full_sync_step_change(post_activations: np.ndarray) -> np.ndarray:
    """Return ||S^t-S^(t-1)||_F / ||S^t||_F without materializing S."""
    z = np.asarray(post_activations, dtype=np.float64)
    gram = z @ z.T
    result = np.full(z.shape[0], np.inf, dtype=np.float64)
    for tick in range(2, z.shape[0] + 1):
        current_norm_sq = np.square(gram[:tick, :tick]).sum()
        increment_norm = gram[tick - 1, tick - 1]
        result[tick - 1] = increment_norm / np.sqrt(max(current_norm_sq, 1e-24))
    return result.astype(np.float32)


def compute_convergence_metrics(
    predictions: np.ndarray,
    certainties: np.ndarray,
    sync_action: np.ndarray,
    sync_out: np.ndarray,
    post_activations: np.ndarray,
    js_threshold: float = 1e-3,
    certainty_tolerance: float = 0.01,
    cosine_threshold: float = 0.99,
    relative_l2_threshold: float = 0.05,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    logits = predictions - predictions.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = probabilities.argmax(axis=1)
    final_label = labels[-1]
    label_stable = labels == final_label

    top5 = np.argsort(probabilities, axis=1)[:, -5:]
    final_top5 = set(top5[-1].tolist())
    top5_stable = np.asarray([set(row.tolist()) == final_top5 for row in top5])

    js = _js_to_final(probabilities)
    certainty_delta = np.abs(certainties - certainties[-1])
    action_cos = _cosine_to_final(sync_action)
    action_l2 = _relative_l2_to_final(sync_action)
    out_cos = _cosine_to_final(sync_out)
    out_l2 = _relative_l2_to_final(sync_out)
    full_cos, full_fro = full_sync_similarity(post_activations)
    action_step = _step_relative_l2(sync_action)
    out_step = _step_relative_l2(sync_out)
    full_step = full_sync_step_change(post_activations)

    series = {
        "js_to_final": js.astype(np.float32),
        "certainty_delta_to_final": certainty_delta.astype(np.float32),
        "action_cosine_to_final": action_cos.astype(np.float32),
        "action_relative_l2_to_final": action_l2.astype(np.float32),
        "out_cosine_to_final": out_cos.astype(np.float32),
        "out_relative_l2_to_final": out_l2.astype(np.float32),
        "full_sync_cosine_to_final": full_cos,
        "full_sync_relative_frobenius_to_final": full_fro,
        "action_step_relative_l2": action_step.astype(np.float32),
        "out_step_relative_l2": out_step.astype(np.float32),
        "full_sync_step_relative_frobenius": full_step,
    }
    metrics: dict[str, float | int] = {
        "final_class": int(final_label),
        "final_certainty": float(certainties[-1]),
        "label_stable_tick": _first_sustained(label_stable),
        "top5_stable_tick": _first_sustained(top5_stable),
        "probability_js_tick": _first_sustained(js <= js_threshold),
        "certainty_plateau_tick": _first_sustained(certainty_delta <= certainty_tolerance),
        "action_cosine_tick": _first_sustained(action_cos >= cosine_threshold),
        "action_relative_l2_tick": _first_sustained(action_l2 <= relative_l2_threshold),
        "out_cosine_tick": _first_sustained(out_cos >= cosine_threshold),
        "out_relative_l2_tick": _first_sustained(out_l2 <= relative_l2_threshold),
        "full_sync_cosine_tick": _first_sustained(full_cos >= cosine_threshold),
        "full_sync_frobenius_tick": _first_sustained(full_fro <= relative_l2_threshold),
        "action_step_plateau_tick": _first_sustained(action_step <= relative_l2_threshold),
        "out_step_plateau_tick": _first_sustained(out_step <= relative_l2_threshold),
        "full_sync_step_plateau_tick": _first_sustained(full_step <= relative_l2_threshold),
    }
    return metrics, series


def compute_sync_convergence_metrics(
    sync_action: np.ndarray | None,
    sync_out: np.ndarray | None,
    post_activations: np.ndarray,
    cosine_threshold: float = 0.99,
    relative_l2_threshold: float = 0.05,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Compute task-independent convergence for available CTM representations."""
    full_cos, full_fro = full_sync_similarity(post_activations)
    full_step = full_sync_step_change(post_activations)
    metrics: dict[str, int] = {
        "full_sync_cosine_tick": first_sustained(full_cos >= cosine_threshold),
        "full_sync_frobenius_tick": first_sustained(full_fro <= relative_l2_threshold),
        "full_sync_step_plateau_tick": first_sustained(full_step <= relative_l2_threshold),
    }
    series: dict[str, np.ndarray] = {
        "full_sync_cosine_to_final": full_cos,
        "full_sync_relative_frobenius_to_final": full_fro,
        "full_sync_step_relative_frobenius": full_step,
    }
    for name, values in (("action", sync_action), ("out", sync_out)):
        if values is None or np.asarray(values).size == 0:
            continue
        values = np.asarray(values)
        cosine = _cosine_to_final(values).astype(np.float32)
        relative = _relative_l2_to_final(values).astype(np.float32)
        step = _step_relative_l2(values).astype(np.float32)
        series[f"{name}_cosine_to_final"] = cosine
        series[f"{name}_relative_l2_to_final"] = relative
        series[f"{name}_step_relative_l2"] = step
        metrics[f"{name}_cosine_tick"] = first_sustained(cosine >= cosine_threshold)
        metrics[f"{name}_relative_l2_tick"] = first_sustained(relative <= relative_l2_threshold)
        metrics[f"{name}_step_plateau_tick"] = first_sustained(step <= relative_l2_threshold)
    return metrics, series
