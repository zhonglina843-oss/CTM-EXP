"""Read-only trace extraction for the official CTM ``track=True`` API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from torch.nn.parameter import UninitializedParameter


DEFAULT_TICK_IDS = (1, 2, 4, 8, 12, 16, 20)


class CTMTraceWrapper(torch.nn.Module):
    """Preserve the official call by default and expose an opt-in trace path."""

    def __init__(self, model: torch.nn.Module, tick_ids: Sequence[int] = DEFAULT_TICK_IDS):
        super().__init__()
        self.model = model
        self.tick_ids = tuple(int(tick) for tick in tick_ids)

    def forward(self, inputs: torch.Tensor, *, return_trace: bool = False) -> Any:
        if not return_trace:
            return self.model(inputs, track=False)
        return trace_ctm(self.model, inputs, self.tick_ids)


def trace_ctm(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    tick_ids: Sequence[int] = DEFAULT_TICK_IDS,
) -> dict[str, Any]:
    """Run one deterministic tracked inference and return selected 1-based ticks.

    ``sync_history`` preserves the two exact representations returned by the
    official model (``out`` and ``action``); neither is reshaped or called a
    full synchronization matrix. The upstream forward currently clamps two
    decay parameters in-place, so their values are restored before returning.
    """
    ticks = tuple(int(tick) for tick in tick_ids)
    iterations = int(getattr(model, "iterations"))
    if not ticks or any(tick < 1 or tick > iterations for tick in ticks):
        raise ValueError(f"tick_ids must be non-empty and within [1, {iterations}]")
    if len(set(ticks)) != len(ticks):
        raise ValueError("tick_ids must not contain duplicates")
    if not isinstance(inputs, torch.Tensor) or inputs.ndim < 2:
        raise ValueError("inputs must be a batched torch.Tensor")
    if any(isinstance(parameter, UninitializedParameter) for parameter in model.parameters()):
        raise RuntimeError("CTM has uninitialized lazy parameters; initialize/load it before read-only probing")

    training_modes = [(module, module.training) for module in model.modules()]
    decay_params = [
        (parameter, parameter.detach().clone())
        for name in ("decay_params_action", "decay_params_out")
        if isinstance((parameter := getattr(model, name, None)), torch.Tensor)
    ]
    model.eval()
    try:
        with torch.inference_mode():
            result = model(inputs, track=True)
        if not isinstance(result, tuple) or len(result) != 6:
            raise TypeError("expected official CTM forward(..., track=True) six-item return")
        _, _, sync_pair, _, post_history, _ = result
        if not isinstance(sync_pair, tuple) or len(sync_pair) != 2:
            raise TypeError("expected tracked synchronization pair (sync_out, sync_action)")

        post = np.asarray(post_history, dtype=np.float32)
        sync_out = np.asarray(sync_pair[0], dtype=np.float32)
        sync_action = np.asarray(sync_pair[1], dtype=np.float32)
        if post.ndim != 3 or sync_out.ndim != 3 or sync_action.ndim != 3:
            raise ValueError("expected histories shaped (ticks, batch, feature)")
        if post.shape[:2] != sync_out.shape[:2] or post.shape[:2] != sync_action.shape[:2]:
            raise ValueError("activation and synchronization histories are not tick/batch aligned")
        if post.shape[0] != iterations:
            raise ValueError(f"tracked {post.shape[0]} ticks; model.iterations={iterations}")

        indices = np.asarray(ticks, dtype=np.int64) - 1
        selected_z = post[indices].copy()
        selected_out = sync_out[indices].copy()
        selected_action = sync_action[indices].copy()
        selection = getattr(model, "neuron_select_type", None)
        sync_kind = (
            "flattened_upper_triangle_vector"
            if selection in ("first-last", "random")
            else "sampled_pair_vector"
        )
        return {
            "tick_ids": np.asarray(ticks, dtype=np.int64),
            "z_history": selected_z,
            "sync_history": {
                "out": selected_out,
                "action": selected_action,
            },
            "post_state": post[-1].copy(),
            "metadata": {
                "model_class": type(model).__name__,
                "batch_size": int(post.shape[1]),
                "total_iterations": iterations,
                "post_state_tick": iterations,
                "neuron_select_type": selection,
                "z_history_shape": list(selected_z.shape),
                "sync_out_shape": list(selected_out.shape),
                "sync_action_shape": list(selected_action.shape),
                "post_state_shape": list(post[-1].shape),
                "sync_representation_kind": sync_kind,
                "sync_semantics": "official flattened/vector representation; original shape preserved",
            },
        }
    finally:
        with torch.no_grad():
            for parameter, original in decay_params:
                if not torch.equal(parameter.detach(), original):
                    parameter.copy_(original)
        for module, was_training in training_modes:
            module.training = was_training


def initialize_frozen_feature_ctm(
    input_dim: int = 64,
    d_model: int = 128,
    n_synch: int = 32,
    seed: int = 1,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    """Create an untrained frozen CTM probe and fixed input adapter.

    This is an engineering smoke configuration. It deliberately does not load
    a research checkpoint or claim a calibrated representation.
    """
    try:
        from models.ctm import ContinuousThoughtMachine
    except ImportError as exc:
        raise ImportError(
            "CTM checkout must be importable (e.g. add ctm-official/ to PYTHONPATH)"
        ) from exc

    if input_dim <= 0 or d_model < 2 or n_synch <= 0 or 2 * n_synch > d_model:
        raise ValueError("require input_dim > 0 and d_model >= 2*n_synch > 0")
    torch.manual_seed(seed)
    adapter = torch.nn.Linear(input_dim, input_dim, bias=False)
    with torch.no_grad():
        torch.nn.init.orthogonal_(adapter.weight)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)

    model = ContinuousThoughtMachine(
        iterations=max(DEFAULT_TICK_IDS),
        d_model=d_model,
        d_input=input_dim,
        heads=1,
        n_synch_out=n_synch,
        n_synch_action=n_synch,
        synapse_depth=1,
        memory_length=4,
        deep_nlms=False,
        memory_hidden_dims=16,
        do_layernorm_nlm=True,
        backbone_type="none",
        positional_embedding_type="none",
        out_dims=input_dim,
        prediction_reshaper=[input_dim],
        dropout=0,
        dropout_nlm=0,
        neuron_select_type="first-last",
    )
    model.eval()
    dummy = torch.zeros(2, input_dim, 1, 1)
    with torch.inference_mode():
        model(dummy, track=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return adapter, model
