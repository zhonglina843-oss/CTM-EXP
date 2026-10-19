"""Tiny deterministic integration check against the local official CTM checkout."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
CTM_REPO = ROOT.parent / "ctm-official"
sys.path.insert(0, str(CTM_REPO))

from models.ctm import ContinuousThoughtMachine  # noqa: E402
from ucl_ctm_probe.probe.ctm_trace import CTMTraceWrapper, trace_ctm  # noqa: E402


def main() -> None:
    np.random.seed(7)
    torch.manual_seed(7)
    model = ContinuousThoughtMachine(
        iterations=20,
        d_model=8,
        d_input=8,
        heads=1,
        n_synch_out=4,
        n_synch_action=4,
        synapse_depth=1,
        memory_length=2,
        deep_nlms=False,
        memory_hidden_dims=2,
        do_layernorm_nlm=False,
        backbone_type="none",
        positional_embedding_type="none",
        out_dims=3,
        dropout=0,
        dropout_nlm=0,
        neuron_select_type="first-last",
    )
    inputs = torch.randn(2, 8, 1, 1)
    # Initialize the upstream LazyLinear modules once; initialization is not
    # part of the repeated deterministic probe itself.
    with torch.inference_mode():
        model(inputs, track=False)
    state_before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    first = trace_ctm(model, inputs)
    second = trace_ctm(model, inputs)
    wrapper = CTMTraceWrapper(model)
    wrapped = wrapper(inputs, return_trace=True)
    np.testing.assert_array_equal(first["tick_ids"], second["tick_ids"])
    np.testing.assert_array_equal(first["z_history"], second["z_history"])
    np.testing.assert_array_equal(first["sync_history"]["out"], second["sync_history"]["out"])
    np.testing.assert_array_equal(first["sync_history"]["action"], second["sync_history"]["action"])
    np.testing.assert_array_equal(first["post_state"], second["post_state"])
    np.testing.assert_array_equal(first["z_history"], wrapped["z_history"])
    state_unchanged = all(
        torch.equal(value, state_before[key]) for key, value in model.state_dict().items()
    )
    assert state_unchanged, "trace extraction changed CTM state_dict"
    assert first["z_history"].shape == (7, 2, 8)
    assert first["sync_history"]["out"].shape == (7, 2, 10)
    assert first["sync_history"]["action"].shape == (7, 2, 10)
    assert first["post_state"].shape == (2, 8)

    artifact_root = ROOT / "artifacts" / "A1"
    table_root = artifact_root / "tables"
    figure_root = artifact_root / "figures"
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    shapes = {
        "upstream_commit": subprocess.check_output(
            ["git", "-C", str(CTM_REPO), "rev-parse", "HEAD"], text=True
        ).strip(),
        "seed": 7,
        "input_shape": list(inputs.shape),
        "tick_ids_1_based": first["tick_ids"].tolist(),
        "z_history_shape": list(first["z_history"].shape),
        "sync_out_shape": list(first["sync_history"]["out"].shape),
        "sync_action_shape": list(first["sync_history"]["action"].shape),
        "post_state_shape": list(first["post_state"].shape),
        "sync_note": "first-last gives a flattened upper-triangle vector (10 values here), not a square matrix",
        "repeated_trace_exactly_equal": True,
        "model_state_dict_unchanged": state_unchanged,
        "wrapper_return_trace_path": True,
    }
    (table_root / "ctm_trace_smoke_shapes.json").write_text(
        json.dumps(shapes, indent=2) + "\n", encoding="utf-8"
    )
    blocks = [
        (20, "Input features", "B×8×1×1"),
        (260, "CTM recurrence", "20 ticks · tracked"),
        (500, "Post-activation", "20×2×8"),
        (740, "Sync out / action", "20×2×10 each"),
        (980, "Selected ticks", "z 7×2×8; sync 7×2×10"),
    ]
    boxes = "\n".join(
        f'<rect x="{x}" y="95" width="180" height="115" rx="12" fill="#eaf2f8" stroke="#315a76" stroke-width="2"/>'
        f'<text x="{x + 90}" y="138" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold">{title}</text>'
        f'<text x="{x + 90}" y="170" text-anchor="middle" font-family="sans-serif" font-size="14">{detail.replace(chr(10), "&#10;")}</text>'
        for x, title, detail in blocks
    )
    arrows = "\n".join(
        f'<path d="M{x} 152 H{x + 56}" stroke="#444" stroke-width="2" marker-end="url(#arrow)"/>'
        for x in (202, 442, 682, 922)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="1190" height="300" viewBox="0 0 1190 300">'
        '<defs><marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0 0 L10 4 L0 8 z" fill="#444"/></marker></defs>'
        f'{boxes}{arrows}'
        '<text x="595" y="260" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#8b2f2f">Sync outputs preserve official vector shape; no square matrix is inferred.</text>'
        '</svg>\n'
    )
    (figure_root / "ctm_trace_shape_smoke.svg").write_text(svg, encoding="utf-8")
    print(json.dumps(shapes, indent=2))


if __name__ == "__main__":
    main()
