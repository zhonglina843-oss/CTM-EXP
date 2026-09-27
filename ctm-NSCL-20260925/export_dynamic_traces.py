#!/usr/bin/env python3
"""Export frozen per-tick CTM traces from cached feature files.

Run this file from the continuous-thought-machines checkout. It reuses the
existing stage-2 model and dictionary encoder; it does not train or update
the CTM.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.dictionary_learning.common import model_kwargs_from_checkpoint
from scripts.dictionary_learning.train_cached_feature_ctm_stage2 import (
    CachedFeatureCTMStage2,
    encode,
    load_npz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True,
                   help="CSV with one row per feature: sample_id,label,split")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base-checkpoint", type=Path, default=None,
                   help="Original CTM checkpoint used to create stage2 checkpoint")
    p.add_argument("--dictionary", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def read_manifest(path: Path, n: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "label", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain columns {sorted(required)}")
    if len(rows) != n:
        raise ValueError(f"manifest rows={len(rows)} but features={n}")
    return rows


def resolve_device(value: str) -> torch.device:
    if value.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(value)


def load_model(args: argparse.Namespace, input_dim: int, device: torch.device):
    stage2 = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base_path = args.base_checkpoint
    if base_path is None and stage2.get("base_checkpoint"):
        base_path = Path(stage2["base_checkpoint"])
    if base_path is None or not base_path.exists():
        raise FileNotFoundError(
            "Pass --base-checkpoint: stage2 checkpoint does not contain an "
            "accessible original CTM checkpoint."
        )
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    checkpoint_args = base.get("args")
    if checkpoint_args is None:
        raise ValueError("base checkpoint is missing args")
    model_args = stage2.get("args", {})
    iterations = model_args.get("iterations")
    if iterations is None:
        iterations = getattr(checkpoint_args, "iterations", None)
    out_dims = model_args.get("out_dims", getattr(checkpoint_args, "out_dims", None))
    if out_dims is None:
        for key, value in stage2["model_state_dict"].items():
            if key.endswith("output_projector.0.weight") or key.endswith("output_projector.weight"):
                out_dims = int(value.shape[0])
                break
    overrides = {
        "out_dims": int(out_dims or 10),
        "prediction_reshaper": [-1],
        "iterations": int(iterations or 50),
    }
    kwargs = model_kwargs_from_checkpoint(checkpoint_args, overrides=overrides)
    hidden_dim = int(model_args.get("hidden_dim", 256))
    model = CachedFeatureCTMStage2(input_dim=input_dim, hidden_dim=hidden_dim, **kwargs).to(device)
    _ = model(torch.zeros(1, input_dim, device=device))
    result = model.load_state_dict(stage2["model_state_dict"], strict=False)
    print(f"Loaded checkpoint. missing={result.missing_keys} unexpected={result.unexpected_keys}", flush=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, stage2


@torch.inference_mode()
def export(model, inputs: np.ndarray, device: torch.device, batch_size: int):
    post, cert, pred = [], [], []
    for start in range(0, len(inputs), batch_size):
        x = torch.from_numpy(inputs[start:start + batch_size]).to(device)
        batch = x.size(0)
        kv = model.input_encoder(x).unsqueeze(1)
        state_trace = model.start_trace.unsqueeze(0).expand(batch, -1, -1)
        activated = model.start_activated_state.unsqueeze(0).expand(batch, -1)
        predictions = torch.empty(batch, model.out_dims, model.iterations, device=device)
        certainties = torch.empty(batch, 2, model.iterations, device=device)
        da = db = None
        model.decay_params_action.data.clamp_(0, 15)
        model.decay_params_out.data.clamp_(0, 15)
        ra = torch.exp(-model.decay_params_action).unsqueeze(0).repeat(batch, 1)
        ro = torch.exp(-model.decay_params_out).unsqueeze(0).repeat(batch, 1)
        _, doa, dob = model.compute_synchronisation(activated, None, None, ro, synch_type="out")
        post_batch = []
        for tick in range(model.iterations):
            sa, da, db = model.compute_synchronisation(activated, da, db, ra, synch_type="action")
            q = model.q_proj(sa).unsqueeze(1)
            attn, _ = model.attention(q, kv, kv, average_attn_weights=False, need_weights=False)
            state = model.synapses(torch.cat((attn.squeeze(1), activated), dim=-1))
            state_trace = torch.cat((state_trace[:, :, 1:], state.unsqueeze(-1)), dim=-1)
            activated = model.trace_processor(state_trace)
            so, doa, dob = model.compute_synchronisation(activated, doa, dob, ro, synch_type="out")
            logits = model.output_projector(so)
            predictions[..., tick] = logits
            certainties[..., tick] = model.compute_certainty(logits)
            post_batch.append(activated.detach().cpu().numpy())
        post.append(np.stack(post_batch, axis=1))
        cert.append(certainties[:, 1].detach().cpu().numpy())
        pred.append(predictions.argmax(dim=1).detach().cpu().numpy())
    return np.concatenate(post), np.concatenate(cert), np.concatenate(pred)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    features, labels = load_npz(args.features)
    rows = read_manifest(args.manifest, len(features))
    dictionary_path = args.dictionary
    stage2 = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if dictionary_path is None and stage2.get("dictionary"):
        dictionary_path = Path(stage2["dictionary"])
    dictionary = None
    if dictionary_path is not None:
        dictionary = torch.load(dictionary_path, map_location="cpu", weights_only=False)["atoms"].numpy()
    meta = stage2
    stage2_args = stage2.get("args", {})
    representation = stage2_args.get("representation", "recon")
    solver = stage2_args.get("solver", "omp")
    encode_k = stage2_args.get("encode_k", 32)
    encoded = encode(features, dictionary, representation, solver, encode_k)
    device = resolve_device(args.device)
    model, _ = load_model(args, encoded.shape[1], device)
    post, certainty, prediction = export(model, encoded, device, args.batch_size)
    np.savez_compressed(
        args.output_dir / "traces.npz",
        post_state=post.astype(np.float32),
        certainty=certainty.astype(np.float32),
        prediction=prediction.astype(np.int64),
        labels=labels.astype(np.int64),
    )
    (args.output_dir / "manifest.csv").write_text(
        "sample_id,label,split\n" + "\n".join(
            f"{r['sample_id']},{r['label']},{r['split']}" for r in rows
        ) + "\n", encoding="utf-8"
    )
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "features": str(args.features), "checkpoint": str(args.checkpoint),
        "base_checkpoint": str(args.base_checkpoint), "representation": representation,
        "solver": solver, "encode_k": encode_k, "shape": list(post.shape),
    }, indent=2), encoding="utf-8")
    print(f"DONE traces={post.shape} output={args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
