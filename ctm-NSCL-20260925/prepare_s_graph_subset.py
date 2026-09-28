#!/usr/bin/env python3
"""Materialize a small S_T subset for the complete graph-similarity suite."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ticks", default="1,5,10,15,20,30,40,50")
    p.add_argument("--classes-per-split", type=int, default=5)
    p.add_argument("--samples-per-class", type=int, default=5)
    a = p.parse_args()
    data = np.load(a.trace_dir / "traces.npz", mmap_mode="r")
    post = data["post_state"].astype(np.float32)
    with (a.trace_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        metadata = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for i, row in enumerate(metadata):
        if row["split"] in ("known_test", "novel_test"):
            grouped[(row["split"], row["label"])].append((i, row))
    chosen = []
    for split in ("known_test", "novel_test"):
        for key in sorted(k for k in grouped if k[0] == split)[:a.classes_per_split]:
            chosen.extend(grouped[key][:a.samples_per_class])
    ticks = [int(x) for x in a.ticks.split(",") if x.strip()]
    out = a.output_dir; (out / "samples").mkdir(parents=True, exist_ok=True)
    class_ids = {key: i for i, key in enumerate(sorted({(r["split"], r["label"]) for _, r in chosen}))}
    counts = defaultdict(int); label_rows = []
    active = np.arange(post.shape[-1], dtype=np.int64)
    for original_index, row in chosen:
        key = (row["split"], row["label"]); sample_id = f"s_{len(label_rows):04d}"
        sample_dir = out / "samples" / sample_id; sample_dir.mkdir(parents=True, exist_ok=True)
        matrices = []
        for tick in ticks:
            x = post[original_index, :tick]
            matrix = np.einsum("th,tk->hk", x, x) / tick
            matrix = (matrix + matrix.T) * 0.5
            matrix /= max(float(np.linalg.norm(matrix)), 1e-8)
            matrices.append(matrix.astype(np.float32))
        np.savez_compressed(sample_dir / "S_full_active_all_ticks.npz", matrices=np.stack(matrices))
        np.save(sample_dir / "active_neuron_indices.npy", active)
        label_rows.append({"sample_id": sample_id, "label": class_ids[key], "class_sample_index": counts[key], "label_name": f"{key[0]}_{key[1]}"})
        counts[key] += 1
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(label_rows[0])); writer.writeheader(); writer.writerows(label_rows)
    print(f"selected={len(label_rows)} ticks={ticks} output={out}", flush=True)


if __name__ == "__main__":
    main()
