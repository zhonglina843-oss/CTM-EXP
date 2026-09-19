#!/usr/bin/env python3
"""Low-cost graph similarity benchmark for saved CTM S_full matrices.

The input format follows the existing ImageNet grid analysis:
  result_dir/samples/<sample_id>/S_full_active_all_ticks.npz
  result_dir/samples/<sample_id>/active_neuron_indices.npy
  subset_dir/labels.csv

The script is CPU-only. It writes pairwise same/different-class metrics,
per-sample metrics, leave-one-out prototype grids, and runtime records.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


METHODS = (
    "weighted_jaccard",
    "edge_cosine",
    "degree_strength_cosine",
    "spectral_similarity",
)


@dataclass
class Sample:
    sample_id: str
    label: int
    class_sample_index: int
    label_name: str
    active_ids: np.ndarray
    matrices: np.ndarray


def read_labels(path: Path, samples_per_class: int) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["class_sample_index"] = int(row.get("class_sample_index", 0))
    rows.sort(key=lambda row: (row["label"], row["class_sample_index"]))
    selected: list[dict] = []
    counts: dict[int, int] = {}
    for row in rows:
        if counts.get(row["label"], 0) < samples_per_class:
            selected.append(row)
            counts[row["label"]] = counts.get(row["label"], 0) + 1
    if not selected or min(counts.values()) < samples_per_class:
        raise ValueError(f"expected {samples_per_class} samples per class, got {counts}")
    return selected


def load_samples(result_dir: Path, rows: list[dict]) -> list[Sample]:
    samples = []
    for row in rows:
        directory = result_dir / "samples" / row["sample_id"]
        matrix_path = directory / "S_full_active_all_ticks.npz"
        active_path = directory / "active_neuron_indices.npy"
        if not matrix_path.exists() or not active_path.exists():
            raise FileNotFoundError(f"missing files for {row['sample_id']}: {directory}")
        matrices = np.asarray(np.load(matrix_path)["matrices"], dtype=np.float64)
        active_ids = np.asarray(np.load(active_path), dtype=np.int64)
        if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
            raise ValueError(f"invalid matrix shape for {row['sample_id']}: {matrices.shape}")
        if matrices.shape[1] != active_ids.size:
            raise ValueError(f"matrix/active mismatch for {row['sample_id']}")
        samples.append(Sample(
            row["sample_id"], row["label"], row["class_sample_index"],
            row.get("label_name", str(row["label"])), active_ids, matrices,
        ))
    return samples


def select_graph(matrix: np.ndarray, active_ids: np.ndarray, density: float, component: str):
    n = matrix.shape[0]
    ii, jj = np.triu_indices(n, k=1)
    weights = np.maximum(matrix[ii, jj], 0.0)
    valid = np.isfinite(weights) & (weights > 0)
    ii, jj, weights = ii[valid], jj[valid], weights[valid]
    k = min(max(1, int(round(n * (n - 1) / 2 * density))), weights.size)
    if k:
        chosen = np.argpartition(weights, -k)[-k:]
        ii, jj, weights = ii[chosen], jj[chosen], weights[chosen]
    else:
        ii = jj = np.array([], dtype=np.int64)
        weights = np.array([], dtype=np.float64)

    if component == "all":
        keep = np.ones(n, dtype=bool)
    else:
        # ponytail: one BFS over the sparse Top-k graph; replace with a graph
        # library only if later methods need richer graph algorithms.
        adjacency = [[] for _ in range(n)]
        for a, b in zip(ii, jj):
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))
        seen = np.zeros(n, dtype=bool)
        best: list[int] = []
        for start in range(n):
            if seen[start]:
                continue
            stack = [start]
            seen[start] = True
            current = []
            while stack:
                node = stack.pop()
                current.append(node)
                for neighbor in adjacency[node]:
                    if not seen[neighbor]:
                        seen[neighbor] = True
                        stack.append(neighbor)
            if len(current) > len(best):
                best = current
        keep = np.zeros(n, dtype=bool)
        keep[best] = True
        edge_keep = keep[ii] & keep[jj]
        ii, jj, weights = ii[edge_keep], jj[edge_keep], weights[edge_keep]
    return active_ids[keep], ii, jj, weights, keep


def edge_map(ii: np.ndarray, jj: np.ndarray, weights: np.ndarray, active_ids: np.ndarray):
    return {
        tuple(sorted((int(active_ids[i]), int(active_ids[j])))): float(weight)
        for i, j, weight in zip(ii, jj, weights)
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def compare_graphs(left: dict, right: dict, method: str) -> float:
    if method == "weighted_jaccard":
        keys = set(left["edges"]) | set(right["edges"])
        a = np.asarray([left["edges"].get(k, 0.0) for k in keys])
        b = np.asarray([right["edges"].get(k, 0.0) for k in keys])
        denominator = float(np.maximum(a, b).sum())
        return float(np.minimum(a, b).sum() / denominator) if denominator else 1.0
    if method == "edge_cosine":
        keys = set(left["edges"]) | set(right["edges"])
        a = np.asarray([left["edges"].get(k, 0.0) for k in keys])
        b = np.asarray([right["edges"].get(k, 0.0) for k in keys])
        return cosine(a, b)
    if method == "degree_strength_cosine":
        keys = sorted(set(left["degree"]) | set(right["degree"]))
        a = np.asarray([left["degree"].get(k, 0.0) for k in keys])
        b = np.asarray([right["degree"].get(k, 0.0) for k in keys])
        return cosine(a, b)
    if method == "spectral_similarity":
        a = left["spectrum"]
        b = right["spectrum"]
        scale = float(np.linalg.norm(a) + np.linalg.norm(b))
        distance = float(np.linalg.norm(a - b) / (scale + 1e-12))
        return 1.0 / (1.0 + distance)
    raise ValueError(f"unknown method: {method}")


def build_state(sample: Sample, tick: int, density: float, component: str) -> dict:
    matrix = sample.matrices[tick]
    active_ids, ii, jj, weights, keep = select_graph(matrix, sample.active_ids, density, component)
    local_ids = sample.active_ids[keep]
    n = local_ids.size
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i, j, weight in zip(ii, jj, weights):
        # ii/jj refer to original local indices; remap after LCC filtering.
        left = int(np.flatnonzero(local_ids == sample.active_ids[i])[0])
        right = int(np.flatnonzero(local_ids == sample.active_ids[j])[0])
        adjacency[left, right] = adjacency[right, left] = weight
    degree = adjacency.sum(axis=1)
    spectrum = np.linalg.eigvalsh(adjacency)[::-1] if n else np.zeros(1)
    return {
        "edges": edge_map(ii, jj, weights, sample.active_ids),
        "degree": {int(node): float(value) for node, value in zip(active_ids, degree)},
        "spectrum": resample(spectrum, 64),
        "n_nodes": int(active_ids.size),
        "n_edges": int(weights.size),
    }


def resample(values: np.ndarray, size: int) -> np.ndarray:
    if values.size == size:
        return values
    if values.size == 0:
        return np.zeros(size, dtype=np.float64)
    return np.interp(np.linspace(0, 1, size), np.linspace(0, 1, values.size), values)


def mean_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": "", "median": "", "std": "", "n": 0}
    data = np.asarray(values, dtype=float)
    return {"mean": float(data.mean()), "median": float(np.median(data)), "std": float(data.std()), "n": int(data.size)}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_grid(path: Path, rows: list[dict], samples_per_class: int) -> None:
    by_label = {}
    for row in rows:
        by_label.setdefault(row["label"], {})[row["class_sample_index"]] = row["similarity_to_same_class_mean"]
    output = []
    for label in sorted(by_label):
        output.append({
            "label": label,
            **{f"sample_{index}": by_label[label].get(index, "") for index in range(samples_per_class)},
        })
    write_csv(path, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--subset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ticks", default="1,5,10,15,20,30,40,50")
    parser.add_argument("--density", type=float, default=0.01)
    parser.add_argument("--component", choices=["all", "lcc"], default="lcc")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = set(methods) - set(METHODS)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    ticks = [int(item) for item in args.ticks.split(",") if item.strip()]
    rows = read_labels(args.subset_dir / "labels.csv", args.samples_per_class)
    samples = load_samples(args.result_dir, rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict] = []
    sample_rows: list[dict] = []
    tick_rows: list[dict] = []
    grid_rows: dict[tuple[str, int], list[dict]] = {}
    runtime_rows: list[dict] = []
    for tick_number in ticks:
        tick_index = tick_number - 1
        if any(tick_index < 0 or tick_index >= sample.matrices.shape[0] for sample in samples):
            raise ValueError(f"tick {tick_number} is unavailable for at least one sample")
        states = {}
        state_start = time.perf_counter()
        for sample in samples:
            states[sample.sample_id] = build_state(sample, tick_index, args.density, args.component)
        state_seconds = time.perf_counter() - state_start
        for method in methods:
            start = time.perf_counter()
            same: list[float] = []
            different: list[float] = []
            per_sample: dict[str, dict[str, list[float]]] = {
                sample.sample_id: {"same": [], "different": []} for sample in samples
            }
            for index, left in enumerate(samples):
                for right in samples[index + 1:]:
                    similarity = compare_graphs(states[left.sample_id], states[right.sample_id], method)
                    relation = "same_class" if left.label == right.label else "different_class"
                    (same if relation == "same_class" else different).append(similarity)
                    per_sample[left.sample_id]["same" if relation == "same_class" else "different"].append(similarity)
                    per_sample[right.sample_id]["same" if relation == "same_class" else "different"].append(similarity)
                    pair_rows.append({
                        "method": method, "tick": tick_number, "relation": relation,
                        "sample_a": left.sample_id, "label_a": left.label,
                        "sample_b": right.sample_id, "label_b": right.label,
                        "similarity": similarity,
                    })
            s = mean_stats(same)
            d = mean_stats(different)
            tick_rows.append({
                "method": method, "tick": tick_number,
                "same_mean": s["mean"], "same_median": s["median"], "same_std": s["std"], "same_n": s["n"],
                "different_mean": d["mean"], "different_median": d["median"], "different_std": d["std"], "different_n": d["n"],
                "same_minus_different": float(s["mean"] - d["mean"]) if s["mean"] != "" and d["mean"] != "" else "",
            })
            for sample in samples:
                s_sample = mean_stats(per_sample[sample.sample_id]["same"])
                d_sample = mean_stats(per_sample[sample.sample_id]["different"])
                sample_rows.append({
                    "method": method, "tick": tick_number, "sample_id": sample.sample_id,
                    "label": sample.label, "class_sample_index": sample.class_sample_index,
                    "same_mean": s_sample["mean"], "different_mean": d_sample["mean"],
                    "same_minus_different": float(s_sample["mean"] - d_sample["mean"])
                    if s_sample["mean"] != "" and d_sample["mean"] != "" else "",
                })
            # Grid values: each sample compared with the other four samples in
            # its class, preserving the same row/column layout as the figure.
            for label in sorted({sample.label for sample in samples}):
                class_samples = [sample for sample in samples if sample.label == label]
                for sample in class_samples:
                    prototype = [peer for peer in class_samples if peer.sample_id != sample.sample_id]
                    values = [compare_graphs(states[sample.sample_id], states[peer.sample_id], method) for peer in prototype]
                    grid_rows.setdefault((method, tick_number), []).append({
                        "label": label, "sample_id": sample.sample_id,
                        "class_sample_index": sample.class_sample_index,
                        "similarity_to_same_class_mean": float(np.mean(values)),
                    })
            runtime_rows.append({
                "method": method, "tick": tick_number,
                "state_build_seconds": state_seconds,
                "method_pairwise_seconds": time.perf_counter() - start,
                "n_samples": len(samples), "n_pairs": len(samples) * (len(samples) - 1) // 2,
                "mean_nodes": float(np.mean([states[s.sample_id]["n_nodes"] for s in samples])),
                "mean_edges": float(np.mean([states[s.sample_id]["n_edges"] for s in samples])),
                "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            })
    write_csv(args.output_dir / "pairwise_similarity.csv", pair_rows)
    write_csv(args.output_dir / "sample_similarity.csv", sample_rows)
    write_csv(args.output_dir / "tick_summary.csv", tick_rows)
    write_csv(args.output_dir / "runtime_metrics.csv", runtime_rows)
    for (method, tick), grid in grid_rows.items():
        write_grid(
            args.output_dir / f"grid_{method}_tick-{tick:03d}.csv",
            grid,
            args.samples_per_class,
        )
    (args.output_dir / "run_manifest.json").write_text(json.dumps({
        "methods": methods, "ticks": ticks, "density": args.density,
        "component": args.component, "samples_per_class": args.samples_per_class,
        "similarity_definition": "pairwise same/different class; grid is mean similarity to the other four same-class samples",
    }, indent=2), encoding="utf-8")
    print(f"DONE samples={len(samples)} ticks={len(ticks)} methods={methods} output={args.output_dir}")


if __name__ == "__main__":
    main()
