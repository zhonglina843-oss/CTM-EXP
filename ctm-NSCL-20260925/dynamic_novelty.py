#!/usr/bin/env python3
"""CPU-only dynamic known-versus-novel scoring for exported CTM traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--known-train-split", default="known_train")
    p.add_argument("--eval-splits", default="known_test,novel_test")
    p.add_argument("--pca-rank", type=int, default=0,
                   help="0 disables known-subspace residual")
    return p.parse_args()


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.trace_dir / "traces.npz")
    post = normalize(np.asarray(data["post_state"], dtype=np.float64))
    rows = read_rows(args.trace_dir / "manifest.csv")
    if len(rows) != post.shape[0]:
        raise ValueError("manifest and traces have different sample counts")
    labels = np.asarray([int(r["label"]) for r in rows])
    splits = np.asarray([r["split"] for r in rows])
    known_train = splits == args.known_train_split
    if not np.any(known_train):
        raise ValueError(f"no rows with split={args.known_train_split}")

    classes = sorted(set(labels[known_train].tolist()))
    prototypes = np.stack([
        normalize(post[known_train & (labels == cls)].mean(axis=0)) for cls in classes
    ])
    distances = 1.0 - np.einsum("ntd,ctd->nct", post, prototypes)
    min_dist = distances.min(axis=1)
    nearest = distances.mean(axis=2).argmin(axis=1)
    sorted_dist = np.sort(distances, axis=1)
    gap = sorted_dist[:, 1] - sorted_dist[:, 0] if len(classes) > 1 else np.zeros_like(min_dist)

    residual = np.full_like(min_dist, np.nan)
    if args.pca_rank > 0:
        for tick in range(post.shape[1]):
            basis = post[known_train, tick]
            basis -= basis.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(basis, full_matrices=False)
            rank = min(args.pca_rank, vh.shape[0])
            u = vh[:rank].T
            residual[:, tick] = np.sum((post[:, tick] - post[:, tick] @ u @ u.T) ** 2, axis=1)

    # Calibrate each tick at the largest known-train score.
    threshold = np.quantile(min_dist[known_train], 0.95, axis=0)
    novelty = min_dist.mean(axis=1)
    late_start = post.shape[1] // 2
    late = min_dist[:, late_start:].mean(axis=1)
    final = min_dist[:, -1]
    rows_out = []
    for i, row in enumerate(rows):
        rows_out.append({
            "sample_id": row["sample_id"], "label": row["label"], "split": row["split"],
            "novelty_mean": float(novelty[i]), "novelty_late": float(late[i]),
            "novelty_final": float(final[i]), "nearest_known_class": classes[int(nearest[i])],
            "nearest_gap_mean": float(gap[i].mean()),
            "first_out_tick": int(np.argmax(np.convolve(min_dist[i] > threshold, np.ones(3), mode="same") >= 3) + 1)
            if np.any(np.convolve(min_dist[i] > threshold, np.ones(3), mode="same") >= 3) else "",
        })
    with (args.output_dir / "per_sample_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_out[0]))
        writer.writeheader(); writer.writerows(rows_out)
    np.savez_compressed(args.output_dir / "tick_scores.npz", min_distance=min_dist, gap=gap, threshold=threshold)

    metrics = {}
    eval_splits = [x.strip() for x in args.eval_splits.split(",") if x.strip()]
    eval_mask = np.isin(splits, eval_splits)
    y = (splits[eval_mask] == "novel_test").astype(int)
    if np.unique(y).size < 2:
        raise ValueError("evaluation splits must contain both known_test and novel_test")
    for name, score in {"mean": novelty, "late": late, "final": final}.items():
        metrics[name] = {
            "auroc": float(roc_auc_score(y, score[eval_mask])),
            "aupr_novel": float(average_precision_score(y, score[eval_mask])),
        }
        fpr, tpr, _ = roc_curve(y, score[eval_mask])
        hit = np.flatnonzero(fpr >= 0.05)
        metrics[name]["tpr_at_fpr_5"] = float(tpr[hit[0]]) if hit.size else float(tpr[-1])
    metrics["known_train_threshold_95"] = threshold.tolist()
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
