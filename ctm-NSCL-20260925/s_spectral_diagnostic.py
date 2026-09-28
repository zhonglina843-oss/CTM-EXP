#!/usr/bin/env python3
"""Small S-space spectral diagnostic for a checkpoint-matched subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--rank", type=int, default=8)
    a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    with (a.subset_dir / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    features = []
    for row in rows:
        matrices = np.load(a.subset_dir / "samples" / row["sample_id"] / "S_full_active_all_ticks.npz")["matrices"]
        matrix = matrices[-1]
        tri = matrix[np.triu_indices(matrix.shape[0])]
        features.append(tri / max(float(np.linalg.norm(tri)), 1e-8))
    x = np.asarray(features, dtype=np.float64)
    similarity = np.maximum(x @ x.T, 0.0); np.fill_diagonal(similarity, 0.0)
    degree = similarity.sum(axis=1)
    d = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-8)))
    normalized = d @ similarity @ d
    values, vectors = np.linalg.eigh(normalized)
    keep = np.argsort(values)[-min(a.rank, len(values)):]
    embedding = vectors[:, keep] * np.sqrt(np.maximum(values[keep], 0.0))[None, :]
    known = np.asarray([r["label_name"].startswith("known_test_") for r in rows])
    novel = np.asarray([r["label_name"].startswith("novel_test_") for r in rows])
    labels = np.asarray([r["label_name"] for r in rows])
    known_classes = sorted(set(labels[known].tolist()))
    prototypes = np.stack([embedding[known & (labels == c)].mean(axis=0) for c in known_classes])
    novelty = 1.0 - np.max(embedding @ prototypes.T, axis=1)
    mask = known | novel; y = novel[mask].astype(int)
    metrics = {
        "auroc": float(roc_auc_score(y, novelty[mask])),
        "aupr_novel": float(average_precision_score(y, novelty[mask])),
        "rank": int(embedding.shape[1]),
        "nodes": int(len(rows)),
    }
    np.savez_compressed(a.output_dir / "S_spectral_embedding.npz", embedding=embedding, eigenvalues=values[keep], novelty=novelty)
    (a.output_dir / "S_spectral_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
