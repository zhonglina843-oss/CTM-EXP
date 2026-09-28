#!/usr/bin/env python3
"""S-based static/dynamic novelty analysis from an existing CTM trace.

The CTM synchronization matrix S is the embedding used here. post_state is
only used to reconstruct S with the existing cumulative Gram convention.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--known-train-split", default="known_train")
    p.add_argument("--eval-splits", default="known_test,novel_test")
    return p.parse_args()


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_s_from_cumulative(cumulative, count):
    s = cumulative / float(count)
    s = (s + np.swapaxes(s, 1, 2)) * 0.5
    norm = np.linalg.norm(s.reshape(s.shape[0], -1), axis=1)[:, None, None]
    return (s / np.maximum(norm, 1e-8)).astype(np.float32)


def build_prototypes(post, labels, train_mask, classes, batch_size):
    h = post.shape[-1]
    sums = np.zeros((post.shape[1], len(classes), h, h), dtype=np.float32)
    counts = np.zeros(len(classes), dtype=np.int64)
    class_index = {c: i for i, c in enumerate(classes)}
    total_batches = (len(post) + batch_size - 1) // batch_size
    print(f"[S-PROTOTYPES] start samples={len(post)} ticks={post.shape[1]} batches={total_batches}", flush=True)
    started = time.time()
    for batch_number, start in enumerate(range(0, len(post), batch_size), 1):
        end = min(len(post), start + batch_size)
        mask = train_mask[start:end]
        if not mask.any():
            continue
        cumulative = np.zeros((end - start, h, h), dtype=np.float32)
        for tick in range(post.shape[1]):
            x = post[start:end, tick]
            cumulative += np.einsum("bi,bj->bij", x, x, optimize=True)
            s = normalized_s_from_cumulative(cumulative, tick + 1)
            for local in np.flatnonzero(mask):
                ci = class_index[int(labels[start + local])]
                sums[tick, ci] += s[local]
        batch_classes = [class_index[int(labels[start + local])] for local in np.flatnonzero(mask)]
        if batch_classes:
            counts += np.bincount(batch_classes, minlength=len(classes))
        if batch_number == 1 or batch_number % 50 == 0 or batch_number == total_batches:
            elapsed = time.time() - started
            rate = batch_number / max(elapsed, 1e-6)
            remaining = (total_batches - batch_number) / max(rate, 1e-6)
            print(f"[S-PROTOTYPES] batch={batch_number}/{total_batches} elapsed={elapsed:.1f}s eta={remaining:.1f}s", flush=True)
    prototypes = sums / np.maximum(counts[None, :, None, None], 1)
    norms = np.linalg.norm(prototypes.reshape(post.shape[1], len(classes), -1), axis=2)
    prototypes /= np.maximum(norms[:, :, None, None], 1e-8)
    return prototypes.astype(np.float32)


def score_ticks(post, prototypes, batch_size):
    scores = np.empty((len(post), post.shape[1]), dtype=np.float32)
    flat_proto = prototypes.reshape(prototypes.shape[0], prototypes.shape[1], -1)
    h = post.shape[-1]
    total_batches = (len(post) + batch_size - 1) // batch_size
    started = time.time()
    print(f"[S-SCORES] start samples={len(post)} ticks={post.shape[1]} batches={total_batches}", flush=True)
    for batch_number, start in enumerate(range(0, len(post), batch_size), 1):
        end = min(len(post), start + batch_size)
        cumulative = np.zeros((end - start, h, h), dtype=np.float32)
        for tick in range(post.shape[1]):
            x = post[start:end, tick]
            cumulative += np.einsum("bi,bj->bij", x, x, optimize=True)
            s = normalized_s_from_cumulative(cumulative, tick + 1)
            flat_s = s.reshape(len(s), -1)
            scores[start:end, tick] = 1.0 - (flat_s @ flat_proto[tick].T).max(axis=1)
        if batch_number == 1 or batch_number % 50 == 0 or batch_number == total_batches:
            elapsed = time.time() - started
            rate = batch_number / max(elapsed, 1e-6)
            remaining = (total_batches - batch_number) / max(rate, 1e-6)
            print(f"[S-SCORES] batch={batch_number}/{total_batches} elapsed={elapsed:.1f}s eta={remaining:.1f}s", flush=True)
    print(f"[S-SCORES] complete elapsed={time.time() - started:.1f}s", flush=True)
    return scores


def main():
    a = args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[S-LOAD] trace_dir={a.trace_dir} output_dir={a.output_dir}", flush=True)
    data = np.load(a.trace_dir / "traces.npz", mmap_mode="r")
    post = data["post_state"].astype(np.float32)
    print(f"[S-LOAD] using full S neurons={post.shape[-1]}", flush=True)
    metadata = rows(a.trace_dir / "manifest.csv")
    labels = np.asarray([int(r["label"]) for r in metadata])
    splits = np.asarray([r["split"] for r in metadata])
    train = splits == a.known_train_split
    classes = sorted(set(labels[train].tolist()))
    prototypes = build_prototypes(post, labels, train, classes, a.batch_size)
    print(f"[S-PROTOTYPES] complete classes={len(classes)}", flush=True)
    tick_scores = score_ticks(post, prototypes, a.batch_size)
    np.savez_compressed(a.output_dir / "S_tick_novelty.npz", novelty=tick_scores)

    eval_mask = np.isin(splits, [x.strip() for x in a.eval_splits.split(",") if x.strip()])
    y = (splits[eval_mask] == "novel_test").astype(int)
    scores = {
        "S_dynamic_mean": tick_scores.mean(axis=1),
        "S_dynamic_late": tick_scores[:, tick_scores.shape[1] // 2:].mean(axis=1),
        "S_static_final": tick_scores[:, -1],
    }
    metrics = {}
    for name, score in scores.items():
        fpr, tpr, thresholds = roc_curve(y, score[eval_mask])
        valid = np.flatnonzero(fpr <= 0.05)
        metrics[name] = {
            "auroc": float(roc_auc_score(y, score[eval_mask])),
            "aupr_novel": float(average_precision_score(y, score[eval_mask])),
            "max_tpr_at_fpr_le_5": float(tpr[valid].max()) if valid.size else 0.0,
            "threshold_at_max_tpr_fpr_le_5": float(thresholds[valid[np.argmax(tpr[valid])]]) if valid.size else None,
        }
    with (a.output_dir / "S_novelty_scores.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "label", "split", *scores]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for i, row in enumerate(metadata):
            writer.writerow({"sample_id": row["sample_id"], "label": row["label"], "split": row["split"], **{k: float(v[i]) for k, v in scores.items()}})
    (a.output_dir / "S_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[S-OUTPUT] wrote {a.output_dir / 'S_novelty_scores.csv'}", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
