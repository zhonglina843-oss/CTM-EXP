#!/usr/bin/env python3
"""Create a balanced small S-analysis subset from the existing trace file."""

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
    p.add_argument("--known-classes", type=int, default=5)
    p.add_argument("--novel-classes", type=int, default=5)
    p.add_argument("--samples-per-class", type=int, default=10)
    args = p.parse_args()

    with (args.trace_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if row["split"] in ("known_train", "known_test", "novel_test"):
            groups[(row["split"], int(row["label"]))].append((index, row))

    known_labels = sorted({label for split, label in groups if split == "known_train"})[:args.known_classes]
    novel_labels = sorted({label for split, label in groups if split == "novel_test"})[:args.novel_classes]
    selected = []
    for split, labels in (("known_train", known_labels), ("known_test", known_labels), ("novel_test", novel_labels)):
        for label in labels:
            selected.extend(groups[(split, label)][:args.samples_per_class])
    if not selected:
        raise ValueError("no samples selected")

    data = np.load(args.trace_dir / "traces.npz", mmap_mode="r")
    indices = np.asarray([index for index, _ in selected], dtype=np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "traces.npz",
        post_state=np.asarray(data["post_state"][indices], dtype=np.float32),
        labels=np.asarray(data["labels"][indices]),
    )
    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "split"])
        writer.writeheader()
        for new_index, (_, row) in enumerate(selected):
            writer.writerow({"sample_id": f"small_{new_index:04d}", "label": row["label"], "split": row["split"]})
    summary = {"known_train": 0, "known_test": 0, "novel_test": 0}
    for _, row in selected:
        summary[row["split"]] += 1
    print({"known_labels": known_labels, "novel_labels": novel_labels, "counts": summary}, flush=True)


if __name__ == "__main__":
    main()
