#!/usr/bin/env python3
"""Combine known source features and novel downstream features for tracing."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-train-features", type=Path, required=True)
    p.add_argument("--source-test-features", type=Path, required=True)
    p.add_argument("--novel-features", type=Path, required=True)
    p.add_argument("--source-train-manifest", type=Path, required=True)
    p.add_argument("--source-test-manifest", type=Path, required=True)
    p.add_argument("--novel-manifest", type=Path, required=True)
    p.add_argument("--output-features", type=Path, required=True)
    p.add_argument("--output-manifest", type=Path, required=True)
    args = p.parse_args()

    blocks = []
    out_rows = []
    for feature_path, manifest_path, split in [
        (args.source_train_features, args.source_train_manifest, "known_train"),
        (args.source_test_features, args.source_test_manifest, "known_test"),
        (args.novel_features, args.novel_manifest, "novel_test"),
    ]:
        data = np.load(feature_path, allow_pickle=True)
        features = np.asarray(data["features"], dtype=np.float32)
        manifest_rows = rows(manifest_path)
        if len(features) != len(manifest_rows):
            raise ValueError(f"{feature_path}: features={len(features)} manifest={len(manifest_rows)}")
        blocks.append(features)
        for i, row in enumerate(manifest_rows):
            label = row.get("source_label", "") if split != "novel_test" else row.get("global_label", row.get("label", "-1"))
            out_rows.append({"sample_id": f"{split}_{i:06d}", "label": label, "split": split})

    args.output_features.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_features, features=np.concatenate(blocks), labels=np.asarray([int(r["label"]) for r in out_rows]))
    with args.output_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "label", "split"])
        writer.writeheader(); writer.writerows(out_rows)
    print(f"saved features={sum(len(x) for x in blocks)} dim={blocks[0].shape[1]}", flush=True)


if __name__ == "__main__":
    main()
