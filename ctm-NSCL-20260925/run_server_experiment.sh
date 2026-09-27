#!/usr/bin/env bash
set -euo pipefail

PY=/root/miniconda3/bin/python
CTM=/root/autodl-tmp/continuous-thought-machines
EXP=/root/autodl-tmp/CTM-EXP/ctm-NSCL-20260925
WORK=/root/autodl-tmp/ctm-nscl-20260925
DATA=/root/autodl-tmp/ctm-dictionary-exp-20260816/self_fe_multik

mkdir -p "$WORK/logs" "$WORK/features" "$WORK/traces" "$WORK/analysis"
cd "$CTM"
export PYTHONPATH=.

"$PY" scripts/dictionary_learning/extract_self_resnet_features.py \
  --zip-path /autodl-pub/data/ImageNet100/imagenet100.zip \
  --checkpoint "$DATA/feature_extractor/resnet18_scratch_best.pt" \
  --manifest "$DATA/data/downstream_test.csv" \
  --output "$WORK/features/novel_test_features.npz" \
  --device 0 --batch-size 128 --num-workers 4 --label-column global_label \
  2>&1 | tee "$WORK/logs/01_extract_novel_features.log"

"$PY" "$EXP/prepare_dynamic_input.py" \
  --source-train-features "$DATA/features/source_train_features.npz" \
  --source-test-features "$DATA/features/source_test_features.npz" \
  --novel-features "$WORK/features/novel_test_features.npz" \
  --source-train-manifest "$DATA/data/source_train.csv" \
  --source-test-manifest "$DATA/data/source_test.csv" \
  --novel-manifest "$DATA/data/downstream_test.csv" \
  --output-features "$WORK/features/combined_features.npz" \
  --output-manifest "$WORK/features/combined_manifest.csv" \
  2>&1 | tee "$WORK/logs/02_prepare_input.log"

"$PY" "$EXP/export_dynamic_traces.py" \
  --features "$WORK/features/combined_features.npz" \
  --manifest "$WORK/features/combined_manifest.csv" \
  --checkpoint "$DATA/stage2_checkpoint_freeze_unit/D_2048_k32_encode32_omp_recon/stage2_checkpoint.pt" \
  --base-checkpoint "$CTM/checkpoints/imagenet/extracted/ctm_imagenet_D=4096_T=50_M=25.pt" \
  --dictionary "$DATA/dictionaries/D_2048_k32/dictionary.pt" \
  --output-dir "$WORK/traces" \
  --device cuda:0 --batch-size 64 \
  2>&1 | tee "$WORK/logs/03_export_traces.log"

MPLCONFIGDIR="$WORK/mplconfig" "$PY" "$EXP/dynamic_novelty.py" \
  --trace-dir "$WORK/traces" \
  --output-dir "$WORK/analysis" \
  --known-train-split known_train --eval-splits known_test,novel_test \
  2>&1 | tee "$WORK/logs/04_dynamic_novelty.log"

echo "EXPERIMENT_DONE $(date -Is)" | tee "$WORK/logs/99_done.log"
