#!/usr/bin/env bash
set -euo pipefail

# Run from the continuous-thought-machines checkout.
CTM_REPO="${CTM_REPO:-/root/autodl-tmp/continuous-thought-machines}"
EXP_REPO="${EXP_REPO:-/root/autodl-tmp/CTM-EXP}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
CHECKPOINT="${CHECKPOINT:-/root/autodl-tmp/ctm-dictionary-exp-20260816/self_fe_multik/stage2_checkpoint_freeze_unit/D_2048_k32_encode32_omp_recon/stage2_checkpoint.pt}"
FEATURES="${FEATURES:?Set FEATURES to an .npz with features and labels}"
MANIFEST="${MANIFEST:?Set MANIFEST to a CSV with sample_id,label,split}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/ctm-nscl-20260925/traces}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/root/autodl-tmp/ctm-nscl-20260925/analysis}"

cd "$CTM_REPO"
export PYTHONPATH="${PYTHONPATH:-.}"

"$PYTHON_BIN" "$EXP_REPO/ctm-NSCL-20260925/export_dynamic_traces.py" \
  --features "$FEATURES" \
  --manifest "$MANIFEST" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --device "$DEVICE"

"$PYTHON_BIN" "$EXP_REPO/ctm-NSCL-20260925/dynamic_novelty.py" \
  --trace-dir "$OUTPUT_DIR" \
  --output-dir "$ANALYSIS_DIR"
