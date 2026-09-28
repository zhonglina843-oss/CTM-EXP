#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
EXP_REPO="${EXP_REPO:-/root/autodl-tmp/CTM-EXP}"
TRACE_DIR="${TRACE_DIR:-/root/autodl-tmp/ctm-nscl-20260925/traces}"
ROOT="${ROOT:-/root/autodl-tmp/ctm-nscl-20260925/s_based_results}"
SCRIPT="$EXP_REPO/ctm-NSCL-20260925"
CTM_TICKS="${CTM_TICKS:-1,5,10,15,20,30,40,50}"

mkdir -p "$ROOT/logs" "$ROOT/dynamic_S" "$ROOT/static_S_NSCL" "$ROOT/static_graph_extended" "$ROOT/static_graph_input"

echo "[1/4] S-based dynamic and final-static novelty" | tee "$ROOT/logs/01_s_novelty.log"
"$PYTHON_BIN" "$SCRIPT/s_based_dynamic_analysis.py" \
  --trace-dir "$TRACE_DIR" \
  --output-dir "$ROOT/dynamic_S" \
  --batch-size 32 \
  2>&1 | tee -a "$ROOT/logs/01_s_novelty.log"

echo "[2/4] Materialize checkpoint-matched S graph subset" | tee "$ROOT/logs/02_prepare_graph.log"
"$PYTHON_BIN" "$SCRIPT/prepare_s_graph_subset.py" \
  --trace-dir "$TRACE_DIR" \
  --output-dir "$ROOT/static_graph_input" \
  --ticks "$CTM_TICKS" \
  --classes-per-split 5 \
  --samples-per-class 5 \
  2>&1 | tee -a "$ROOT/logs/02_prepare_graph.log"

echo "[3/4] S-space spectral diagnostic" | tee "$ROOT/logs/03_s_spectral.log"
"$PYTHON_BIN" "$SCRIPT/s_spectral_diagnostic.py" \
  --subset-dir "$ROOT/static_graph_input" \
  --output-dir "$ROOT/static_S_NSCL" \
  --rank 8 \
  2>&1 | tee -a "$ROOT/logs/03_s_spectral.log"

echo "[4/4] Complete extended graph similarity suite" | tee "$ROOT/logs/04_graph_extended.log"
"$PYTHON_BIN" "$SCRIPT/extended_similarity_experiment.py" \
  --result-dir "$ROOT/static_graph_input" \
  --subset-dir "$ROOT/static_graph_input" \
  --output-dir "$ROOT/static_graph_extended" \
  --ticks "$CTM_TICKS" \
  --density 0.01 \
  --component lcc \
  --samples-per-class 5 \
  --delta-probes 8 \
  2>&1 | tee -a "$ROOT/logs/04_graph_extended.log"

echo "DONE $(date -Is)" | tee "$ROOT/logs/99_done.log"
