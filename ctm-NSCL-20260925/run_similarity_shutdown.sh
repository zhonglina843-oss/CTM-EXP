#!/usr/bin/env bash
set -uo pipefail

PY=/root/miniconda3/bin/python
EXP=/root/autodl-tmp/CTM-EXP/ctm-NSCL-20260925
SUBSET=/root/autodl-tmp/ctm-nscl-20260925/s_based_results_small/static_graph_input
OUT=/root/autodl-tmp/ctm-nscl-20260925/s_based_results_small/static_graph_sparse_final
LOG=/root/autodl-tmp/ctm-nscl-20260925/s_based_results_small/logs/10_similarity_final.log

mkdir -p "$OUT" "$(dirname "$LOG")"
echo "[SIMILARITY] start $(date -Is)" | tee "$LOG"
"$PY" "$EXP/extended_s_sparse.py" \
  --subset-dir "$SUBSET" \
  --output-dir "$OUT" \
  --ticks 1,5,10,15,20,30,40,50 \
  --density 0.01 \
  2>&1 | tee -a "$LOG"
status=${PIPESTATUS[0]}
echo "[SIMILARITY] exit_status=$status $(date -Is)" | tee -a "$LOG"
if [ "$status" -eq 0 ]; then
  sync
  echo "[SIMILARITY] completed; shutting down $(date -Is)" | tee -a "$LOG"
  shutdown -h now
else
  echo "[SIMILARITY] failed; server will remain on for inspection" | tee -a "$LOG"
fi
exit "$status"
