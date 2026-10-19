# Handoff: A5 / analysis contract

## Status
- status: ready-for-review
- branch: `ctm-ucl-experiment-20261019`
- commit: not committed (shared worktree; A0 owns integration)
- owner: A5 analysis and record infrastructure
- date: 2026-10-19

## Question answered

Can checkpoint records be read and shape-checked, and can the planned fixed-probe drift quantities be computed while preserving the actual CTM tensor shapes?

## Source and version lock

- Design sources: `docs/UCL_CTM_Experiment_Design_CN.md`, `docs/UCL_CTM_Experiment_Design_EN.md`, `docs/UCL_CTM_Probe_Implementation_Plan.md`.
- Record layout follows the plan: `manifest.json`, `probe_ids.npy`, `features.npy`, `native_metrics.json`, and `ctm_trace.npz`.
- The current CTM integration reports two separate outputs: `out` and `action`. They are serialized as `sync_history` and `action_history`, respectively; no concatenation or substitution occurs.
- No upstream model code or training logic was changed.

## Changes

- `ucl_ctm_probe/schemas/checkpoint-record.schema.json`: JSON Schema for manifest fields, including source mapping and exact array-shape declarations.
- `ucl_ctm_probe/schemas/records.py`: runtime validation for required fields, numeric finiteness, nonempty arrays, unique fixed IDs, strictly ordered ticks, exact shapes, and separately declared `out`/`action` representations.
- `ucl_ctm_probe/evaluation/records.py`: strict reader for one checkpoint and stable traversal of record directories.
- `ucl_ctm_probe/evaluation/metrics.py`: aligned feature drift, fixed-set similarity-structure drift (upper-triangle MAE/correlation and k-NN overlap), trajectory drift (pre-registered normalized-tick concatenation plus per-tick sensitivity), synchronization drift for the primary `out`, and separate action-output drift. Offline dynamic separability is available but does not enter model/training calls.
- `ucl_ctm_probe/tests/test_records_smoke.py`: runnable synthetic smoke tests, including vector and matrix sync shape checks, shape mismatch failures, and offline-only separability.
- `artifacts/A5/tables/synthetic_smoke_metrics.json` and `.csv`; `artifacts/A5/figures/synthetic_smoke_metrics.png`.

## Evidence

- Command: `PYTHONPYCACHEPREFIX=/tmp/ucl-ctm-pyc MPLCONFIGDIR=/tmp/ucl-ctm-mpl XDG_CACHE_HOME=/tmp/ucl-ctm-cache python3 -m unittest ucl_ctm_probe.tests.test_records_smoke -v`
- Result: 5 tests passed.
- Additional checks: `python3 -m json.tool ucl_ctm_probe/schemas/checkpoint-record.schema.json` and `git diff --check` passed.
- Raw synthetic table: `artifacts/A5/tables/synthetic_smoke_metrics.csv` (full JSON: `artifacts/A5/tables/synthetic_smoke_metrics.json`).
- Figure: `artifacts/A5/figures/synthetic_smoke_metrics.png`.
- The generated metrics and figure are **synthetic pipeline-validation outputs only**. They are not model measurements, training results, or scientific evidence.

## Findings

- The reader rejects missing files, non-finite or empty arrays, duplicated sample IDs, malformed tick order, manifest/array shape mismatches, and inconsistent fixed-probe metadata before calculation.
- The contract stores `sync_history` (`out`) and `action_history` (`action`) independently. Comparison reports separate shapes and drift values for both outputs.
- Synchronization arrays keep the exported rank. Rank-3 vector data use same-tick cosine drift; rank-4 matrices use baseline-relative normalized Frobenius distance. Higher-rank arrays are treated as tensors, not mislabeled matrices.
- Metrics are descriptive changes, not forgetting scores. Functional-retention conclusions still require the route-specific native evidence described in the design documents.

## Limitations and unresolved issues

- No real AE/CURL/InfoUCL records exist in this A5 deliverable, so no experimental effect is estimated.
- The A1 trace producer must use the canonical NPZ keys `sync_history` for CTM `out` and `action_history` for CTM `action`; these names and mappings are fixed in the manifest contract.
- If the CTM implementation's `out` or `action` is not a per-sample vector/matrix-like tensor with leading `[T,B,...]`, revise the metric selection while preserving its true shape; do not flatten without an explicit, justified representation transform.
- `dynamic_separability` is implemented as an offline descriptive helper, but has no plot or aggregate report yet.
- No prospective `M_native` versus `M_ctm` model is implemented; that requires real independent seeds/streams and aligned native retention outcomes.

## Next agent request

- A0: review the canonical `out`/`action` mapping with A1, then freeze the schema version before A2 starts writing records.
- A2: emit records with the required manifest shape fields and the two CTM arrays separately; run the A5 reader against one real checkpoint before a longer experiment.
- A6: audit cross-checkpoint fixed IDs/ticks, real tensor shapes, and that no synthetic artifact is described as an experimental result.
