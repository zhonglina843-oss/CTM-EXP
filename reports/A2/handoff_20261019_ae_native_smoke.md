# Handoff: A2 / AE-native MNIST smoke

## Status

- status: ready-for-review
- branch: `ctm-ucl-experiment-20261019`
- commit: `a0adef3` before this uncommitted A2 delivery
- owner: A2
- date: 2026-10-19 (Asia/Hong_Kong)

## Question answered

Can the agreed minimal AE route train on an offline MNIST class stream and emit
checkpoint-aligned native retention evidence without labels entering the
optimizer? Yes, for a deliberately small CPU smoke run. This is an engineering
validation only; CTM is intentionally not connected yet.

## Source and version lock

- local MNIST (already present; `download=False`):
  `/Users/lina/Desktop/lab/西瓜书及代码库/datasets/MNIST/raw/`
- MNIST file hashes and exact package versions: `artifacts/A2/raw/ae_seed01_manifest.json`
- experiment repository commit before this uncommitted delivery: `a0adef3`
- local CTM reference commit inspected (not modified or invoked): `4a6c9c3`
- environment: Python 3.9.6, PyTorch 2.8.0 CPU, torchvision 0.23.0, NumPy 2.0.2
- command:

  ```bash
  python3 ucl_ctm_probe/adapters/ae/run_smoke.py \
    --mnist-root '/Users/lina/Desktop/lab/西瓜书及代码库/datasets' \
    --output-dir artifacts/A2 --seed 1 --per-class 256 \
    --old-probe-size 50 --initial-epochs 2 --update-epochs 1
  ```

## Changes

- added `ucl_ctm_probe/adapters/ae/adapter.py` and
  `ucl_ctm_probe/adapters/ae/run_smoke.py`
  - `AEAdapter` exposes `train_stream_batch`, `encode`, `native_metrics`, and
    `event_metadata`; it does not impose a shared record schema
  - FC AE: `784 -> 256 -> 64 -> 256 -> 784` (435,536 parameters)
  - Adam, reconstruction MSE, CPU deterministic algorithms, one seed
  - controlled stream: initial `0/1/2`, then `3`, then `4`
  - labels choose the controlled stream and balanced held-out probe only; the
    optimizer receives pixel tensors only
  - a fixed 50-example old probe comes from MNIST test classes `0/1/2`
  - outputs per-checkpoint old-probe reconstruction MSE and drift of the 64-D
    encoder latent from the initial checkpoint
- training-logic changes outside A2 ownership: none
- CTM/schema changes: none

## Evidence

- checkpoint summary (raw CSV): `artifacts/A2/tables/ae_seed01_checkpoints.csv`
- per-sample old-probe values (raw CSV): `artifacts/A2/tables/ae_seed01_per_sample.csv`
- optimizer history (raw CSV): `artifacts/A2/tables/ae_seed01_training.csv`
- compact summary plus final-checkpoint repeat-evaluation check (JSON):
  `artifacts/A2/tables/ae_seed01_summary.json`
- run manifest (JSON): `artifacts/A2/raw/ae_seed01_manifest.json`
- figure: `artifacts/A2/figures/ae_seed01_checkpoint_panel.png`
  - left: mean per-sample reconstruction MSE on the same 50 old test samples
  - right: mean `1 - cosine` between their current and initial 64-D latents
  - input: checkpoint CSV above; script: `run_smoke.py`; seed: 1
- reproducibility check: reran the exact command to ignored
  `artifacts/A2/tmp/repeat/`; `cmp` passed for checkpoint and per-sample CSVs,
  and both PNGs had SHA-256
  `0702ab0b6e61a9214c09730dc6bc832c7b3acf185cf124ce4f9008584c00c17b`.
  The current summary also records zero differences on a repeated native
  evaluation of the unchanged final checkpoint.
- static check: `PYTHONPYCACHEPREFIX=/tmp/a2_pycache python3 -m py_compile
  ucl_ctm_probe/adapters/ae/run_smoke.py` passed; `git diff --check` passed.

## Findings

| Checkpoint | Old-probe reconstruction MSE | Mean latent drift (1 − cosine) |
| --- | ---: | ---: |
| after initial `0/1/2` | 0.07630 | ~0.00000 |
| after `3` update | 0.08644 | 0.00031 |
| after `4` update | 0.09510 | 0.00124 |

In this small, one-seed/no-replay smoke stream, old-probe reconstruction error
increased at each update and latent drift from the initial encoder also
increased. These are native AE observations, not evidence of a CTM signal or a
general claim about catastrophic forgetting.

## Limitations and unresolved issues

- This is a compact smoke run (256 training samples/class, two initial epochs,
  one update epoch, 50 probes, one seed), not the planned fast-validation or
  final protocol.
- No similarity/k-NN metric is written yet, and no checkpoint/trace record
  schema is assumed; A0/A5 must freeze the shared contract first.
- CTM is not attached. A1 must publish the frozen trace contract and A0 must
  select the adapter/calibration protocol before this route can emit CTM
  metrics.
- `artifacts/A2/raw/` is ignored by repository policy; it contains only a
  small manifest and remains locally available for handoff. The compact CSVs
  and PNG are outside the ignored `raw/` and `tmp/` directories.

## Next agent request

1. A0/A5: provide the frozen checkpoint-record and native-metric field names;
   then A2 can emit one record per checkpoint without altering AE training.
2. A1: provide a frozen CTM adapter/trace API for a 64-D latent input and its
   deterministic test. A2 will call it under `torch.inference_mode()` only.
3. A6: audit that `train_labels` are used only in `stream_ids` construction,
   and `test_labels` only in balanced probe selection/reporting.
