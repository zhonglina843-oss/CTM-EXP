# Handoff: A0 / AE + frozen CTM integration

## Status
- status: ready-for-review
- branch: `ctm-ucl-experiment-20261019`
- commit: local uncommitted work on top of `a0adef3`
- date: 2026-10-19 (Asia/Hong_Kong)

## Result

The small local MNIST stream now records AE-native features and a frozen CTM trace at all three checkpoints. The A5 reader validates the records, including separate `out` and `action` synchronization vectors.

## Implementation

- `ucl_ctm_probe/probe/ctm_trace.py`: adds `initialize_feature_ctm()` and `calibrate_ctm_reconstructor()`. CTM is trained only to reconstruct initial-stage unlabelled AE features, then frozen with its orthogonal 64-D adapter.
- `ucl_ctm_probe/adapters/ae/run_ctm_smoke.py`: runs MNIST `0/1/2 -> 3 -> 4`, writes the A5 record contract, validates and compares checkpoints, and emits CSV/JSON plus a PNG panel.
- `.gitignore`: ignores raw CTM record arrays under `artifacts/**/records/`.

No upstream CTM or AE source was modified. No labels enter model training. The CTM and adapter hashes are included in each manifest.

## Evidence

Command:

```bash
PYTHONPATH=.:../ctm-official python3 ucl_ctm_probe/adapters/ae/run_ctm_smoke.py \
  --mnist-root '/Users/lina/Desktop/lab/西瓜书及代码库/datasets' \
  --output-dir artifacts/AE_CTM --seed 1 --per-class 128 \
  --old-probe-size 30 --initial-epochs 1 --update-epochs 1
```

The run completed on CPU. The three records passed `load_records()`. Shapes per checkpoint:

- AE features: `[30, 64]`
- CTM activity: `[7, 30, 128]`
- sync out: `[7, 30, 528]`
- sync action: `[7, 30, 528]`

Summary CSV/JSON: `artifacts/AE_CTM/drift_summary.csv`, `artifacts/AE_CTM/drift_summary.json`.
Figure: `artifacts/AE_CTM/figures/ae_ctm_seed01_checkpoint_panel.png`.
Raw trace arrays are ignored by Git under `artifacts/AE_CTM/records/`.

Observed one-seed smoke values:

| Checkpoint | AE reconstruction MSE | Feature drift | CTM trajectory drift | Sync out drift | Sync action drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| after `3` | 0.15926 | 0.00346 | 0.14530 | 0.06882 | 0.06571 |
| after `4` | 0.12307 | 0.01106 | 0.22359 | 0.10829 | 0.10889 |

The random frozen probe's dynamics changed as its AE input changed. Since AE reconstruction improved over this tiny run and the CTM is uncalibrated, these numbers are only an interface/measurement smoke, not evidence of forgetting or scientific value added by CTM.

## Checks

- `python3 -m ucl_ctm_probe.probe.smoke_test`: passed; two identical traces and unchanged CTM state.
- `python3 -m unittest ucl_ctm_probe.tests.test_records_smoke -v`: 5/5 passed (synthetic schema/metric tests).
- A5 `load_records()` validated all three real AE+CTM checkpoint records.
- `git diff --check` and Python compilation passed before the latest rerun.

## Limits and next step

- The table above is from the earlier CPU run with an untrained random CTM: one seed, 128 training examples per stream class, 30 old probes, and one epoch per stage.
- The runner now supports CUDA and initial-feature calibration. A six-probe CPU calibration smoke passed after materializing lazy CTM parameters under `no_grad`, keeping them trainable during calibration.
- CPU calibration-smoke evidence: `artifacts/AE_CTM_calibration_cpu/summary_seed01.json`, `drift_summary_seed01.csv`, and `figures/ae_ctm_seed01_checkpoint_panel.png`. It used 16 samples/class, 6 old probes, and one calibration epoch; this validates the code path only.
- Planned GPU feasibility run: seeds 1-3, 512 examples per stream class, 600 fixed old probes (200 each from classes 0/1/2), 2 initial epochs, 1 update epoch for classes 3 and 4, and 5 CTM calibration epochs. Review runtime and stability before expanding.
- These runs validate feasibility and measurement behavior; a later scientific study should compare calibration choices and relate CTM changes to native retention across independent streams.
- CURL/InfoUCL remain source-readiness audits; their upstream code/versions and runnable configurations still need to be locked before adapter work.
