# Handoff: A0 / GPU AE+CTM three-seed feasibility run

## Status
- status: completed-feasibility; review before extending
- branch/experiment commit: `ctm-ucl-experiment-20261019` / `86ad11b` at run time
- server checkout: `/root/CTM-EXP`
- date: 2026-10-19 (Asia/Hong_Kong)

## Protocol

- Data: public MNIST, copied from the existing local torchvision raw cache to ignored `/root/CTM-EXP/data/mnist/`.
- Stream: unlabelled AE training on classes `0/1/2` (512 images/class, 2 epochs), then class `3` (512, 1 epoch), then class `4` (512, 1 epoch). Labels construct the stream and offline balanced probe only.
- Fixed old set: MNIST test, classes `0/1/2`, 200/class (600 total).
- Probe: orthogonal frozen 64-D adapter; CTM trained for 5 epochs to reconstruct only the initial-stage unlabelled AE training features, then frozen. Old test probes and future stream classes are excluded from calibration.
- CTM: 20 internal ticks; saved ticks `1,2,4,8,12,16,20`; `z` has 128 units. Official sync `out` and `action` are kept separately as 528-D vectors.
- Seeds: 1, 2, 3. Hardware: RTX 4080 SUPER, CUDA 12.1, PyTorch 2.3.0. CTM source commit: `4a6c9c3a7fb5dc4bca6381cc7883a3b9252c6466`.
- Each runner completed A5 `load_records()` validation on all three checkpoints; raw arrays remain ignored on the server.
- MNIST gzip SHA-256: train images `440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609`; train labels `3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c`; test images `8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6`; test labels `f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6`.

## Aggregate (mean ± sample SD, n=3)

| Checkpoint | Old-test reconstruction MSE | Feature drift | CTM trajectory drift | Sync out drift | Sync action drift |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial `0/1/2` | 0.06294 ± 0.00198 | 0 | 0 | 0 | 0 |
| after class `3` | 0.08322 ± 0.00106 | 0.00377 ± 0.00237 | 0.01653 ± 0.01443 | 0.01438 ± 0.01106 | 0.01888 ± 0.01345 |
| after class `4` | 0.08903 ± 0.00098 | 0.01873 ± 0.00778 | 0.08342 ± 0.07029 | 0.08174 ± 0.07665 | 0.08194 ± 0.03809 |

Similarity-structure upper-triangle MAE was `0.02843 ± 0.01321` after class `3` and `0.07275 ± 0.03020` after class `4`.

## Interpretation boundary

This feasibility run shows that initial-feature calibration, freezing, checkpoint capture, and A5 analysis work on the GPU. Old-sample AE reconstruction worsened over the no-replay class stream. CTM trajectory/synchronization changes also grew, especially after class `4`, but varied substantially across three seeds. This does not establish that CTM adds predictive value or identifies a forgetting mechanism; the stream is short, each class uses 512 training images, and the number of independent runs is three.

## Evidence

- Per-seed summaries and source config: `artifacts/AE_CTM_GPU_multiseed/per_seed/`
- Aggregate CSV/JSON and plot: `artifacts/AE_CTM_GPU_multiseed/summary/`
- Reproducible aggregator: `ucl_ctm_probe/evaluation/summarize_multiseed.py`
- Server raw records: ignored at `artifacts/AE_CTM_GPU_multiseed/records/`
- Public MNIST gzip hashes are recorded in the run log; the dataset itself is ignored and not committed.

## Next

1. Review this feasibility result and decide whether to expand MNIST training samples/epochs or treat the protocol as fixed for the next analysis.
2. Test whether checkpoint-wise CTM drift predicts subsequent native reconstruction degradation across independent seeds/streams; do not compare raw drift magnitudes as proof of added value.
3. A3 has locked CURL but its legacy TensorFlow/Sonnet dependencies are absent from the server base environment. A4 has locked InfoUCL and its true 512-D backbone feature tap and `lambda_info` switch; its official CIFAR-10 config is 200 epochs/task. Plan their isolated environments and smaller smoke configs before full reproductions.
