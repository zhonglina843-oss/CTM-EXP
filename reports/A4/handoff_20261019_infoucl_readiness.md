# A4 Handoff — InfoUCL readiness audit

## Status
- status: ready-for-review (partial source verification)
- owner: A4 InfoUCL audit
- checked: 2026-10-19
- scope: read-only; no upstream checkout, implementation, server access, or commit

## Question answered

InfoUCL has a documented sequential train/evaluate entry point and supports CIFAR-10, CIFAR-100, Tiny-ImageNet, iDigits (including MNIST/SVHN/MNIST-M/SYN), and Split ColoredMNIST. The repository README establishes that InfoDrop is integrated to encourage informative image features and discard less informative ones. The exact evaluation feature tap (backbone vs projection) and InfoDrop module/config switch could not be confirmed from source-file contents in this environment; do not lock the adapter tap based on assumptions.

## Source / version and evidence

- Upstream: [learninginvision/InfoUCL](https://github.com/learninginvision/InfoUCL), default branch `main` as displayed by GitHub; repository page crawled 2026-10-19. A stable commit SHA was not exposed in the fetched page, so version is not locked.
- README path: `README.md`; GitHub page lists repository root files/folders and README content.
- Root paths confirmed by GitHub listing: `train.py`, `test.py`, `configs/`, `models/`, `metrics/`, `datasets/`, `augmentations/`, `utils/`, `environment.yaml`, `LICENSE`.
- Model folders confirmed: `models/backbones/`, `models/optimizers/`, `models/ssl_models/`; model methods listed include `cassle.py`, `der.py`, `finetune.py`, `mixup.py`, `si.py`.
- Metric paths confirmed: `metrics/knn_monitor.py`, `metrics/utils.py`.
- README training command: `CUDA_VISIBLE_DEVICES=<GPU_IDX> python train.py --log_dir <LOG_PATH> -c <CONFIG_PATH>`; documented default config `./configs/simsiam_c10_cassle.yaml`; README states sequential train/evaluate and checkpoints/logs under experiment folder.
- README reevaluation command: `CUDA_VISIBLE_DEVICES=<GPU_IDX> python test.py --log_dir <EXP_PATH>`.
- README environment: Python >=3.7.13, PyTorch >=1.9.1; provides `environment.yaml` and `conda env create -f environment.yaml`.
- Dataset location documented as `../dataset/`; expected folders include `MNIST/raw`, `CIFAR10/cifar-10-batches-py`, `CIFAR100/cifar-100-python`, `IDIGITS/{train,test}`, and `TINYIMG/processed`. Split ColoredMNIST is linked separately; exact local layout not stated in the captured README excerpt.
- Evidence URLs: [README](https://github.com/learninginvision/InfoUCL); [train.py](https://github.com/learninginvision/InfoUCL/blob/main/train.py); [test.py](https://github.com/learninginvision/InfoUCL/blob/main/test.py); [metrics](https://github.com/learninginvision/InfoUCL/tree/main/metrics); [models](https://github.com/learninginvision/InfoUCL/tree/main/models).

## Findings and recommendation

1. **Training/evaluation entry points:** use the documented `train.py` sequential workflow for a reproduction baseline, followed by `test.py` to reevaluate a run. The repository README says the default config is SimSiam + CIFAR-10 + CaSSLe (`simsiam_c10_cassle.yaml`); this is a sensible small/standard first reproduction relative to CIFAR-100/Tiny-ImageNet, but it is not proven fast without reading epoch/task/batch/backbone config values.
2. **InfoDrop:** official README describes its intended role as informative feature selection/suppression and says InfoUCL integrates it into UCL. It does not show its exact insertion layer, equations, enabled flag, or config key. The separate InfoDrop upstream is acknowledged, but do not infer the local implementation from the acknowledgement.
3. **Feature tap:** **unresolved**. The retrieved repository listing confirms `metrics/knn_monitor.py`, but source text for `test.py`, metric implementation, model forward outputs, and config could not be fetched. Therefore there is not enough direct evidence to say evaluation consumes backbone features or projection-head features. Before implementation, inspect the locked upstream `test.py` evaluation call chain and the model output contract. Primary tap should exactly match the tensor consumed by official evaluation; the other tensor can be a preregistered secondary analysis only.
4. **Observation hook recommendation:** hook model outputs at the evaluation feature-extraction boundary, with an optional second hook immediately before/after the projection head for the declared secondary tap. Run frozen `eval()`/no-grad inference over an unchanged fixed probe set at each checkpoint. Record tensor names, dimensions, normalization, and official metric call-site in the adapter manifest. Do not modify InfoDrop loss or update logic for observation-only integration.
5. **Ablation:** compare InfoDrop off/on only after confirming a real official switch or introducing a carefully documented single-factor patch. Keep config, task sequence, initialization, seeds, data order, augmentation, optimizer and update count matched; document any added switch as experiment code rather than upstream behavior.
6. **Dataset:** MNIST is documented under iDigits and can be used as a pipeline smoke test only if its official loader/task setup is preserved. For InfoDrop's claimed texture-bias motivation, a later confirmatory benchmark should use official Split ColoredMNIST or CIFAR-10; do not claim MNIST alone validates information-selection behavior.

## Official behavior vs experiment proposal

- **Official/documented:** sequential UCL training/evaluation; supported listed benchmarks; InfoDrop integrated for informative-feature learning; default README config name `simsiam_c10_cassle.yaml`.
- **Our proposal, not upstream guarantee:** use MNIST for quick instrumentation, freeze a CTM probe, export backbone/projection features at checkpoints, attach trace-only hooks, and perform InfoDrop off/on paired runs. The exact tap and off switch remain pending source inspection.

## Limitations / blockers

GitHub repository README and directory listings were accessible; direct file pages/raw content for `train.py`, `test.py`, YAML, model implementation, augmentation, and metric implementation returned cache-miss/fetch errors. Network access from the local shell was unavailable. Thus I could not verify configuration hyperparameters, exact InfoDrop implementation or flag, evaluation feature tensor, data-loader transforms, checkpoint cadence, or runtime feasibility. No upstream code was locally present in the project.

## Next agent request

Fetch/clone the official repository in an approved network-enabled environment; record `git rev-parse HEAD`; inspect `test.py` -> metric call -> model feature extraction and identify exact tensor used; inspect InfoDrop module/config toggles, CIFAR/MNIST task transforms, and small-run YAML values. Append those code paths and line references here before coding the adapter. See `artifacts/A4/figures/infoucl_feature_tap_dataflow.svg` for the current evidence-bound flow diagram; dashed tap denotes unresolved selection.
