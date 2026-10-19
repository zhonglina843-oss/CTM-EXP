# Handoff: A4 / InfoUCL Source Readiness

## Status
- status: ready-for-review
- date: 2026-10-19
- scope: official-source audit only; no adapter implementation, server access, shared-framework changes, or commit/push
- upstream revision: `learninginvision/InfoUCL`, `main`, commit `deacb18ba2d5d9425fa968f45ef91db59050fd30`
- isolated checkout: `/private/tmp/infoucl-upstream-a4` (not part of the experiment repository)

## Question answered

The official UCL kNN evaluation feature tap is the **backbone pooled feature**, not the projection-head output. The InfoDrop training pathway is implemented in the ResNet-18 IFD backbone and added as an auxiliary SimSiam objective; its coefficient is `train.lambda_info`. The public CIFAR-10 baseline is computationally substantial as written (5 tasks x 200 epochs, batch 256), so it is useful as an upstream-reproduction reference, not a fast smoke configuration.

## Evidence and exact paths

All paths below are relative to the pinned upstream checkout.

- Training entry: `train.py`; YAML parsing, model construction, per-task training, checkpoint save, then per-task kNN evaluation. README command is `python train.py --log_dir <LOG_PATH> -c <CONFIG_PATH>`.
- Reevaluation entry: `test.py`; loads each sorted task checkpoint and loads only `state_dict` entries matching `backbone.` into `model.net.module.backbone` (`test.py:95-110`), then calls `knn_monitor(model.net.module.backbone, ...)` (`test.py:118-122`).
- Official metric extraction: `metrics/knn_monitor.py:24-54` calls `net(data)` in UCL mode, L2-normalizes that returned vector, and uses it for weighted kNN. The supervised path uses `return_features=True`; UCL path does not.
- Feature definition: `models/backbones/ResNet18_Infodrop.py:347-364` runs stem/layers, adaptive average pooling, flattening and returns this vector from `embed`; `forward(..., return_features=False)` applies the classifier head, while `return_features=True` returns the pooled vector (`:366-374`). In the UCL configuration `cl_default=False`, `models/__init__.py:16-20` replaces `fc` with `Identity`, therefore `net(data)` returns the pooled backbone feature. The SimSiam projector is only downstream of the backbone in `models/ssl_models/simsiam.py:99-101`; it is **not** passed to the official kNN evaluator. For ResNet-18 the pooled representation is 512-D.
- Training also evaluates via backbone: `train.py` calls `knn_monitor(model.net.module.backbone, ...)` after each task. This agrees with `test.py`.
- InfoDrop module: `models/backbones/ResNet18_Infodrop.py`; class `Info_Dropout` (`:20+`) estimates local patch similarity and stochastically masks feature locations when `use_info_drop=True`. It is attached at image stem and selected ResNet stages according to `dropout_layers` (`:306-316`, `:347-360`). In the public YAML, `dropout_layers: 0.5`, so only the stem-level `dropout_layers > 0` branch is active; stage checks use thresholds >=1, >=2, etc.
- InfoDrop loss: `models/ssl_models/simsiam.py:107-128` computes standard SimSiam loss, then a second pass through the same backbone with `use_info_dropout=True` and an auxiliary prediction head. It combines as `L_all = L + lambda_info * L_info`. `models/cassle.py:58-61` supplies `args.train.lambda_info`.
- InfoDrop hyperparameters: `models/__init__.py:30-35` sets class parameters from YAML; `Info_Dropout.set_hyperparameter` in `ResNet18_Infodrop.py:74-90`. Public defaults/config: drop_rate 1.25, temperature 0.03, band_width 1.0, radius 2; config also sets `dropout_layers: 0.5`.
- Official off/on knob: keep `model.backbone: resnet18_ifd`; use positive `train.lambda_info: 0.1` for published default and `train.lambda_info: null` for the SimSiam forward's no-InfoDrop-loss branch. `utils/arguments.py:42-49` checks that if `lambda_info` exists, backbone name contains `ifd`; if the key is absent it sets `lambda_info=None`. Avoid changing the backbone type between the paired conditions.
- Public CIFAR-10 config: `configs/simsiam_c10_cassle.yaml`: sequential CIFAR-10, image 32, 5 tasks x 2 classes, ResNet-18 IFD, CaSSLe, 2-layer projection, batch 256, 200 epochs/task, 10 warmup epochs, lambda_info 0.1, 4 workers, kNN k=200. This is not light. TinyImageNet config also has 200 epochs; IDIGITS config uses NNCLR+SI and similarly 200 epochs.
- Data root: `utils/conf.py:17-22` returns `../dataset/`. README documents CIFAR-10 archive and `../dataset/CIFAR10/cifar-10-batches-py`; `datasets/seq_cifar10.py:21-35` loads CIFAR10 with `download=False` on normal train/eval paths. Its auxiliary `not_aug_dataloader` uses `download=True`, but that does not make the main task loaders auto-download. The training CLI's `--download` does not override those explicit calls in this loader.
- MNIST availability: README lists MNIST as part of iDigits, but `datasets/seq_idigits.py:38-60, 87-110` reads preprocessed NPY arrays from `../dataset/IDIGITS/{train,test}` for `mnist`, `mnistm`, `synnum`, `svhn`; this is a four-domain stream, not standalone 0/1/2 MNIST class-incremental training. README links the iDigits archive. ColoredMNIST is listed in README, but there is no corresponding local dataset loader in `datasets/` at this pinned revision, so using it requires an adapter/loader addition and is outside strict reproduction.
- Environment: `environment.yaml` pins an old Python 3.7 / PyTorch 1.9.1 + CUDA 11.1 stack plus many packages. README says Python >=3.7.13, PyTorch >=1.9.1. Environment reproducibility/compatibility with the experiment's current stack has not been tested.

## Recommended experiment decisions

1. Use the **512-D pooled backbone feature** as the primary CTM adapter input, because it is exactly the representation consumed by official UCL evaluation. Do not use projector output as primary. If helpful, log the projector output as a named secondary analysis only; it is an SSL training representation, not this repository's official kNN eval feature.
2. Keep the official backbone, data order, transforms, objective, and task protocol intact. For the InfoDrop paired contrast, positive `lambda_info` vs `null` is the minimal official switch. Preserve the IFD backbone and all other config values. This comparison is not entirely computationally cost-free: the positive condition performs an extra InfoDrop forward path.
3. For pipeline smoke tests, reduce both `stop_at_epoch` and `num_epochs` (e.g. 1-2 epochs/task) in a clearly marked non-reproduction config and set the seed; do not label it as an official baseline. Keep five CIFAR-10 tasks if testing the repository's actual task loader. Then run official-length config only after hooks and checkpoint alignment pass.
4. CIFAR-10 is the most straightforward official dataset path, but must be downloaded/placed separately under the documented relative root. MNIST from iDigits is a packaged 4-domain NPY protocol, not the planned class stream; don't silently call it the same experiment. ColoredMNIST is listed but not wired in this source tree.
5. CTM capture hook should sit on `model.net.module.backbone` output. In eval/no-grad mode the official backbone forward is deterministic (InfoDrop is only requested explicitly by `use_info_dropout=True`, which kNN evaluation does not do). Record the tensor before kNN normalization for native feature drift, and separately record normalized vectors only if needed for metric parity.

## Official behavior vs our proposed behavior

- Official: CaSSLe/SimSiam sequential training; InfoDrop auxiliary loss with a local masked-feature pass; backbone pooled feature for UCL kNN evaluation; task-end `.pth` checkpoints; CIFAR-10 5x2-class protocol.
- Our proposal: add checkpoint-aligned fixed probe sample extraction and frozen CTM trace recording; add a short smoke YAML before official-length run; compare `lambda_info=0.1` vs `null`; use CTM as observer only. These are experiment instrumentation/ablation choices and not claimed as upstream defaults.

## Limitations

- The requested fast setup is not an official existing config. A tiny run alters optimization schedule and must be called an integration smoke test.
- CIFAR-10 code hardcodes `classes=100` in `metrics/knn_monitor.py`; top-1 comparison still works, but task masking/output behavior and the metric's class assumptions merit validation before using logged scores as a benchmark.
- Dataset download was not attempted. Git clone succeeded after network approval; no external dataset archive or Google Drive artifact was downloaded. Training/runtime and the pinned environment were not tested.
- `Info_Dropout` samples stochastic masks in its enabled forward. Determinism must be checked for training ablations; the official evaluation path has it disabled.

## Next step

Proceed to implement the A4 adapter against the pinned checkout contract: export backbone pooled output at each saved task checkpoint; preserve native official kNN evaluation separately; add the agreed fixed-old-probe CTM-only branch; make a small smoke config distinct from baseline. Before claiming official reproduction, run CIFAR-10 official config to completion and archive config, commit SHA, environment, and results.

## Figure

See `artifacts/A4/figures/infoucl_feature_tap_dataflow.svg`. Solid edges describe pinned upstream behavior; dashed edges are proposed observation hooks.
