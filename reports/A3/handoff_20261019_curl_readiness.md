# Handoff: A3 / CURL readiness audit

## Status
- status: ready-for-review (read-only source audit; not a reproduction)
- branch: `ctm-ucl-experiment-20261019`
- checked local HEAD: `a0adef3` (`Track experiment coordination and status`)
- owner: A3 CURL route
- date: 2026-10-19 (Asia/Hong_Kong)

## Question answered

The official DeepMind CURL code exposes usable observation points for the fixed CTM probe and for tracking component expansion/replay. The source audit supports a route plan, but does not establish that the upstream code has been installed, run, or reproduced locally.

## Source and version lock

- Local workspace: `/Users/lina/Desktop/lab/CTM_Project`; no CURL checkout was found. `ctm-official/` is the CTM repository, not CURL.
- Upstream: [google-deepmind/deepmind-research/tree/master/curl](https://github.com/google-deepmind/deepmind-research/tree/master/curl), inspected through official GitHub/raw URLs on 2026-10-19 (Asia/Hong_Kong).
- Repository branch shown by GitHub: `master`. Exact commit SHA was not resolved in this browser audit; pin it before implementation and rerun this audit against that SHA.
- Paper: [Continual Unsupervised Representation Learning, NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/file/861578d797aeb0634f77aff3f488cca2-Paper.pdf).
- Files inspected: `README.md`, `model.py`, `training.py`, `train_main.py`, `requirements.txt`, repository `LICENSE`; paper sections on model, dynamic expansion, replay, and evaluation.
- Access limitations: GitHub rendered `model.py` and README; official raw endpoints rendered `model.py`, `training.py` (minified in browser view), and `train_main.py`. Some raw endpoints (`layers.py`, `utils.py`, `train_sup.py`, `train_unsup.py`, raw requirements) returned browser internal errors. No claim is made about uninspected lines/files beyond the displayed official content.
- No clone, install, execution, server access, or credential use was performed.

## Official behavior (verified)

1. `curl/model.py` defines `Curl.get_shared_rep(x, is_training)` as the shared encoder output; `infer_cluster(hiddens)` returns the categorical `q(y|x)` distribution; `infer_latent(hiddens, y, use_mean_y=False)` returns `q(z|x,y)`. With `use_mean_y=True`, the code computes component-conditioned latent means and takes their posterior-weighted sum, yielding a fixed-size continuous representation independent of the active-component count.
2. `training.py::setup_training_and_eval_graphs` constructs evaluation quantities including categorical probabilities and latent vectors. The README identifies `train_main.py` as the script for most non-IID/non-supervised experiments; it calls `training.run_training` with MNIST defaults, sequential classes, dynamic expansion enabled, initial `n_y_active=1`, capacity `n_y=30`, and `n_z=32`. These are upstream defaults, not approved experiment settings.
3. `training.py::run_training` includes event-relevant logic: low-likelihood real samples are buffered; after the configured buffer threshold, the best existing component is selected by aggregated `q(y|x)`, its component-specific parameters are copied to the next slot, `n_y_active` increments, and the new component is tuned using a component-constrained objective. This gives a concrete source-level expansion hook at the buffer-to-expansion transition and active-component increment.
4. Generative replay (MGR) is also explicit in `run_training`: it periodically creates generated-data snapshots/buffers from the model and alternates generated versus real training batches. The paper describes the previous model snapshot and component-use-weighted mixture prior. Replay-on/off must be a controlled ablation and its actual flags/cadence must be taken from the pinned code/config.
5. `requirements.txt` rendered by GitHub lists a legacy stack: `tensorflow==1.14.0`, `tensorflow-probability==0.7.0`, `dm-sonnet==1.35`, `numpy==1.16.4`, `tensorflow-datasets==1.2.0`, plus pinned supporting packages. Expect compatibility/environment work on current Python; do not silently modernize dependencies and call it an exact reproduction.
6. The repository root LICENSE is Apache License 2.0; the inspected `model.py` also carries the Apache-2.0 header. Preserve upstream notices and record any modifications if source is vendored or edited.

## Hook recommendation: verified facts vs proposed instrumentation

| Concern | Official code fact | Proposed probe/event hook (not an upstream feature) |
|---|---|---|
| CTM input | Shared representation is available through `get_shared_rep`; posterior-weighted latent mean is available with `infer_latent(..., use_mean_y=True)`. | Primary candidate: posterior-weighted `z_mean(x)` (fixed dimension `n_z`) to preserve a continuous input while reducing hard-assignment jumps; keep shared `h(x)` as a declared alternate/sensitivity route. Confirm which tensor is built in eval graph and exportable at the pinned revision. |
| Cluster posterior | `q(y|x)` is exposed by `infer_cluster`; active probability support is controlled by `n_y_active`, with a fixed maximum `n_y` tensor in the inspected model. | Log full padded vector, active prefix, entropy, argmax ID, `n_y_active`, and component IDs per checkpoint. Compare pre/post expansion only on shared component identities; new component mass/identity separately. Do not use raw posterior L2 across changed semantics. |
| Expansion event | `training.py` buffers low-ELBO/low-likelihood samples, copies a selected component, increments active count, then performs constrained updates. | Log step, threshold, buffer size, selected source ID, newly assigned ID, old/new active count, source snapshot/checkpoint and follow-up fine-tune range at the transition. This needs an observer callback or minimal logging-only patch; it is not a formal callback API verified upstream. |
| Replay | Training logic creates generated sample buffers/snapshots and alternates generated and real batches. | Log batch source (`real`/`generated`), snapshot step, generation mode and active-component prior counts. Replay ablation should change only replay setting after baseline reproduction. |

The CTM should consume a fixed-dimensional continuous representation, not `q(y|x)` in phase one: categorical posterior width may be padded to the maximum capacity, but active support and component meaning change at expansion. Hard-mode `z(x)` is also less suitable as the sole primary tap because posterior assignment changes can cause discontinuous input changes; the posterior-weighted mean is a recommendation, not a built-in experiment choice.

## Training entry point, run notes, and reproducibility status

- Official README entry points: `train_main.py` (other paper experiments; suggested MNIST command), `train_unsup.py` (unsupervised IID benchmark), `train_sup.py` (supervised continual benchmark). The intended UCL route is `train_main.py`, not the contrastive-RL CURL project/repository.
- README command (not run): `PYTHONPATH=../:$PYTHONPATH python3 train_main.py --dataset=mnist` from the upstream `curl/` directory after creating its recommended environment.
- Retrieved source run record: browser-open official directory and README; official `raw.githubusercontent.com` endpoints for `model.py`, `training.py`, `train_main.py`; GitHub-rendered `requirements.txt` and root license; official NeurIPS paper PDF. All were accessed 2026-10-19 Asia/Hong_Kong. Local inventory command: `find . -maxdepth 3 -type f`; no CURL source tree present.
- Execution status: no commands that run CURL; no numerical results, seeds, checkpoint tables, or experiment figures. This handoff's required visual is the structural flow diagram in `artifacts/A3/figures/curl_observation_flow.svg`.

## Follow-up: local startup readiness check (2026-10-19)

- Isolated source acquisition: shallow official repository clone at `/private/tmp/curl-a3-audit.kFswUA/deepmind-research`; `git rev-parse HEAD` = `f5de0ede8430809180254ee957abf36ed62579ef`. The full checkout exposed upstream case-colliding paths on the case-insensitive macOS filesystem; those affected only unrelated `satore/` files. `git diff --quiet HEAD -- curl LICENSE` passed. To avoid using that checkout as a working tree, the exact committed `curl/` and `LICENSE` were exported with `git archive` into `/private/tmp/curl-a3-audit.kFswUA/source/`. No project/upstream files were edited.
- Local interpreter: Python 3.9.6; installed TensorFlow 2.20.0, NumPy 2.0.2, scikit-learn 1.6.1, absl-py 2.3.1. `sonnet`, `tensorflow_probability`, and `tensorflow_datasets` are not installed. Official pinned requirements are TensorFlow 1.14.0, TensorFlow Probability 0.7.0, Sonnet 1.35, NumPy 1.16.4, TFDS 1.2.0, plus supporting pins; the existing runtime is not a compatible reproduction environment.
- Data inventory: `/Users/lina/.keras/datasets/mnist.npz` exists. No `tensorflow_datasets` package or local TFDS MNIST cache directory was found, so this file does not satisfy the unmodified upstream `tfds.load` entry point.
- Syntax check: in-memory `compile()` over 8 upstream `.py` files passed (no `.pyc` generated).
- Startup check (no training): `PYTHONPATH=/private/tmp/curl-a3-audit.kFswUA/source python3 /private/tmp/curl-a3-audit.kFswUA/source/curl/train_main.py --helpshort` exited 1 before flag parsing with `ModuleNotFoundError: No module named 'sonnet'` (import chain: `train_main.py` -> `curl.training` -> `import sonnet as snt`).
- No dependencies installed, no MNIST download attempted, no training run, server untouched, no commit/push. Environment/status diagram: `artifacts/A3/figures/curl_runtime_readiness.svg`.
- Next step required: create a dedicated legacy-compatible environment (or a deliberately validated compatibility port, which must not be called exact reproduction), install the exact pinned stack in isolation, provide/cache MNIST in TFDS 1.2 format, then rerun `--helpshort`, unit tests, and the unmodified official MNIST command. Pin the resolved commit in the experiment manifest.

## Limitations and unresolved issues

- `master` is mutable; exact source commit/tree hash and current branch status must be pinned.
- The browser exposed only selected/minified `training.py` fragments. Before implementing, inspect the pinned source locally around the full `run_training` loop, expansion bookkeeping, save/restore/checkpoint behavior, replay cadence, outputs, and tests.
- We have not confirmed availability of checkpoint export/resume APIs or a stable callback boundary; the likely least-invasive observer hook is around the existing `run_training` evaluation/expansion event code.
- The upstream setup is TensorFlow 1.x / Sonnet 1.x era. Actual Python/CUDA compatibility, dataset download behavior, runtime, and official smoke tests remain unknown.
- CURL's own official measurements include cluster purity/accuracy and latent k-NN error (labels used for offline evaluation); they are not directly comparable to AE or InfoUCL scores.
- The prior design's broad phrase “posterior/fit/replay results” needs concrete fields from the pinned code. Preserve ELBO terms, `q(y|x)`, fixed-dimensional latent, expansion metadata, and replay batch-source logs separately rather than collapsing them into a single retention scalar.

## Next agent request

After G2/AE passes, A3 should obtain an immutable upstream commit, inspect the complete local source and license, reproduce the unmodified `train_main.py` MNIST path on a compatible isolated environment, and deliver a Markdown handoff plus raw summary table and plots. Only then propose a logging-only adapter in `ucl_ctm_probe/adapters/curl/`. Keep the official training objective and update schedule unchanged; request A0 approval before adding any observer patch or choosing replay-on/off as a phase-1 condition.

## References

- Official source directory and README: https://github.com/google-deepmind/deepmind-research/tree/master/curl
- Official model source: https://github.com/google-deepmind/deepmind-research/blob/master/curl/model.py
- Official training source: https://github.com/google-deepmind/deepmind-research/blob/master/curl/training.py
- Official training defaults: https://github.com/google-deepmind/deepmind-research/blob/master/curl/train_main.py
- Official dependency list: https://github.com/google-deepmind/deepmind-research/blob/master/curl/requirements.txt
- Official repository license: https://github.com/google-deepmind/deepmind-research/blob/master/LICENSE
- Rao et al., NeurIPS 2019: https://proceedings.neurips.cc/paper_files/paper/2019/file/861578d797aeb0634f77aff3f488cca2-Paper.pdf
- Local experiment context: `docs/MULTI_AGENT_EXECUTION_PLAN.md`, `docs/UCL_CTM_Experiment_Design_CN.md`, `docs/UCL_CTM_Probe_Implementation_Plan.md`.
