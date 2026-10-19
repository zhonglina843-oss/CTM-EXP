# UCL x CTM: Knowledge-Structure Probe Implementation Plan

## 中文概览

这份方案的核心不是让 CTM 直接减轻遗忘，而是把它固定为一个“动态显微镜”：
UCL 模型照常训练，CTM 只读取模型在每个检查点对同一批旧样本产生的特征，并记录固定
20 个内部 tick 的神经元活动和同步表征。这样才能回答：原模型的知识结构变化，是否先在
CTM 动态里出现了可重复的信号。

三条路线分别保留原模型自己的知识结构定义，CTM 不取代这些信号：

| 路线 | 原模型追踪的知识结构 | 接到 CTM 的位置 | 第一轮必须同时保存的原生证据 |
| --- | --- | --- | --- |
| AE | latent 流形及其重构旧样本的能力 | Encoder 的 64 维 latent | 重构误差、latent 漂移、样本相似关系/k-NN 稳定性 |
| CURL | 连续 latent、概念后验 `q(y | x)`、活跃成分 | 连续 latent，而不是 `q(y | x)` | 后验/熵、拟合度、成分扩展事件、原版回放效果 |
| InfoUCL | Backbone/投影特征及 InfoDrop 的信息筛选 | 官方评估所用的主特征层 | 原版表征质量评估、特征/近邻结构漂移、InfoDrop 开关 |

第一阶段 CTM、输入适配器和归一化统计量都要冻结。每次新数据到来后，仅更新原 UCL
模型；对固定旧测试样本按固定顺序做一次无梯度 CTM 推理。需要共享和固定的是：旧样本、
预处理、tick、随机性和记录格式；不应强行统一的是：三种模型的 loss、网络维度、持续
学习机制和原生指标。

四类共同指标为：最终 feature drift、样本间 similarity-structure drift、CTM trajectory
drift、CTM synchronization drift。它们均为“变化”的量，不是遗忘的同义词。结论必须再由
AE 的旧样本重构/邻域保持、CURL 的概念/拟合/回放指标或 InfoUCL 的原版旧表征评估支撑。
真正检验 CTM 是否有增量价值时，比较“只用原生指标预测下一个检查点的旧知识损失”和
“加入 CTM 两类漂移后预测”的结果，并按完整种子/数据流划分，避免同一路径的样本泄漏。

执行顺序固定为：先用 AE 跑通记录和可重复性检查；再用 CURL 做成分扩展事件附近的
分析；最后用 InfoUCL 做 InfoDrop 开/关的严格单因素对照。MNIST 仅用于快速验证接口，
InfoUCL 的正式验证还应加入 ColoredMNIST 或 Split CIFAR-10。当前分支只包含该设计文档，
尚未连接服务器、推送 GitHub、下载模型或修改任何模型代码。

## 1. Scope and decision

This document is an implementation plan, not a claim that CTM itself solves
unsupervised continual learning (UCL). The first study uses CTM as a **frozen,
read-only probe**. Each UCL method retains its original loss, update rule and
evaluation protocol. CTM receives a representation produced by the UCL model,
unrolls it for a fixed number of internal ticks, and records its dynamics.

The question is deliberately narrower than "does CTM reduce forgetting?":

> On a fixed set of old samples, do CTM dynamics reveal a reproducible change
> in knowledge structure that is complementary to the original model's own
> representation and retention measurements?

The two supplied PDFs are treated as research context and prior planning, not
as instructions that override this document or future user decisions.

### Non-goals of phase 1

- Do not add CTM output to AE, CURL or InfoUCL losses.
- Do not train or reinitialize CTM after the initial calibration step.
- Do not replace the models' native measurements with a universal CTM score.
- Do not compare raw accuracy/reconstruction values across methods that use
  different architectures or data protocols.
- Do not infer forgetting merely because a CTM trajectory has changed.

## 2. Model roles and their native knowledge-structure signals

| Route | Phase-1 model definition | What constitutes its native knowledge structure | Native change/retention signals | CTM's added observation |
| --- | --- | --- | --- | --- |
| AE | Small fully connected MNIST AE: `784 -> 256 -> 64 -> 256 -> 784` (about 0.436M parameters) | A 64-dimensional latent manifold and its ability to reconstruct old inputs | Per-sample reconstruction error, latent drift, pairwise similarity drift, classwise cluster/k-NN separation (evaluation only) | Whether latent changes have an earlier or finer-grained internal trajectory/synchronization signature |
| CURL | Official CURL continuous-latent, concept-mixture model | Continuous latent representation plus posterior concept allocation `q(y | x)` and active mixture components | Latent drift, posterior entropy/assignment drift, fit/likelihood signal, component birth/expansion events, original replay outcome | What changes in the continuous latent dynamic response around concept reassignment or component birth |
| InfoUCL | Official self-supervised backbone with InfoDrop | Backbone/projection representations and the information selected/suppressed by InfoDrop | Representation drift, representation-quality protocol from the official repo, neighbor/cluster stability, InfoDrop on/off contrast | Whether the same representation-quality outcome hides a reorganization of intermediate CTM dynamics |

The three routes answer progressively harder questions: AE validates the probe
and logging contract; CURL tests concept-structure changes; InfoUCL tests the
relationship between feature selection and dynamic stability.

## 3. CTM probe contract

### 3.1 Frozen probe setup

For each source representation `h(x) in R^d`, use a fixed adapter
`a: R^d -> R^d_ctm`, followed by CTM. The adapter may be a single linear layer
with a fixed normalization transform. It is calibrated only on the initial
unlabelled training phase, then frozen alongside CTM and its normalization
statistics before any continual updates begin.

Initial calibration is an engineering prerequisite, not a scientific result.
Accept either of the following, chosen before running the stream:

1. A CTM checkpoint whose expected input matches the adapter output; or
2. Unsupervised probe calibration on initial-task features, e.g. reconstructing
   the adapter input from a pooled CTM representation.

The calibration data must not include labels, future tasks, or the fixed old
test set. Keep a complete manifest of CTM commit/checkpoint, adapter shape,
normalization, tick count, seeds and preprocessing.

### 3.2 Required trace interface

Implement a non-default `return_trace=True` path around the CTM forward pass;
do not change its default prediction behaviour. For batch `B`, it returns:

```python
{
    "tick_ids": [1, 2, 4, 8, 12, 16, 20],
    "z_history": Tensor[checkpoint_tick, B, n_units_or_dynamics_dim],
    "sync_history": Tensor[checkpoint_tick, B, sync_dim],
    "post_state": Tensor[B, post_state_dim],
    "metadata": {"ctm_checkpoint": ..., "adapter_hash": ...},
}
```

`sync_history` means the exact synchronization representation actually
exported by the CTM implementation. If full per-sample synchronization matrices
are available, save a compact matrix representation plus its dimensions; do not
silently substitute a vector and continue to call it a matrix. The existing
repository convention already supports multi-tick `post_state` and `S_full`;
reuse it where possible.

Use deterministic CTM evaluation (eval mode, fixed seeds, no stochastic data
augmentation) and the same tick IDs at every checkpoint.

### 3.3 Model-adapter interface

Every route exposes its own adapter without modifying its learning algorithm:

```python
class UCLAdapter(Protocol):
    name: str
    def train_stream_batch(self, x_unlabelled) -> dict: ...
    def encode(self, x) -> Tensor: ...
    def native_metrics(self, x_old, labels_for_eval=None) -> dict: ...
    def event_metadata(self) -> dict: ...
```

The standardized record schema is intentionally small:

```text
records/{route}/seed_{seed}/checkpoint_{id}/
  manifest.json                 # immutable configuration and source versions
  probe_ids.npy                 # fixed sample IDs, in fixed order
  features.npy                  # h_s(x)
  native_metrics.json           # route-specific scalar/event fields
  ctm_trace.npz                 # z_history, sync_history, post_state, tick_ids
```

Native fields are not forced into identical names. Examples: AE stores
`reconstruction_error`; CURL stores `posterior`, `active_components`,
`component_expanded`; InfoUCL stores its official evaluation metrics and
`infodrop_enabled`.

## 4. Common stream and evaluation protocol

### 4.1 Minimal common protocol

- Debug phase: MNIST, two or three disjoint classes, one seed, 50 fixed old
  probe samples. Purpose: validate the entire trace and comparison pipeline.
- Fast validation: MNIST or Split MNIST, initial classes `0/1/2`, then one
  sequential incoming class per stage; 200 fixed old probe samples after the
  pipeline is stable.
- Checkpoint cadence: initial state plus every 512 new samples (or an exactly
  equivalent fixed number of optimization steps). Record the cadence in the
  manifest rather than assuming batches are comparable across repositories.
- Probe configuration: 20 CTM ticks; retain ticks `1, 2, 4, 8, 12, 16, 20`.
- Final evidence: at least 3 seeds; use 5 only after configuration is fixed.
- Labels: only construct the controlled stream and calculate offline metrics;
  they never enter model, adapter or CTM input.

MNIST establishes the integration only. For InfoUCL, add ColoredMNIST or Split
CIFAR-10 after MNIST because InfoDrop is intended to affect information such as
texture/colour relevance. Do not combine absolute results from heterogeneous
datasets in one ranking.

### 4.2 Fixed comparison set

Select `X_old_fixed` once immediately after initial training. Preserve sample
IDs, order and preprocessing. At each checkpoint `s`, save native feature
`h_s(x)` and CTM trace `T_s(x)` for exactly this same set. Optional fixed
new-task and held-out future-task probe sets may be recorded separately; never
replace the old probe set mid-run.

## 5. Measures of change

For old sample `x`, baseline checkpoint `0`, and update checkpoint `s`:

| Quantity | Definition / implementation | Interpretation boundary |
| --- | --- | --- |
| Feature drift | `D_h(s,x) = 1 - cos(h_0(x), h_s(x))`, with L2 distance as a sensitivity check | Final representation changed; it does not alone mean old knowledge failed |
| Similarity-structure drift | Correlate/compare the upper triangles of fixed-set cosine-similarity matrices `K_0` and `K_s`; also report k-NN overlap | Detects relational reorganization that a global feature rotation can hide |
| Trajectory drift | Fixed-tick mean of `1 - cos(z_0,t(x), z_s,t(x))`, retaining both per-tick and pooled curves | CTM response changed at a defined internal time; not automatically forgetting |
| Synchronization drift | Same fixed-tick comparison of `sync_0,t(x)` and `sync_s,t(x)`; use Frobenius-normalized distance when the representation is a matrix | Change in the selected synchronization representation, respecting its real shape |
| Dynamic separability | `S_dyn = mean(sim(T_i,T_j) within group) - mean(sim(T_i,T_j) between groups)` | A structural discriminability measure; labels are offline-only |
| Native retention | Route-specific old-data metric: AE reconstruction/latent neighborhood; CURL posterior/fit/component and replay metric; InfoUCL official representation evaluation | The functional evidence needed to interpret the other changes |

For CURL, posterior vectors are not directly compared across component-expansion
events. Preserve identities for pre-existing components when the implementation
supports that alignment, store new components separately, and annotate the
event. A dimension-changing posterior cannot have a meaningful unaligned L2
drift.

Use pre-registered trajectory pooling: concatenate normalized selected ticks as
the primary feature, with mean-over-ticks as a sensitivity analysis. Do not
introduce a trainable temporal encoder in the first round.

## 6. Route-specific integration and experiments

### 6.1 Route A: AE + CTM - establish valid observation

**Input point.** Feed the frozen adapter from encoder latent `h(x)`; keep the
decoder and reconstruction loss entirely unchanged.

**Stream.** Train initial AE on MNIST `0/1/2`; store `X_old_fixed`; then update
with class `3`, then `4` (and expand only after the two-stage protocol is
reproducible).

**At every checkpoint.** Save latent vectors, reconstruction error per old
sample, k-NN/cluster evaluation, and the CTM trace. The primary plot is a
checkpoint-aligned panel: old reconstruction retention, feature/similarity
drift, trajectory drift, and synchronization drift.

**Decision criterion.** The probe is technically valid only if a repeated
evaluation of an unchanged checkpoint produces identical (or explicitly
tolerance-bounded) traces. A scientifically interesting result requires a
repeated relationship with independent retention evidence, not merely a
non-zero CTM distance.

**Later ablation, not phase 1.** Compare no replay with a fixed small replay
buffer while leaving the probe unchanged.

### 6.2 Route B: CURL + CTM - concept structure through a stable input

**Input point.** Feed the continuous latent `z(x)` / continuous encoder feature
chosen from the official implementation. Do **not** feed `q(y | x)` to CTM in
the first version because component expansion changes both its dimensionality
and semantics.

**Native event log.** For every checkpoint include posterior distribution,
entropy/assignment summaries, fit signal, active component identifiers and the
boolean/event payload for component expansion. Treat component expansion as a
marked event, not evidence by itself that a stable new concept exists.

**Primary analyses.**

1. Around an old-to-new update, compare old sample `z(x)`, `q(y | x)`, CTM
   trajectory and old-concept grouping stability.
2. Around every expansion event, plot old/new samples immediately before and
   after the event. Test three alternatives: old dynamics are disrupted; old
   and new dynamics coexist but interfere; or new samples form a relatively
   separate dynamic pattern.
3. Once the original path is stable, run the official generative replay switch
   on/off as a single-factor ablation. Compare latent/posterior stability, CTM
   stability and old-concept retention.

The route deliverable is one checkpoint timeline with component events,
posterior/fit change, CTM change and retention on common aligned axes.

### 6.3 Route C: InfoUCL + CTM - information selection and dynamics

**Input point.** Inspect the official code before implementation and preselect
one of `(a)` backbone feature or `(b)` projection-head feature as primary based
on the representation used by that repository's own evaluation. Record the
other location as a declared secondary ablation, not a post-hoc choice.

**Core contrast.** Hold backbone, task order, augmentation, optimizer,
learning-rate schedule, update steps, probe, adapter and seeds fixed. Change
only `InfoDrop = off/on`. CTM does not participate in either training run.

**Primary analyses.** Compare feature and similarity-structure drift, CTM
trajectory/synchronization drift and InfoUCL's native old-feature evaluation.
The valuable discordant result is: end-point features remain useful while CTM
intermediate dynamics reliably reorganize. It still requires retention evidence
before being described as harmful or beneficial.

## 7. Does CTM add information beyond ordinary feature drift?

The main analysis is prospective at the checkpoint level. For each seed and
checkpoint `s`, predict the next checkpoint's old-knowledge loss `F_(s+1)`.

```text
M_native: F_(s+1) ~ feature drift + similarity drift + native metrics
M_ctm:    F_(s+1) ~ M_native terms + trajectory drift + synchronization drift
```

Start with descriptive correlations, lagged correlations and a regularized
linear model. Split by seed/stream run rather than individual samples to avoid
leakage from repeated observations of the same training trajectory. Report
effect size and uncertainty across seeds, not only a pooled p-value. Complex
predictors are deferred until this comparison has enough independent streams.

Success means `M_ctm` provides stable additional predictive/explanatory value;
it does not require CTM to improve the UCL model's final score.

## 8. Controls, deliberate differences, and failure checks

| Control/failure check | Required action |
| --- | --- |
| Probe identity | Freeze CTM, adapter and normalization before continual updates; hash them in every manifest |
| Evaluation determinism | Fixed old samples/order/preprocessing/ticks; eval mode; repeated same-checkpoint trace check |
| Confounding code edits | First run each original training route with output hooks only; no simultaneous loss or update-rule change |
| Label leakage | Assert labels are absent from `train_stream_batch`, `encode` and CTM calls |
| Storage/shape failure | Store array dimensions and tick IDs; reject checkpoint records with a changed fixed-probe order |
| Feature rotation ambiguity | Always pair absolute feature drift with relational similarity-structure drift |
| Component expansion ambiguity | Align existing CURL components where valid; report new components separately |
| Selective reporting | Predeclare tick pooling, feature tap and primary retention measure before full runs |

Do not force AE/CURL/InfoUCL to share a loss, backbone dimension, update
mechanism or single native metric. Those are the causal mechanisms under study.
Only probe conditions and the cross-checkpoint comparison contract are shared.

## 9. Implementation order and acceptance gates

### Stage 0 - repository and trace foundations

1. Add a vendored/reference checkout strategy for the three upstream projects
   without modifying their source in place.
2. Add the adapter protocol, immutable record writer, manifest validator and
   CTM `return_trace` wrapper compatible with the existing `post_state`/
   synchronization exports.
3. Add a synthetic smoke test: fixed feature batch, fixed CTM/adapter, twice
   evaluated, exact/tolerance-bounded equal trace arrays.

**Gate:** one small fixed batch produces valid records and manifest hashes.

### Stage 1 - AE fast validation

1. Implement the specified small FC AE and unlabelled class stream.
2. Produce at least one checkpoint-aligned analysis figure and one raw table.
3. Repeat with three seeds only after the one-seed run is stable.

**Gate:** metrics are reproducible, old-retention and CTM measurements have
aligned sample IDs/checkpoints, and analysis does not claim causality.

### Stage 2 - CURL event-centered analysis

1. Run the official CURL workflow unchanged with trace hooks.
2. Validate continuous-feature extraction and component event serialization.
3. Produce expansion-marked timeline; only then run replay ablation.

**Gate:** component expansion cannot corrupt posterior comparisons, and all
events can be traced to an original-model log entry.

### Stage 3 - InfoUCL on/off contrast

1. Validate official baseline training and select the primary feature tap.
2. Run paired InfoDrop off/on experiments under matched settings.
3. Repeat the comparison on a data setting appropriate for InfoDrop.

**Gate:** run manifests differ only in the declared InfoDrop condition plus
run-generated state; CTM remains identical.

### Stage 4 - cross-route analysis

1. Standardize plots/tables while retaining route-specific native fields.
2. Run the prospective native-vs-native-plus-CTM analysis.
3. Document negative results and cases where CTM simply mirrors feature drift.

## 10. Proposed repository layout and synchronization

```text
CTM-EXP/
  docs/UCL_CTM_Probe_Implementation_Plan.md
  ucl_ctm_probe/
    adapters/            # AE, CURL and InfoUCL wrappers (future)
    probe/               # frozen adapter + CTM trace wrapper (future)
    evaluation/          # drift metrics and aligned analyses (future)
    tests/               # determinism/schema tests (future)
  configs/               # frozen run manifests (future)
  records/               # ignored large runs; manifests may be versioned (future)
```

The current local Git branch is `ctm-ucl-probe-plan`. It contains this planning
document only; no model implementation, server connection, credential storage
or GitHub push has been performed. After approval, create a clean commit,
push this branch to `origin`, and use the same branch on the server. Keep raw
traces/checkpoints out of Git; sync them via a server-side results directory
and version only manifests, scripts, summaries and selected compact tables.

Recommended post-approval workflow:

```text
local working tree -> committed branch -> origin/ctm-ucl-probe-plan
                                      -> server checkout of same branch
server outputs -> compact summaries/manifests -> commit -> origin -> local pull
```

Before the first server sync, rotate the password that was pasted into the
conversation and replace password-based login with a dedicated SSH key stored
outside the repository. Never commit credentials, remote checkpoints or raw
multi-gigabyte traces.

## 11. Approval decisions needed before coding

1. Approve the phase-1 interpretation: CTM is frozen and observational, not a
   training-loss component.
2. Confirm the AE architecture follows the prior plan (`784-256-64-256-784`)
   rather than reproducing the linked convolutional-AE repository verbatim.
3. Confirm whether the branch should remain named `ctm-ucl-probe-plan`.
4. Confirm the phase-1 task stream (default: MNIST `0/1/2 -> 3 -> 4`) and
   checkpoint cadence (default: every 512 samples/equivalent fixed updates).
