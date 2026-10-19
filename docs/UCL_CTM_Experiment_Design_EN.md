# UCL x CTM: Experimental Design for Observing Knowledge-Structure Change

> Status: awaiting PI/group approval. This is a phase-1 research design, not model code, server configuration, credentials, or experimental results.
>
> The detailed engineering contract is in `docs/UCL_CTM_Probe_Implementation_Plan.md`.

## 1. Research question

During unsupervised continual learning (UCL), how does the knowledge structure of a fixed set of old samples change as new data arrive, and can a frozen Continuous Thought Machine (CTM) expose reproducible activity and synchronisation dynamics beyond endpoint-feature drift and each model's native retention measurements?

The operative word is **observe**. In phase 1, CTM is neither an anti-forgetting module nor a term in the loss of any UCL model. The UCL method continues to learn normally; CTM is a fixed-condition dynamic microscope that only reads its representation and records a response.

## 2. Terms and interpretation boundary

| Term | Meaning in this design | What it does not establish |
| --- | --- | --- |
| Continual-learning stage `s` | A checkpoint after the original UCL model has processed new data | Not a CTM internal tick |
| CTM tick `t` | An internal time step within one CTM forward pass | Not a model-training update |
| Knowledge structure | Native representations and sample relations: latent/features, concept posterior, active components, neighbourhoods, clusters | Not a single scalar comparable across all methods |
| Process change | Change in a fixed old sample's activity/synchronisation trajectory across CTM internal time | A changed trajectory is not forgetting |
| Structural change | Reorganisation of sample relations, concept assignment, or synchronisation representation | A structural change is not functional loss |
| Functional forgetting | Degradation in an independent native old-sample retention measure | Must be supported by a native metric |

Accordingly, the defensible claim is a stable association or predictive value between CTM dynamics and knowledge retention/loss. A synchronisation difference alone cannot establish a forgetting mechanism or show that CTM solves forgetting.

## 3. What each model uses to characterise knowledge structure

The three routes deliberately retain three different representational objects. Training and update mechanisms remain native to each model; only the observation contract is shared.

| Route | Native knowledge carrier and tracking mechanism | First CTM input | Native evidence of change/retention | Central question |
| --- | --- | --- | --- | --- |
| AE | Encoder latent manifold and decoder's ability to reconstruct old inputs | 64-dimensional encoder latent through a frozen adapter | Per-sample reconstruction error, latent drift, similarity matrix/k-NN stability | Can the dynamic probe be recorded reliably and add to the simplest representation-drift baseline? |
| CURL | Continuous latent, concept posterior `q(y|x)`, active mixture components, dynamic expansion, and generative replay | Fixed-dimensional continuous latent or latent mean, not the dimension-changing `q(y|x)` | Posterior/entropy or assignment change, fit signal, component-birth event, original replay outcome | When a new concept arrives or a component expands, are old concepts disrupted, coexisting, or relatively separated? |
| InfoUCL | Backbone/projection features and information selection/suppression by InfoDrop | The primary feature level used by the official evaluation; the other level is a preregistered secondary ablation | Official representation-quality evaluation, neighbourhood/similarity stability, InfoDrop on/off contrast | Can internal dynamics reorganise even when endpoint feature quality is similar? |

AE is a minimal validation baseline, not a claim of a complete UCL mechanism. For CURL and InfoUCL, knowledge structure must be grounded in variables actually emitted by the locked upstream implementation; the version, interfaces, and runnable configuration are to be verified before coding.

## 4. CTM as a frozen probe

For a native representation `h_s(x)` made by the source model at checkpoint `s`, the read-only branch is:

```text
Unlabelled sample x
  -> source UCL encoder/backbone -> h_s(x) -> frozen adapter -> frozen CTM
  -> z_s,t(x), sync_s,t(x), post_state_s(x)

Independent native branch
  -> reconstruction / latent / posterior / component event / representation quality
```

The adapter, CTM weights, normalisation statistics, preprocessing, inference mode, tick count, and sample order are all frozen once initial learning finishes. Two preregistered calibration variants are acceptable:

1. an existing CTM checkpoint whose expected input matches the adapter output; or
2. unsupervised calibration of an independent probe using only initial-phase features, for example reconstruction of the adapter input from a pooled CTM state, followed by freezing.

A randomly frozen CTM is useful for interface and logging checks but is not the only scientific baseline. The main study retains both random-frozen and initially calibrated-frozen probes and fixes the comparison before results are inspected.

CTM requires an explicit, non-default `return_trace=True` path. Each record must minimally contain `tick_ids`, selected-tick `z_history`, the actual exported synchronisation representation `sync_history`, `post_state`, and CTM/adapter version hashes. If the implementation exports a synchronisation vector rather than a full matrix, it must be documented and measured as a synchronisation representation, not mislabelled as a synchronisation matrix.

## 5. Shared minimum protocol

**Stream.** The first pass uses MNIST to validate interfaces and analysis logic: unlabelled training data begin with classes `0/1/2`, then introduce `3` and `4` sequentially. Labels only construct the controlled stream and offline evaluation; they never enter the model, adapter, CTM, or training loss. InfoUCL requires a later ColoredMNIST or Split CIFAR-10 validation because InfoDrop cannot be characterised from greyscale MNIST alone.

**Fixed old set.** After initial learning, select `X_old_fixed` once from a held-out test set: 50 samples while debugging and 200 samples per old class when stable. Persist IDs, order, and preprocessing; never replace this set mid-stream.

**Checkpoints.** Save the initial state and one checkpoint per 512 incoming samples, or a precisely equivalent fixed number of optimisation steps. At each checkpoint, run the original model and CTM over the same `X_old_fixed` in `eval` and no-gradient mode. The first version uses 20 ticks and stores ticks `1, 2, 4, 8, 12, 16, 20`. Debug with one seed; establish main evidence with at least three seeds.

**Minimum per-checkpoint record.**

```text
records/{route}/seed_{seed}/checkpoint_{id}/
  manifest.json          # immutable configuration, source versions, hashes, seed
  probe_ids.npy          # fixed old-sample IDs and order
  features.npy           # h_s(x)
  native_metrics.json    # route-specific metrics and events
  ctm_trace.npz          # z_history, sync_history, post_state, tick_ids
```

## 6. Measures and interpretation rules

For an old sample `x`, baseline checkpoint `0`, and subsequent checkpoint `s`, report complementary quantities:

| Quantity | Suggested calculation | Question answered | Interpretation limit |
| --- | --- | --- | --- |
| Feature drift | `1 - cos(h_0(x), h_s(x))`; use L2 only as a sensitivity analysis | Did endpoint representation change? | Does not alone show old knowledge failed |
| Similarity-structure drift | Compare upper triangles of fixed-set similarity matrices and report k-NN overlap | Did sample relations reorganise? | Controls ambiguity from a global feature rotation |
| Trajectory drift | Compare `z_0,t(x)` and `z_s,t(x)` at matched ticks; retain per-tick curves and preregistered pooling | Did the internal activity process change? | Is not forgetting by itself |
| Synchronisation drift | Compare actual `sync` outputs at matched ticks; use normalised Frobenius distance for matrices | Did synchronisation relationships change? | Must respect true tensor shape |
| Dynamic separability | Mean within-group trajectory similarity minus mean between-group similarity | Do CTM dynamics distinguish old/new concepts? | Labels are offline-only |
| Native retention | AE reconstruction/neighbourhood; CURL posterior/fit/replay; InfoUCL official evaluation | Does the source model retain old knowledge? | Necessary evidence for any forgetting claim |

CURL component expansion changes the dimension and semantic meaning of posterior vectors. Unaligned L2 drift of `q(y|x)` across expansion is invalid. Persist identities for retained components where supported, record new components separately, and mark expansion as an event.

Assess CTM's incremental value at the checkpoint level, rather than splitting repeated samples from the same learning stream across train and test sets:

```text
M_native: next_old_knowledge_loss ~ feature_drift + similarity_drift + native_metrics
M_ctm:    M_native terms + trajectory_drift + synchronisation_drift
```

Begin with descriptive statistics, lagged correlations, and regularised linear models split by full seed/stream. Do not overstate pooled p-values from dependent, repeated observations.

## 7. Stages and acceptance gates

| Stage | Work | Gate |
| --- | --- | --- |
| 0: trace foundations | Adapters, CTM trace wrapper, manifest validation, deterministic smoke test | The same fixed input twice yields identical traces, or traces within a declared tolerance |
| 1: AE fast validation | MNIST `0/1/2 -> 3 -> 4` and aligned retention/drift curves | Sample IDs, checkpoints, and traces align; dynamic change is not called causal forgetting |
| 2: CURL event analysis | Leave the original algorithm unchanged and add only hooks; plot component-expansion timelines | Component change cannot corrupt posterior comparison; each event traces to native logs |
| 3: InfoUCL contrast | Change InfoDrop off/on only, holding CTM and all other conditions fixed | Manifests differ only in the preregistered condition and run-generated state |
| 4: cross-route analysis | Standardise plots while preserving route-specific native fields; compare `M_native` and `M_ctm` | Report negative findings, including complete overlap with feature drift |

## 8. First figures and interpretable outcomes

For every route, produce a checkpoint-aligned panel containing old-knowledge retention, feature/similarity drift, trajectory drift, and synchronisation drift. CURL adds component-expansion markers. Raw tables and manifests accompany each panel so it can be reproduced.

| Observed combination | Defensible preliminary interpretation |
| --- | --- |
| Small feature drift, large CTM drift, then later native-retention decline | CTM may expose change earlier than endpoint features; confirm with seeds and prospective analysis |
| Large CTM drift with stable native retention | Likely benign representational reorganisation, not forgetting |
| Higher feature and CTM drift with worse native retention | Multiple observations agree; analyse their association and timing further |
| CTM indices fully track feature drift | CTM may add no information; this is a valid negative result |

## 9. Decisions required before coding

1. Approve CTM as a frozen, read-only observation probe for phase 1, not a loss term or anti-forgetting module.
2. Approve `784-256-64-256-784` for AE fast validation rather than reproducing every implementation detail of the linked AE repository.
3. Approve MNIST `0/1/2 -> 3 -> 4`, a checkpoint every 512 samples/equivalent steps, and 20 CTM ticks for the first stream.
4. Approve the order AE -> CURL -> InfoUCL: reproduce the official training path first, then add observation hooks only.

## 10. Repository and server synchronisation

The current work branch is `ctm-ucl-probe-plan`. This phase only adds design documents. After approval, commit and push this branch, then check out the same branch on the server. Raw traces, checkpoints, and large result directories are ignored; Git stores code, fixed configurations, manifests, compact tables, and figure descriptions only.

```text
Local worktree -> commit -> origin/ctm-ucl-probe-plan <- server checkout
Server large outputs -> compact manifests/summaries/scripts -> commit -> origin -> local pull
```

No password, private key, token, or authentication-bearing server address belongs in Markdown, configuration, shell history, or Git history. Rotate credentials exposed in the conversation before connecting to the server and use a dedicated SSH key.

## 11. Upstream references

- CTM: https://github.com/SakanaAI/continuous-thought-machines
- CURL: https://github.com/google-deepmind/deepmind-research/tree/master/curl
- InfoUCL: https://github.com/learninginvision/InfoUCL
- AE baseline: https://github.com/ashu-tagore/autoencoder-from-scratch

These repositories are implementation references, not instructions that override this experiment design. Lock and verify commits, interfaces, licences, and dependencies before implementation.
