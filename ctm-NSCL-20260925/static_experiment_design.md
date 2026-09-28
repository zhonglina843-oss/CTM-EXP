# Static Checkpoint Matched Experiment

## 1. Purpose

The earlier `ctm-similarity-computation-20260919` results are not sufficient for a checkpoint-matched comparison. They were produced from an earlier saved CTM matrix set, not from:

```text
stage2_checkpoint_freeze_unit/D_2048_k32_encode32_omp_recon/stage2_checkpoint.pt
```

This experiment will use the same feature input, the same checkpoint, the same known/novel split, and the same traces for both static and dynamic scores.

Following the CTM interpretation from the meeting, the paper's embedding `F`
is mapped to the CTM synchronization matrix `S` (or a selected low-dimensional
subset of `S`). `post_state` is only used to reconstruct `S`; it is not treated
as the final embedding in the static NSCL experiment.

No GPU is needed if `traces.npz` already exists. The GPU is only needed to regenerate traces from the checkpoint.

## 2. Important terminology

### AUROC

AUROC is the area under the receiver operating characteristic curve. It evaluates the ranking of a scalar novelty score:

```text
known sample -> low novelty score
novel sample -> high novelty score
```

The ROC curve changes the threshold over all possible values and plots true-positive rate against false-positive rate. AUROC is 0.5 for random ranking and 1.0 for perfect ranking. It is an evaluation metric, not a training loss.

### NSCL loss

The paper's NSCL is the training objective in Eq. (4):

```text
L_NSCL = -2 alpha L1 - 2 beta L2
          + alpha^2 L3 + 2 alpha beta L4 + beta^2 L5
```

It is optimized to learn a representation from a graph containing known labeled data and unlabeled novel data. It is not the scalar used in the current dynamic experiment. The current experiment used a frozen CTM `post_state` and cosine distance to known-class prototypes, so it should be called `NSCL-inspired novelty scoring`, not `NSCL loss`.

The paper evaluates the learned representation mainly with novel-data linear-probing error and clustering accuracy, not AUROC. AUROC is appropriate here because our task is different: detect whether an incoming sample is known or novel.

## 3. Current dynamic result

The current run found:

| Score | AUROC |
|---|---:|
| Mean nearest-known distance over all ticks | 0.7562 |
| Mean distance over late ticks | 0.7423 |
| Final tick distance | 0.7421 |

This partially meets the experiment goal. The score is better than random ranking and the novel trajectory stays above the known trajectories. However, it is not yet a reliable open-set detector: the reported TPR at 5% FPR is only about 0.124.

The AUPR-Novel must be interpreted carefully because `novel_test` is much larger than `known_test` in this run. AUROC and fixed-FPR operating points are more informative for the intended use.

The current code's `tpr_at_fpr_5` is a rough point sampled at the first ROC threshold with FPR at or above 5%. A final comparison should report the maximum TPR with FPR <= 5%, plus the threshold, to avoid overstating the operating point.

## 4. Static methods to compare

All static methods use the final CTM representation from the same checkpoint-matched trace set.

### S0: final synchronization-matrix prototype distance

Reconstruct the final cumulative synchronization matrix:

```text
S_T(x) = normalize((1/T) sum_t post_state_t(x) post_state_t(x)^T)
```

Use the valid upper-triangular entries, or a fixed spectral summary, as the
vectorized representation of `S_T`. Build one prototype per known class from
`known_train`:

```text
f_S(x) = vectorize(S_T(x))
mu_c = mean(f_S(x) for x in known_train class c)
novelty_final_S(x) = min_c distance(f_S(x), mu_c)
```

This is the static counterpart of the dynamic detector, with `S` as the CTM
embedding rather than `post_state`.

### S1: final known-subspace residual

Fit PCA/SVD only on final `known_train` `S_T` embeddings and compute:

```text
r(x) = ||z(x) - U_k U_k^T z(x)||^2
```

This is the closest frozen-model diagnostic to the paper's “ignorance space” interpretation. It is not the NSCL loss and does not train a new encoder.

### S2: final CTM connection graph similarity

Reconstruct the final cumulative relation matrix from the saved post-state trajectory using the existing project convention:

```text
S_T = normalize((1/T) sum_t post_state_t post_state_t^T)
```

Then apply the same graph preprocessing as the old similarity branch:

- positive edges only;
- Top 1% edges;
- largest connected component;
- fixed neuron IDs.

For each method, calculate the similarity of an incoming sample to each known class and convert it into novelty:

```text
novelty_graph(x) = 1 - max_c mean_{z in class c} similarity(G_T(x), G_T(z))
```

Run the complete method set from the earlier similarity branch:

1. `weighted_jaccard`;
2. `edge_cosine`;
3. `degree_strength_cosine`;
4. `spectral_similarity`;
5. `community_nmi`;
6. `community_ari`;
7. `delta_con`;
8. `delta_con_attr`;
9. `ged_approx`;
10. `wl_kernel`;
11. `gw_approx`;
12. `dynamic_trajectory`.

The last eight retain the previous extended definitions. They must remain
labeled as approximations where appropriate: `delta_con` uses sparse random
probes, `delta_con_attr` is an attribution proxy, `ged_approx` is not exact
GED, `gw_approx` is not optimal-transport GW, and the existing `wl_kernel`
result was degenerate and should be reported as a diagnostic failure.

The old branch's results suggest degree-strength cosine is the strongest graph baseline, but that ranking must be rechecked under the current checkpoint.

### S3: frozen NSCL-inspired static diagnostic

This is the recommended first use of the term NSCL in the checkpoint-matched experiment:

- use final CTM synchronization matrices `S_T` as the frozen representation `f`;
- construct the known-class reference geometry;
- report prototype distance and known-subspace residual;
- optionally report the graph spectral residual from `S_T`.

It tests whether the representation has the geometric property that NSCL is designed to learn, without falsely claiming that Eq. (4) was optimized.

## 5. How the actual NSCL framework maps to CTM

The prototype-distance version is only a diagnostic. The paper's actual
framework should map as follows:

| Paper object | CTM implementation |
|---|---|
| vertex `x` | one CTM input sample or one augmented view |
| labeled known data `D_l` | known-class CTM inputs with labels |
| unlabeled data `D_u` | incoming samples treated as unlabeled |
| learned feature `f(x)` | a small projection head applied to CTM `post_state`, `S`, or a selected CTM representation |
| positive labeled pair | two samples from the same known class |
| positive unlabeled pair | two augmented views of the same incoming sample |
| graph weight `w_xx'` | Eq. (1): known same-class term weighted by `alpha` plus same-instance augmentation term weighted by `beta` |
| normalized adjacency `A_tilde` | degree-normalized sample/view graph |
| NSCL | Eq. (4), equivalent to low-rank factorization of `A_tilde` |

The paper graph is a graph of samples/views. It is not the same object as the
CTM neuron-pair graph used by the 12 similarity baselines:

```text
NSCL sample graph -> learns or diagnoses the representation space
CTM neuron graph  -> compares or attributes internal computation
```

### Static actual NSCL

Freeze the CTM and use the final synchronization matrix `S_T(x)`. Train only a
small projection `f_theta(S_T(x))` with Eq. (4). Evaluate the learned representation
with both the paper-style novel-data clustering accuracy and our additional
known-versus-novel AUROC.

### Dynamic actual NSCL

Use the per-tick synchronization matrix `S_t(x)` and share the same projection head:

```text
L_dynamic_NSCL(theta) = (1/T) sum_t L_NSCL(f_theta(S_t(x)), A_tilde_t)
```

The first implementation should keep the sample graph definition fixed across
ticks and let only the CTM representation change. A time-expanded graph is a
later extension.

### Data requirement

The current cached features contain one deterministic view per image. That is
enough for the frozen diagnostic, but not enough for the paper-faithful `L2`
term. A faithful run needs raw images and two stochastic augmentations, or two
independently generated feature/CTM views. Adding artificial noise is only an
ablation and must be labeled as such.

### S4: actual static NSCL training, optional second step

If the frozen diagnostic is promising, add a separate paper-faithful experiment:

- train a small projection `f_theta` with Eq. (4);
- use same-image augmentation positives for unlabeled samples;
- use same-known-class positives for labeled samples;
- use known-known, known-unlabeled, and unlabeled-unlabeled negative terms;
- evaluate the resulting representation with both clustering accuracy and open-set AUROC.

This is a new training experiment, not merely a post-processing metric. It requires a clear decision about whether unlabeled novel samples may participate in training. The paper allows that transductive NCD setting; an inductive open-set detector should keep novel evaluation samples out of NSCL training.

## 6. Unified output

The script should write one table with one row per sample and one column per score:

```text
static_dynamic_scores.csv
method_metrics.csv
static_dynamic_summary.json
```

Required methods:

```text
dynamic_mean
dynamic_late
dynamic_final
static_final_vector
static_subspace_residual
static_graph_degree_strength
static_graph_edge_cosine
static_graph_weighted_jaccard
 static_graph_spectral
 static_graph_community_nmi
 static_graph_community_ari
 static_graph_delta_con
 static_graph_delta_con_attr
 static_graph_ged_approx
 static_graph_wl_kernel
 static_graph_gw_approx
 static_graph_dynamic_trajectory
```

Required plots:

1. AUROC bar chart for all methods;
2. ROC curves for all methods;
3. known/novel score distributions;
4. static-vs-dynamic score scatter plot;
5. tick trajectory plot from the existing dynamic run.

## 7. Leakage rules

- Prototypes and PCA bases use only `known_train`.
- Thresholds use only a known calibration split.
- `known_test` and `novel_test` are evaluation only.
- Novel labels are never used to compute a score.
- If actual NSCL training is added, its transductive/inductive setting must be recorded explicitly.

## 8. Proposed execution order

1. Reuse the existing checkpoint-matched `traces.npz`.
2. Run S0 and S1 on CPU.
3. Reconstruct `S_T` and rerun all 12 graph methods on a balanced checkpoint-matched subset.
4. Compare all static scores against `dynamic_mean`.
5. Run static actual NSCL with a frozen CTM and a projection head.
6. Run dynamic actual NSCL with the same head shared across ticks.
7. Only then consider updating CTM parameters.

The first NSCL implementation should stop before updating CTM parameters. The
CTM remains frozen; only the small projection head is trained. This directly
tests the paper's framework while keeping the experiment interpretable.
