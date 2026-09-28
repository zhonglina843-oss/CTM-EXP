# CTM Connection-Graph Similarity Results Analysis

## 1. Experiment scope

- Dataset: 25 ImageNet samples, 5 classes x 5 samples.
- Ticks: 1, 5, 10, 15, 20, 30, 40, 50.
- Graph preprocessing: Top 1% positive edges, largest connected component.
- Comparison: same-class pairs versus different-class pairs.
- Main separation metric:

  $$
  \Delta_{\mathrm{class}}=\operatorname{mean}(\mathrm{same\ class})-\operatorname{mean}(\mathrm{different\ class})
  $$

A larger positive value means better separation between samples from the same
class and samples from different classes.

## 2. First four methods

| Method | Mean class gap | Median class gap | Best tick | Best gap | Mean same-class similarity | Mean different-class similarity | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Degree-strength cosine | 0.2693 | 0.2904 | 15 | 0.3642 | 0.6791 | 0.4098 | Best first-line method; captures class-specific hub/neuron participation. |
| Edge cosine | 0.2072 | 0.2266 | 15 | 0.2544 | 0.4691 | 0.2618 | Strong edge-level baseline; retains neuron-pair identity and continuous weights. |
| Weighted-Jaccard | 0.0813 | 0.0907 | 30 | 0.1042 | 0.2034 | 0.1221 | Detects shared strong edges, but is sensitive to Top-k selection. |
| Spectral similarity | 0.0214 | 0.0240 | 40 | 0.0411 | 0.8694 | 0.8480 | High absolute similarity but weak class separation; mostly measures global graph shape. |

### First conclusion

For the current fixed-neuron CTM setting, the practical ranking is:

1. `degree_strength_cosine`
2. `edge_cosine`
3. `weighted_jaccard`
4. `spectral_similarity`

The strongest tick is around tick 15 for the two identity-preserving methods.
This suggests that class-specific graph organization is most distinguishable in
the early-to-middle thinking phase rather than only at the final tick.

## 3. Extended methods

| Method | Mean class gap | Median class gap | Best tick | Best gap | Mean same-class similarity | Mean different-class similarity | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `ged_approx` | 0.1168 | 0.1270 | 20 | 0.1411 | 0.3280 | 0.2112 | Useful weighted edge-change baseline; moderate separation. |
| `community_ari` | 0.0913 | 0.1058 | 10 | 0.1276 | 0.3262 | 0.2349 | Community partition carries some class information. |
| `community_nmi` | 0.0772 | 0.0940 | 10 | 0.1138 | 0.3123 | 0.2351 | Similar conclusion as ARI, slightly weaker separation here. |
| `gw_approx` | 0.0224 | 0.0246 | 40 | 0.0361 | 0.8467 | 0.8243 | Weak separation; current histogram proxy is not exact GW. |
| `dynamic_trajectory` | 0.0107 | 0.0063 | 15 | 0.0309 | 0.8095 | 0.7988 | Weak evidence with the current lightweight trajectory proxy. |
| `delta_con_attr` | 0.0081 | 0.0086 | 15 | 0.0135 | 0.8510 | 0.8429 | Current random-probe attribution proxy does not separate classes well. |
| `delta_con` | 0.0045 | 0.0039 | 1 | 0.0081 | 0.0704 | 0.0659 | Very weak separation; scale is not directly comparable with the other scores. |
| `wl_kernel` | 0.0000 | 0.0000 | 1 | 0.0000 | 0.0000 | 0.0000 | Degenerate implementation/output; do not interpret scientifically. |

## 4. Computational cost

The recorded maximum RSS is a process-level value, not an isolated method
allocation. The first run reached about 2.58 GB RSS; the extended run reached
about 3.56 GB RSS. Relative pairwise method times were:

| Method | Mean pairwise time per tick (s) |
|---|---:|
| Spectral similarity | 0.0071 |
| Degree-strength cosine | 0.1410 |
| Edge cosine | 0.5589 |
| Weighted-Jaccard | 0.6781 |
| `gw_approx` | 0.0061 |
| `wl_kernel` | 0.0573 |
| `community_ari` | 0.1336 |
| `community_nmi` | 0.1371 |
| `dynamic_trajectory` | 0.4487 |
| `ged_approx` | 0.5260 |
| `delta_con_attr` | 0.7815 |
| `delta_con` | 0.9403 |

The timing comparison is approximate because state construction is shared and
the process retains all loaded matrices. It is still useful for ranking the
relative cost of the current implementations.

## 5. Recommended use

### Primary similarity metric

Use `degree_strength_cosine` as the primary class-level similarity metric. It
has the largest same/different separation and is less brittle than exact edge
overlap. Use `edge_cosine` as the paired edge-level metric when the identity of
the changed connections matters.

### Supporting metrics

- Use `weighted_jaccard` to report overlap of the strongest selected edges.
- Use `spectral_similarity` only as a global-topology control, not as the main
  class discriminator.
- Use community ARI/NMI as an interpretable module-level supplement.
- Use the current `ged_approx` only as a simple weighted edge-change baseline.

### Methods requiring revision before publication-level interpretation

- `wl_kernel` is currently degenerate and must be fixed before use. The likely
  issue is the label-refinement feature construction producing no shared
  feature mass between graphs.
- `delta_con` and `delta_con_attr` are sparse random-probe approximations, not
  exact dense DeltaCon-ATTR. Their present class separation is weak, so they
  should not replace the edge/degree metrics yet.
- `gw_approx` is a histogram proxy, not optimal-transport Gromov-Wasserstein.
- `dynamic_trajectory` currently compares changes between requested ticks with
  a lightweight proxy; it is not a full trajectory alignment method.

## 6. Important limitations

1. The dataset has only 5 classes and 5 samples per class. The ranking is a
   pilot result, not a general claim about ImageNet or CTM.
2. All methods use the same Top 1% edge threshold and largest-component rule;
   results should be checked at several densities before final selection.
3. The first four methods use fixed neuron IDs. This is appropriate for the
   current CTM matrices, but not automatically transferable to cross-model or
   neuron-permutation comparisons.
4. The current results compare graph similarity, not continual-learning
   retention. The next experiment should correlate similarity with final
   accuracy, convergence tick, and forgetting after an update.

## 7. Next experiment

Run a density sensitivity check at 0.5%, 1%, and 2%, then evaluate whether the
same method ranking persists. The most promising compact comparison is:

```text
degree-strength cosine  +  edge cosine  +  community ARI
```

For a plasticity/update mechanism, use degree-strength or DeltaCon attribution
to select candidate neurons, but use the direct weighted edge difference

$$
U_{ij}=\left|S_{ij}^{\mathrm{new}}-S_{ij}^{\mathrm{old}}\right|(w_i+w_j)
$$

as a separate edge-update score. This keeps graph similarity and update-region
selection conceptually distinct.
