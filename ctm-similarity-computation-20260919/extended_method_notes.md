# Extended Method Notes

These methods reuse the saved 25-sample, 8-tick CTM matrices. They use the same Top 1% positive-edge and largest-connected-component preprocessing as the first four methods.

| Output method | Method | Definition used here | Exact or approximate | Main cost |
|---|---|---|---|---|
| `community_nmi` | Community similarity | Greedy modularity communities, then normalized mutual information | Exact for the selected partition; heuristic community detector | Medium CPU |
| `community_ari` | Community similarity | Same communities, compared with adjusted Rand index | Exact for the selected partition; heuristic community detector | Medium CPU |
| `delta_con` | DeltaCon | Sparse random-probe approximation to $S=(I+\epsilon D-\epsilon A)^{-1}$ | Approximate; 8 random probes | High CPU/memory |
| `delta_con_attr` | DeltaCon attribution | Mean affinity-row difference over shared active neurons | Approximate attribution | High CPU/memory |
| `ged_approx` | Graph edit baseline | $1-\lVert w_a-w_b\rVert_1/(\lVert w_a\rVert_1+\lVert w_b\rVert_1)$ over the union of weighted edges | Approximate weighted edit similarity, not exact GED | Low CPU |
| `wl_kernel` | Weisfeiler-Lehman | Three rounds of degree-plus-neighbor-label refinement, followed by histogram cosine | Deterministic WL-style approximation | Low/medium CPU |
| `gw_approx` | Gromov-Wasserstein baseline | Cosine-like similarity from sampled shortest-path and weighted-degree histograms | Approximate; not optimal-transport GW | Medium CPU |
| `dynamic_trajectory` | Dynamic trajectory | At the first tick, WL similarity; later ticks compare the magnitude of each sample's graph change since the previous requested tick | Lightweight trajectory proxy | Low CPU |

## Interpretation

`delta_con` and `delta_con_attr` are the closest to the original DeltaCon idea, but they are deliberately implemented with sparse linear solves and random probes so the 512-node active graphs remain tractable. `ged_approx` and `gw_approx` are baselines only; they must not be described as exact GED or exact Gromov-Wasserstein results. Exact GED/GW should be reserved for small extracted subgraphs or a later dedicated experiment.

Every method writes the same pairwise, tick-summary, sample-summary, runtime, and 5x5 grid outputs. The extended results are kept in `results_extended/` so they do not overwrite the first four methods.
