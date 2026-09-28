# Dynamic CTM Novelty Results

## Experiment

This result uses the frozen dictionary CTM checkpoint:

```text
D_2048_k32_encode32_omp_recon/stage2_checkpoint.pt
```

The experiment records the CTM `post_state` at every one of 50 ticks. For each tick, it computes the cosine distance from a sample to the nearest known-class prototype, then aggregates the distance over time.

Data counts:

| Split | Samples |
|---|---:|
| known_train | 11,574 |
| known_test | 500 |
| novel_test | 4,500 |

## Main results

| Score | AUROC | AUPR Novel | TPR at FPR 5% |
|---|---:|---:|---:|
| Mean over all ticks | 0.7562 | 0.9528 | 0.1240 |
| Mean over late ticks | 0.7423 | 0.9507 | 0.1104 |
| Final tick | 0.7421 | 0.9507 | 0.1107 |

## Interpretation

The dynamic score separates novel samples from known samples better than the final-tick score in this run, but the improvement is modest. The strongest simple score is the mean distance over all ticks. The trajectory plot shows a persistent distance gap between `novel_test` and both known splits.

This is evidence that the second scheme can produce a useful new-class signal, not evidence that the current detector is production-ready. The low TPR at 5% FPR means thresholded detection remains difficult. The high AUPR is partly explained by the large novel-test proportion, so AUROC and the fixed-FPR result should receive more weight.

## Files

- `metrics.json`: aggregate metrics.
- `per_sample_scores.csv`: sample-level scores and nearest known class.
- `trajectory_mean.png`: mean distance trajectory with uncertainty bands.
- `roc_pr.png`: ROC and precision-recall curves.
- `tick_heatmap.png`: sample-by-tick novelty heatmap.

The raw `traces.npz` is intentionally not included here because it is approximately 8.6 GB. The server copy remains under `/root/autodl-tmp/ctm-nscl-20260925/traces/`.
