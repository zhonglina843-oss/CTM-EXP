# Sorting reproduction final checkpoint

- 统计样本数：`64`
- 详细样本文件夹数：`2`，包括 `sample-000`, `sample-007`。
- 打包前使用的 checkpoint：`/root/autodl-tmp/ctm-sort-repro-n30/checkpoint.pt`
- 注意：这个 Sorting 结果来自 reproduction checkpoint，不是官方发布 checkpoint。
- 结构说明：`ContinuousThoughtMachineSORT` 中 `q_proj=None` 且 `n_synch_action=0`，所以该任务的 `S_action/query` 是空数组，这是模型结构决定的，不是导出失败。

## tick 收敛 / 正确性

- `final_exact`：min 0, median 1, mean 0.578, max 1
- `final_token_accuracy`：min 0.033, median 1, mean 0.842, max 1
- `first_exact_sustained_tick`：min 0, median 47, mean 27.766, max 50
- `token_accuracy_99_tick`：min 0, median 47, mean 27.766, max 50
- `final_decoded_stable_tick`：min 46, median 48, mean 47.906, max 50

解释：

- `final_exact` 是整个排序序列完全正确。
- `final_token_accuracy` 是逐 token 正确率。
- 该 reproduction checkpoint 不是满分模型，所以存在部分样本最终没有完全排序正确。

## 同步矩阵/同步表示收敛

- `out_cosine_tick`：min 50, median 50, mean 50, max 50
- `full_sync_cosine_tick`：min 49, median 50, mean 49.969, max 50

## 第一个详细样本中的矩阵/query 形状

- `sync_action`：`(50, 0)`
- `sync_out`：`(50, 32)`
- `queries`：`(50, 0)`
- `post_activations`：`(50, 512)`
- `probabilities`：`(50, 31)`
- `decoded_by_tick`：`(50, 30)`
- `predictions`：`(31, 50)`
- `certainties`：`(50,)`
- `S_full_active`：`(50, 256, 256)`

## 关键文件

- `convergence_summary.csv`：逐样本收敛/正确性统计表。
- `manifest.json`：checkpoint、模型元信息和矩阵定义。
- `aggregate_convergence.png`：任务级收敛可视化。
- `samples/*/traces.npz`：原始预测、`S_out`、post-activation；`S_action/query` 为空占位。
- `samples/*/S_full_active_all_ticks.npz`：可视化前的精确活跃神经元 `S_full` 矩阵。
- `samples/*/*.gif`：逐 tick 可视化结果。
