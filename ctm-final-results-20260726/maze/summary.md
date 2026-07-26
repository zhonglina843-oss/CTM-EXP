# Maze large

- 统计样本数：`32`
- 详细样本文件夹数：`3`，包括 `sample-000`, `sample-008`, `sample-014`。
- 打包前使用的模型/来源：`SakanaAI/ctm-maze-large`
- 注意：`navigation_solved` 比逐步路线准确率更严格；因此这里同时报告路径整体是否解出，以及逐步动作准确率。

## tick 收敛 / 正确性

- `final_navigation_solved`：min 0, median 0, mean 0.344, max 1
- `final_step_accuracy`：min 0.640, median 1, mean 0.988, max 1
- `navigation_solved_tick`：min 0, median 0, mean 12.906, max 61
- `exact_route_tick`：min 0, median 69, mean 55.125, max 72
- `step_accuracy_99_tick`：min 0, median 69, mean 55.125, max 72
- `final_route_stable_tick`：min 7, median 69, mean 59.656, max 74

解释：

- `final_step_accuracy` 中位数为 1，说明按步看大多数样本最终路线预测很准。
- `final_navigation_solved` 中位数为 0，说明严格的“完整导航成功”标准下，仍有不少样本没有完全满足。

## 同步矩阵/同步表示收敛

- `action_cosine_tick`：min 63, median 71, mean 70.531, max 73
- `out_cosine_tick`：min 47, median 72, mean 67.906, max 74
- `full_sync_cosine_tick`：min 58, median 74, mean 72.531, max 75

## 第一个详细样本中的矩阵/query 形状

- `sync_action`：`(75, 528)`
- `sync_out`：`(75, 2080)`
- `queries`：`(75, 512)`
- `post_activations`：`(75, 2048)`
- `probabilities`：`(75, 100, 5)`
- `S_full_active`：`(75, 256, 256)`

## 关键文件

- `convergence_summary.csv`：逐样本收敛/正确性统计表。
- `manifest.json`：checkpoint、模型元信息和矩阵定义。
- `aggregate_convergence.png`：任务级收敛可视化。
- `samples/*/traces.npz`：原始预测、`S_action`、`S_out`、post-activation、query 等。
- `samples/*/S_full_active_all_ticks.npz`：可视化前的精确活跃神经元 `S_full` 矩阵。
- `samples/*/*.gif`：逐 tick 可视化结果。
