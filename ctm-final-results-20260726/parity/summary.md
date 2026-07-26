# Parity

- 统计样本数：`64`
- 详细样本文件夹数：`3`，包括 `sample-000`, `sample-016`, `sample-024`。
- 打包前使用的 checkpoint：`/root/autodl-tmp/parity-checkpoint/checkpoint_200000.pt`
- 注意：这里的 correctness 是完整累计 parity 序列是否完全匹配。

## tick 收敛 / 正确性

- `final_accuracy`：min 1, median 1, mean 1, max 1
- `full_correct_sustained_tick`：min 65, median 67, mean 66.734, max 69
- `accuracy_99_tick`：min 65, median 67, mean 66.734, max 69
- `final_sequence_stable_tick`：min 65, median 67, mean 66.734, max 69

解释：

- `full_correct_sustained_tick` 表示从该 tick 开始，完整 parity 序列一直正确直到最终 tick。
- 该任务最终准确率为 1，但一般要到约第 67 个 tick 才稳定地给出完整正确序列。

## 同步矩阵/同步表示收敛

- `action_cosine_tick`：min 59, median 61, mean 61.406, max 64
- `out_cosine_tick`：min 73, median 75, mean 74.594, max 75
- `full_sync_cosine_tick`：min 73, median 73, mean 73, max 73

## 第一个详细样本中的矩阵/query 形状

- `sync_action`：`(75, 528)`
- `sync_out`：`(75, 528)`
- `queries`：`(75, 512)`
- `post_activations`：`(75, 1024)`
- `probabilities`：`(75, 2, 64)`
- `certainties`：`(75,)`
- `S_full_active`：`(75, 256, 256)`

## 关键文件

- `convergence_summary.csv`：逐样本收敛/正确性统计表。
- `manifest.json`：checkpoint、模型元信息和矩阵定义。
- `aggregate_convergence.png`：任务级收敛可视化。
- `samples/*/traces.npz`：原始预测、`S_action`、`S_out`、post-activation、query 等。
- `samples/*/S_full_active_all_ticks.npz`：可视化前的精确活跃神经元 `S_full` 矩阵。
- `samples/*/*.gif`：逐 tick 可视化结果。
