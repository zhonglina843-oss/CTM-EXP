# QAMNIST

- 统计样本数：`64`
- 详细样本文件夹数：`2`，包括 `sample-000`, `sample-009`。
- 打包前使用的 checkpoint：`/root/autodl-tmp/qamnist-checkpoint/checkpoint_300000.pt`
- 注意：QAMNIST 前面包含数字/问题输入阶段，真正回答阶段从后面的 tick 开始；因此这里单独统计 answer-phase 的稳定性。

## tick 收敛 / 正确性

- `final_correct`：min 0, median 1, mean 0.969, max 1
- `most_certain_correct`：min 0, median 1, mean 0.984, max 1
- `answer_correct_sustained_tick`：min 0, median 122, mean 118.625, max 125
- `most_certain_tick`：min 124, median 128, mean 127.781, max 130
- `answer_final_label_stable_tick`：min 121, median 122, mean 122.594, max 129

解释：

- 多数样本最终回答正确。
- 正确答案稳定出现的中位 tick 约为 122；最高置信度通常更靠后，约第 128 个 tick。

## 同步矩阵/同步表示收敛

- `action_cosine_tick`：min 105, median 126, mean 124.422, max 129
- `out_cosine_tick`：min 130, median 130, mean 130, max 130
- `full_sync_cosine_tick`：min 127, median 129, mean 128.484, max 129

## 第一个详细样本中的矩阵/query 形状

- `sync_action`：`(130, 528)`
- `sync_out`：`(130, 528)`
- `queries`：`(130, 64)`
- `post_activations`：`(130, 1024)`
- `probabilities`：`(130, 10)`
- `predictions`：`(10, 130)`
- `certainties`：`(130, 2)`
- `S_full_active`：`(130, 256, 256)`

## 关键文件

- `convergence_summary.csv`：逐样本收敛/正确性统计表。
- `manifest.json`：checkpoint、模型元信息和矩阵定义。
- `aggregate_convergence.png`：任务级收敛可视化。
- `samples/*/traces.npz`：原始预测、`S_action`、`S_out`、post-activation、query 等。
- `samples/*/S_full_active_all_ticks.npz`：可视化前的精确活跃神经元 `S_full` 矩阵。
- `samples/*/*.gif`：逐 tick 可视化结果。
