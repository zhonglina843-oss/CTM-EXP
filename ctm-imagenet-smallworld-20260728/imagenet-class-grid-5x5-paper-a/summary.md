# ImageNet validation 带 label 分析结果

- 统计样本数：`25`
- 保存完整 raw 矩阵/GIF 的详细样本数：`25`
- final top-1 accuracy：`0.640`
- correct_sustained_tick 中位数：`4.5`

## 关键文件

- `convergence_summary.csv`：逐样本正确性 tick、最终预测、同步矩阵收敛 tick。
- `imagenet_sample_labels.csv`：样本 ID、ground-truth label、最终预测类别和是否正确。
- `samples/*/traces.npz`：详细样本的预测、概率、`S_action`、`S_out`、query、post-activation 和 `correct_by_tick`。
- `samples/*/S_full_active_all_ticks.npz`：详细样本每个 tick 的活跃神经元 `S_full` 精确子矩阵。
- `samples/*/*.gif`：query、S_full、S_action/S_out/S_full 和 labeled result 动态可视化。
