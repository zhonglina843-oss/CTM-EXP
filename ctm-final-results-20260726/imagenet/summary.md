# ImageNet 网站 demo / paper checkpoint

- 统计样本数：`20`
- 详细样本文件夹数：`20`，例如 `ctm-input-11037`, `ctm-input-12495`, `ctm-input-13575`, `ctm-input-14308`, `ctm-input-17352`, `ctm-input-17558`, `ctm-input-22403`, `ctm-input-23597` 等。
- 打包前使用的模型/checkpoint：`checkpoints/imagenet/paper_zip_extracted/imagenet/ctm_imagenet_D=4096_T=50_M=25.pt`
- 注意：网站 demo 输入在本地运行时没有 ground-truth label，因此这里统计的是最终类别稳定/收敛情况，不是外部标签验证过的 ImageNet top-1 correctness。

## tick 收敛 / 正确性

- `label_stable_tick`：min 3, median 6, mean 21.250, max 49
- `probability_js_tick`：min 5, median 49, mean 39.900, max 50
- `certainty_plateau_tick`：min 4, median 47.500, mean 37, max 50

解释：

- `label_stable_tick` 表示从该 tick 开始，top-1 类别一直保持为最终类别。
- `probability_js_tick` 表示预测概率分布已经接近最终分布。
- `certainty_plateau_tick` 表示模型置信度进入接近最终值的平台期。

## 同步矩阵/同步表示收敛

- `action_cosine_tick`：min 49, median 50, mean 49.950, max 50
- `out_cosine_tick`：min 49, median 49, mean 49.150, max 50
- `full_sync_cosine_tick`：min 48, median 50, mean 49.700, max 50

这说明 ImageNet 的类别 top-1 往往很早稳定，但同步表示本身通常要到接近最后几个 tick 才与最终状态高度接近。

## 第一个详细样本中的矩阵/query 形状

- `sync_action`：`(50, 2048)`
- `sync_out`：`(50, 8196)`
- `queries`：`(50, 1024)`
- `post_activations`：`(50, 4096)`
- `predictions`：`(50, 1000)`
- `certainties`：`(50,)`
- `S_full_active`：`(50, 512, 512)`

## 关键文件

- `convergence_summary.csv`：逐样本收敛/稳定性统计表。
- `manifest.json`：checkpoint、模型元信息和矩阵定义。
- `aggregate_convergence.png`：任务级收敛可视化。
- `samples/*/traces.npz`：原始预测、`S_action`、`S_out`、post-activation、query 等。
- `samples/*/S_full_active_all_ticks.npz`：可视化前的精确活跃神经元 `S_full` 矩阵。
- `samples/*/*.gif`：逐 tick 可视化结果。
