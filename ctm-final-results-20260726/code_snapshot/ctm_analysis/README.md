# CTM 同步表示与矩阵分析说明

这个 helper 包用于记录官方模型中的两类同步表示，并从神经元 post-activation 轨迹中重建诊断用的 `S_full` 矩阵。

## 记录的量

- `sync_action[t]`：第 `t` 个 tick 用来生成 attention query 的同步表示。
- `sync_out[t]`：第 `t` 个 tick 用来生成输出 logits 的同步表示。
- `S_full[t] = Z[:t]^T Z[:t]`：从 post-activation 重建的累计神经元活动 Gram matrix。

官方实现只为采样进 `S_action` 和 `S_out` 的神经元对学习单独的指数衰减参数，并没有为所有 `D × D` 神经元对都定义一套 learned decay 参数。因此这里的 `S_full` 明确标注为诊断用的 unweighted full-neuron matrix，而不是模型 forward 中直接消费的官方表示。

对于 `random-pairing`，`S_action` 和 `S_out` 是采样神经元对形成的向量；对于 `random` 和 `first-last`，它们是上三角矩阵展开后的表示，并会额外导出成 triangular heatmap。

官方 forward loop 使用一个学习得到的初始 activated state：先在 tick update 前计算 action synchronization，再在 update 后计算 output synchronization。这里保留 `model(..., track=True)` 记录到的官方 action/output trace；`S_full` 则按论文图示习惯，用截至 tick `t` 的 post-activation 轨迹重建。

## 收敛 tick 的定义

所有 tick 都从 1 开始计数。报告的 tick 表示：从这个 tick 开始，该条件一直满足直到最终推理 tick。

- label：top-1 类别等于最终 top-1 类别。
- probability：与最终概率分布的 Jensen-Shannon divergence 小于等于 `1e-3`。
- certainty：与最终 certainty 的绝对差小于等于 `0.01`。
- sampled synchronization：cosine 大于等于 `0.99`，或 relative L2 小于等于 `0.05`。
- full synchronization：matrix cosine 大于等于 `0.99`，或 relative Frobenius 小于等于 `0.05`。
- local plateau：相邻 tick 的 relative L2/Frobenius change 小于等于 `0.05`，且之后一直低于该阈值。

`S_full` 的收敛可以从较小的 tick-by-tick Gram matrix 精确计算，因此不需要真的把所有 `4096 × 4096` 大矩阵都落盘。保存的 `256 × 256` full-matrix view 是精确 block mean，不是普通图片 resize。

为了得到更清楚的高分辨率图，神经元会按完整轨迹中的 RMS post-activation 排序。每个 tick 保存精确的 Top-512 子矩阵；GIF 使用固定 Top-256 子集。原始神经元 ID、活跃度统计、聚类顺序，以及 `S_action`/`S_out` 对应的神经元对都会导出，所以这些筛选视图仍然可以追溯回模型内部神经元。

## 常用命令

下载并裁剪网站 demo 视频第一帧作为输入：

```bash
python -m scripts.download_website_demos \
  --output-dir data/website_imagenet_demos
```

运行当前官方 Hugging Face checkpoint：

```bash
PYTHONPATH=. python scripts/run_ctm_demo_analysis.py \
  --model-id SakanaAI/ctm-imagenet \
  --model-cache-dir /root/autodl-tmp/hf-cache \
  --input-dir data/website_imagenet_demos/inputs \
  --output-dir /root/autodl-tmp/ctm-results/hf-pilot \
  --inference-iterations 50 \
  --matrix-ticks 1,5,10,15,20,30,40,50 \
  --save-full-final
```

运行原论文 checkpoint：

```bash
PYTHONPATH=. python scripts/run_ctm_demo_analysis.py \
  --checkpoint checkpoints/imagenet/ctm_paper.pt \
  --input-dir data/website_imagenet_demos/inputs \
  --output-dir /root/autodl-tmp/ctm-results/paper-pilot \
  --inference-iterations 50 \
  --matrix-ticks 1,5,10,15,20,30,40,50 \
  --save-full-final
```

补充说明：网站视频是 MP4 压缩后的可视化结果。第一帧通常没有路线箭头，但压缩仍可能让像素和原始 ImageNet validation 图片略有差异；这个 caveat 已经记录在 run manifest 中。
