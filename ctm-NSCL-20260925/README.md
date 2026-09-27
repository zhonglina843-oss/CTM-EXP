# Dynamic CTM NSCL Plan

## 1. Goal

验证一个已经训练好的 CTM 能否仅凭多 tick 的内部表示，判断到来的样本属于：

- 已知类：可以由某个已有类别的表示解释；
- 新类：与所有已有类别都存在稳定的表示差异。

本阶段只做 **new-versus-known detection**，不更新 CTM 参数，不增加 head，也不把新类自动写回模型。论文中的 NSCL 负责解释“已知类如何帮助发现未知类”，但没有提供后续增量更新规则，因此检测和更新必须拆成两个实验阶段。

## 2. Default checkpoint

默认使用：

```text
/root/autodl-tmp/ctm-dictionary-exp-20260816/self_fe_multik/stage2_checkpoint_freeze_unit/D_2048_k32_encode32_omp_recon/stage2_checkpoint.pt
```

该 checkpoint 对应 `ctm-freeze-dictionary` 路线。实验必须记录 checkpoint 的完整路径、模型配置、dictionary 配置和输入特征配置，避免把不同 stage 的表示混在一起。

## 3. Main idea

对每个样本保留 CTM 的内部轨迹，而不是只保存最终结果：

```text
e_1(x), e_2(x), ..., e_T(x)
```

每一个 `e_t(x)` 都是同一个样本在第 `t` 个 CTM tick 的 embedding。优先使用以下两类表示：

1. `post_state[t]`：每个 tick 的激活状态向量，作为低成本向量 embedding；
2. `S_full_active_all_ticks[t]`：由激活状态构造的神经元关系矩阵，作为结构 embedding。

第一版以 `post_state` 为主、`S_full` 为辅。原因是 `post_state` 不需要再次从 dictionary key 重建，且可以直接测试 CTM 的动态表征；`S_full` 用来验证差异是否也存在于神经元关系结构中。

这不是把论文的 NSCL loss 原样搬进来。论文的图节点是样本/增强样本，CTM 的 `S_full` 是神经元关系矩阵，二者不能直接等同。这里采用论文的核心思想：利用已有类形成 reference geometry，再观察未知样本是否落在已有类几何之外。

## 4. Data split

必须同时准备三组数据：

| Split | 用途 |
|---|---|
| known-train | 建立每个已知类的 reference bank，只用于确定 prototype 和距离尺度 |
| known-test | 检查方法不会把普通已知类误报为新类 |
| novel-test | 检查方法能否发现完全未参与训练的新类 |

第一轮建议使用 ImageNet 中已有实验的 5 个类别作为 known，另外选择 5 个未用于训练的类别作为 novel。不要只使用肉眼差异很大的类别；同时保留一组“近邻新类”，例如同一粗粒度物种下的不同细分类，用于测试方法是否真正利用 CTM 表示，而不是只识别明显的 domain shift。

所有阈值必须只用 `known-train` 和一个独立的 known calibration split 确定。`novel-test` 只能在最后用于评估。

## 5. Dynamic vector branch

### 5.1 Normalize each tick

对每个样本和 tick 的 `post_state` 做 L2 normalization：

```text
z_t(x) = post_state_t(x) / (||post_state_t(x)||_2 + eps)
```

每个已知类 `c` 在每个 tick 建立 prototype：

```text
mu_c,t = mean(z_t(x)) for x in known-train class c
mu_c,t = mu_c,t / (||mu_c,t||_2 + eps)
```

### 5.2 Per-tick compatibility

用 cosine distance：

```text
d_t(x, c) = 1 - cosine(z_t(x), mu_c,t)
```

每个 tick 的最近已知类距离和第二近距离为：

```text
m_t(x) = min_c d_t(x, c)
gap_t(x) = second_min_c d_t(x, c) - min_c d_t(x, c)
```

解释：已知样本应该在至少一个类别的 reference geometry 内；新样本应该对所有已知类都有较大的 `m_t`。`gap_t` 用来区分“明确属于某个旧类”和“所有旧类都不匹配”。

### 5.3 Aggregate over ticks

不要只看最后一个 tick，至少输出以下四个动态分数：

```text
final = m_T
mean = mean_t(m_t)
late = mean_{t >= T/2}(m_t)
area = mean_t(m_t)  # 等价于离散轨迹面积的归一化版本
```

另加一个首次越界时间：

```text
t_out = first t such that m_t > threshold_t for K consecutive ticks
```

`threshold_t` 从 known calibration 的 `m_t` 分布确定，不能使用 novel-test 标签。若新类在早期已经持续越界，而旧类会在后续 tick 回到某个 known prototype 附近，则 `t_out` 和轨迹曲线会比 final score 更有区分力。

第一版的最终 novelty score 只采用：

```text
novelty_vector(x) = mean_t(m_t(x))
```

其他分数作为 ablation，避免一开始训练一个黑盒 temporal classifier。

## 6. Dynamic graph branch

对 `S_full_active_all_ticks[t]` 使用已有的固定 neuron ID 对齐方式。每个 tick 保留：

- positive edge；
- Top 1% edge 或固定 top-k edge；
- 最大连通分量，保持与 `ctm-similarity-computation-20260919` 的预处理一致。

先只实现三个结构距离：

1. edge cosine distance；
2. weighted-Jaccard distance；
3. spectral distance，比较前 `r` 个归一化特征值。

对每个 known class 计算每个 tick 的 prototype graph 或 class medoid。然后得到：

```text
g_t(x) = min_c graph_distance(G_t(x), G_c,t)
```

动态图 novelty score 为：

```text
novelty_graph(x) = mean_t(g_t(x))
```

图分支的作用首先是验证“新类差异是否体现在神经元关系结构中”，不是立即寻找更新参数。若 graph branch 不能区分而 vector branch 可以，说明差异主要存在于状态值而不是连接结构；反之则说明应继续研究 neuron-pair/subgraph attribution。

## 7. NSCL-inspired diagnostic

为保持和论文的联系，额外计算一个不训练网络的 reference-space residual：

1. 在 known-train 的 `z_t` 上做 PCA/SVD，得到每个 tick 的 known subspace `U_t`；
2. 对到来样本计算投影残差：

```text
r_t(x) = ||z_t(x) - U_t U_t^T z_t(x)||_2^2
```

3. 对 `r_t(x)` 按时间求均值得到 residual novelty score。

这对应论文中“表示空间没有覆盖的 information”这一解释，但不要把它称为论文的 NSCL loss。它是一个 frozen-model diagnostic，用于回答：新类是否在 CTM 的 known representation span 之外。

## 8. Evaluation

必须同时报告：

- AUROC：known vs novel；
- AUPR-Novel；
- FPR95；
- 在 calibration threshold 下的 known false-reject rate；
- 在同一 threshold 下的 novel detection rate；
- 最近 known class 的 top-1 accuracy；
- 每个 tick 和每个 aggregate score 的结果。

推荐输出：

```text
per_sample_tick_scores.csv
class_prototypes.npz
known_calibration.json
novelty_metrics.csv
trajectory_summary.csv
```

至少画三类图：

1. known/novel 的 `m_t` 轨迹均值和置信区间；
2. 每个 aggregate score 的 ROC/PR；
3. 样本级的 tick-by-tick heatmap。

## 9. Minimal implementation order

### Phase A: frozen trace export

使用 checkpoint 做一次 inference，保存每个样本的：

```text
post_state[t]
sync_action[t]
sync_out[t]
S_full_active_all_ticks[t]
active_neuron_indices
prediction[t]
certainty[t]
label and split metadata
```

如果已有这些缓存，直接复用，不重复跑 CTM。

### Phase B: vector detector

只实现 `post_state` prototype、`m_t`、`mean_t(m_t)`、threshold calibration 和指标计算。先证明动态向量表示能否区分 known/novel。

### Phase C: graph detector

复用上一轮图相似度代码，只增加按 tick 的 class prototype 和 novelty aggregation。对照 vector branch，检查差异来自哪里。

### Phase D: ablations

比较：

```text
tick=1
final tick
mean over ticks
late ticks
vector branch
graph branch
vector + graph (只做固定权重，不训练融合器)
```

只有当一个简单的固定规则明显优于 `final tick` baseline，才考虑学习 temporal detector 或重新训练 NSCL-style encoder。

## 10. GPU decision

分两种情况：

- **已有每个样本的 tick trace**：不需要 GPU。prototype、距离、PCA/SVD、图相似度和指标都可以在 CPU 上完成。
- **需要从 `stage2_checkpoint.pt` 重新跑 CTM inference**：建议使用 GPU。因为需要加载 CTM、dictionary/feature pipeline，并对 ImageNet 图片逐批生成多 tick trace；CPU 理论上能跑，但会明显慢，且不适合作为服务器上的正式批量实验。

第一轮建议申请一张 GPU，只做 Phase A；Phase B-D 使用保存的 trace 在 CPU 上跑。这样 GPU 只消耗在必须的 CTM forward，不在后处理阶段浪费。

## 11. What is deliberately not implemented yet

- 不更新 CTM 的 synapse、trace processor 或 dictionary；
- 不自动增加新的分类 head；
- 不把 `novelty score` 当作新类标签写回模型；
- 不直接实现完整 NSCL training；
- 不使用 novel-test 标签调阈值；
- 不把 CTM 的神经元图和论文的样本图混称为同一个图。

这些内容必须等 detection 实验证明有效后再单独设计。下一阶段的更新候选应优先是新类 prototype/head 或小 adapter，并用旧类 replay 和蒸馏约束保持旧类性能。

## 12. Server run outline

服务器上的实际运行顺序应为：

```bash
# 1. prepare the CTM-EXP code and the CTM source tree
# 2. verify the checkpoint can be loaded with the matching model config
# 3. export frozen per-tick traces
# 4. copy only compact trace files to the analysis directory if needed
# 5. run vector detector on CPU
# 6. run graph detector on CPU
# 7. compare metrics and plots
```

运行前必须确认：

- checkpoint 对应的 CTM 源代码版本；
- input feature 的维度确实为 `D=2048`；
- dictionary 参数为 `k=32, encode32, omp_recon`；
- trace 的 tick 数和模型 `iterations` 一致；
- known/novel 类别没有数据泄漏。

## 13. Uploaded scripts

本目录包含三个脚本：

```text
export_dynamic_traces.py  # GPU/CPU: frozen CTM forward，导出每个 tick 的 post_state
dynamic_novelty.py        # CPU: prototype distance、动态 novelty score 和指标
run_dynamic_nscl.sh       # 服务器入口，串联上面两个脚本
```

`export_dynamic_traces.py` 使用缓存的 feature `.npz`，不重新计算 ResNet feature。输入 manifest 至少需要三列：

```csv
sample_id,label,split
img_0001,0,known_train
img_0002,0,known_test
img_0003,10,novel_test
```

建议先在服务器上设置：

```bash
export CTM_REPO=/root/autodl-tmp/continuous-thought-machines
export EXP_REPO=/root/autodl-tmp/CTM-EXP
export FEATURES=/path/to/features.npz
export MANIFEST=/path/to/manifest.csv
export DEVICE=cuda:0
bash "$EXP_REPO/ctm-NSCL-20260925/run_dynamic_nscl.sh"
```

如果已有 `traces.npz`，可以跳过导出阶段，直接运行：

```bash
python "$EXP_REPO/ctm-NSCL-20260925/dynamic_novelty.py" \
  --trace-dir /path/to/traces \
  --output-dir /path/to/analysis
```
