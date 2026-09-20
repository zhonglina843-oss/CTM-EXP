# CTM Connection-Graph Similarity Experiment

## 1. 方法总表

| 方法 | 核心函数/公式 | 比较对象 | 关注 | 是否需要固定 neuron ID | 优势 |
|---|---|---|---|---|---|
| Weighted-Jaccard | $J_w(a,b)=\dfrac{\sum_{e}\min(w_{a,e},w_{b,e})}{\sum_{e}\max(w_{a,e},w_{b,e})}$ | Top-k 边的连续权重 | 两图共享了多少强连接 | 是 | 直观，适合稀疏加权同步图，结果在 0 到 1 |
| Edge cosine | $\operatorname{cos}(a,b)=\dfrac{a^{\mathsf T}b}{\lVert a\rVert_2\lVert b\rVert_2}$ | 按 neuron-pair 对齐的边权向量 | 边的位置和同步强度整体方向 | 是 | 保留连续权重，计算便宜，适合作为主 baseline |
| Degree-strength cosine | $d_i=\sum_j w_{ij}$；再计算 $\operatorname{cos}(d_a,d_b)$ | 每个 neuron 的加权 degree 向量 | 是否依赖相似的关键 neuron/hub | 是 | 对具体边的小扰动较稳定，解释 neuron 层面差异容易 |
| Spectral similarity | $\lambda(A)=\operatorname{eigvalsh}(A)$；$\operatorname{sim}=\dfrac{1}{1+\lVert\lambda_a-\lambda_b\rVert_2/(\lVert\lambda_a\rVert_2+\lVert\lambda_b\rVert_2)}$ | 加权邻接矩阵特征值 | 整体拓扑、集中性、模块化和连通结构 | 否或弱依赖 | 能发现整体结构相似但具体边不完全相同的图 |
| Community similarity | 社区检测 $C(G)=\{C_1,\ldots,C_k\}$；比较 NMI、ARI 或 overlap | 社区划分及社区间连接 | 神经元模块/回路是否相似 | 通常需要匹配 | 更接近回路级解释 |
| DeltaCon | $S=(I+\varepsilon D-\varepsilon A)^{-1}$；比较 affinity 矩阵距离 | 多跳 affinity/influence 矩阵 | 整体结构差异，不只看直接边 | 是 | 适合固定 neuron 的动态图比较 |
| DeltaCon-ATTR | DeltaCon distance + $\text{impact}(v)=\lVert S_{v,:}^{(a)}-S_{v,:}^{(b)}\rVert_2^2$；再做 node/edge attribution | DeltaCon affinity 变化 | 差异最大的 neuron 和 edge | 是 | 同时给出整体相似度和差异归因 |
| Graph Edit Distance | $\operatorname{GED}(G_a,G_b)=\min_{\mathcal E}\operatorname{cost}(\mathcal E)$ | 增删节点/边及修改权重的编辑代价 | 把图 A 变成图 B 需要哪些操作 | 可匹配或固定 | 局部差异解释直观，但大图成本高 |
| Graph Kernel/WL | $h_v^{(t+1)}=\operatorname{HASH}\!\left(h_v^{(t)},\!\operatorname{SORT}\{h_u^{(t)}:u\in N(v)\}\right)$；比较 kernel similarity | 节点邻域标签和局部 motif | 是否存在相似局部拓扑 | 不一定 | 对 neuron 重排较鲁棒，但归因不直接 |
| Gromov-Wasserstein | $\operatorname{GW}(G_a,G_b)=\min_{\pi}\sum_{i,j,k,l}\lvert d_a(i,j)-d_b(k,l)\rvert^2\pi_{ik}\pi_{jl}$ | 两图内部距离关系及节点耦合 | neuron 角色跨图匹配 | 不需要 | 适合 neuron identity 不可靠或跨模型比较 |
| Dynamic trajectory | $\operatorname{TrajSim}=\dfrac{1}{T}\sum_{t=1}^{T}\operatorname{Sim}(G_{a,t},G_{b,t})$；或比较 $\Delta G_t=G_t-G_{t-1}$ | 多个 tick 的图序列或变化量 | CTM 思考过程是否相似 | 通常需要 | 区分静态相似与动态形成过程 |

## 2. 首批实验方法

先做四种低成本、容易解释的方法：

1. `weighted_jaccard`
2. `edge_cosine`
3. `degree_strength_cosine`
4. `spectral_similarity`

矩阵使用已有的 `S_full_active_all_ticks.npz`，神经元通过
`active_neuron_indices.npy` 映射回原始 NLM ID。默认取每个 tick 的 Top 1%
positive edges，并使用最大连通分量，和现有 5x5 图的绘图规则一致。

## 3. 实验流程

1. 从 `labels.csv` 读取类别和 `class_sample_index`，保持图中行列顺序。
2. 对每个 tick、每个 sample 加载保存的同步矩阵。
3. 保留 Top 1% positive edges，并提取最大连通分量。
4. 用固定 neuron ID 对齐两张图。
5. 枚举所有 sample pair：同类 pair 进入 same-class，异类 pair 进入 different-class。
6. 对每个方法记录 mean、median、std、pair 数量和 `same_minus_different`。
7. 对每个 sample 记录它与同类其他样本、异类样本的平均相似度。
8. 生成每个方法每个 tick 的网格 CSV，单元格为该 sample 与同类另外四个样本的平均相似度。
9. 记录矩阵处理时间、方法计算时间、平均节点数、平均边数和最大 RSS。

## 4. 输出文件

- `pairwise_similarity.csv`: 每个 tick 的逐 pair 相似度。
- `tick_summary.csv`: 每个 tick 的同类/异类总体统计。
- `sample_similarity.csv`: 每个 sample 的同类/异类统计。
- `runtime_metrics.csv`: 时间、图规模、内存等成本指标。
- `grid_<method>_tick-XXX.csv`: 与 5x5 图对应的 sample 网格结果。
- `run_manifest.json`: 参数和方法记录。

## 5. 运行方式

```bash
python3 similarity_experiment.py \
  --result-dir /path/to/result-dir \
  --subset-dir /path/to/imagenet-class-grid-5x5-subset-a \
  --output-dir /path/to/ctm-similarity-computation-20260919/results \
  --ticks 1,5,10,15,20,30,40,50 \
  --density 0.01 \
  --component lcc
```

不需要 GPU。脚本只读取已保存的矩阵，首批实验是 CPU 和内存计算。只有在服务器上还没有目标 5x5 样本的 CTM 矩阵时，才需要另外运行 CTM 推理；那一步才可能需要 GPU。

## 6. 后续扩展

后续 DeltaCon、community、dynamic trajectory 等方法应继续写入相同的四类结果：逐 pair 相似度、tick summary、sample summary、runtime metrics。这样可以比较区分能力、稳定性、解释能力和计算成本，而不是只比较某一个相似度数值。
