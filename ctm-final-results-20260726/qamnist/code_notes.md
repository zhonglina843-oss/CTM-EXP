# 代码说明：QAMNIST

## 主要分析代码

- 任务脚本：`scripts/run_ctm_qamnist_analysis.py`
- 原始矩阵导出 helper：`ctm_analysis/probe.py`
- GIF/图像可视化 helper：`ctm_analysis/visualize.py`
- 收敛指标计算：`ctm_analysis/convergence.py`

这些文件的快照已经复制到 `../code_snapshot/`。

## 如何读取原始矩阵

```python
from pathlib import Path
import numpy as np

sample_dir = Path("samples/sample-000")  # 按需要改成具体样本目录
tr = np.load(sample_dir / "traces.npz", allow_pickle=True)

S_action = tr["sync_action"]   # 每个 tick 的 S_action
S_out = tr["sync_out"]         # 官方模型实际使用/输出的 S_out 表示
query = tr["queries"]          # q_proj(S_action)
post = tr["post_activations"]  # 每个 tick 的神经元 post-activation

S_full = np.load(sample_dir / "S_full_active_all_ticks.npz")["matrices"]
active_ids = np.load(sample_dir / "active_neuron_indices.npy")
```

## S_full 的定义

`S_full[t]` 是基于固定活跃神经元子集重建的累计 post-activation Gram matrix：

```python
Z = post[:t, active_ids]
S_full_t = Z.T @ Z
```

完整全神经元 `D × D` 矩阵没有为每个 tick 全量保存；当前保存的是可追溯的精确活跃子矩阵。

## query / S_action 是否存在

QAMNIST 任务存在 `S_action` 和 `q_proj(S_action)`。query 是从记录的 `S_action` 通过训练好的 `q_proj` 重新计算得到的；没有改动模型权重或官方 forward 逻辑。

## 可视化文件

- `result.gif`：每个 tick 的答案/概率变化。
- `query.gif` / `query_overview.png`：query 随 tick 的变化。
- `synchronization_triptych.gif`：`S_action`、`S_out`、活跃神经元 `S_full` 的联合可视化。

## 本次最小代码改动

- 新增 `scripts/run_ctm_qamnist_analysis.py`，用 analysis-only forward tracing 记录每个 tick 的 `S_action`、`S_out` 和 query。
- 官方模型文件没有改动。
