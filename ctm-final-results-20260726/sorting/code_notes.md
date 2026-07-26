# 代码说明：Sorting reproduction final checkpoint

## 主要分析代码

- 任务脚本：`scripts/run_ctm_sort_analysis.py`
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

S_action = tr["sync_action"]   # Sorting 中形状为 ticks x 0，是空占位
S_out = tr["sync_out"]         # 官方模型实际使用/输出的 S_out 表示
query = tr["queries"]          # Sorting 中形状为 ticks x 0，是空占位
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

Sorting 任务没有 `S_action/query`。原因是 `ContinuousThoughtMachineSORT` 的结构中：

- `n_synch_action=0`
- `q_proj=None`
- `attention=None`

因此保存下来的 `sync_action` 和 `queries` 是空数组占位，而不是导出失败。

## 可视化文件

- `result.gif`：每个 tick 的排序输出变化。
- `synchronization_triptych.gif`：`S_out` 和活跃神经元 `S_full` 的可视化；`S_action/query` 为空。

## 本次最小代码改动

- 新增 `scripts/run_ctm_sort_analysis.py`，用于分析和导出。
- 修补了 `tasks/sort/train.py` 中一处打印 bug：原代码引用不存在的 `args.model`，改成打印 `Running SORT CTM`。
