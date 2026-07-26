# 代码说明：ImageNet 网站 demo / paper checkpoint

## 主要分析代码

- 任务脚本：`scripts/run_ctm_demo_analysis.py`
- 原始矩阵导出 helper：`ctm_analysis/probe.py`
- GIF/图像可视化 helper：`ctm_analysis/visualize.py`
- 收敛指标计算：`ctm_analysis/convergence.py`

这些文件的快照已经复制到 `../code_snapshot/`，方便之后追溯本次结果是怎么生成的。

## 如何读取原始矩阵

```python
from pathlib import Path
import numpy as np

sample_dir = Path("samples/sample-000")  # 按需要改成具体样本目录
tr = np.load(sample_dir / "traces.npz", allow_pickle=True)

S_action = tr["sync_action"]   # 每个 tick 的 S_action；Sorting 中为空
S_out = tr["sync_out"]         # 官方模型实际使用/输出的 S_out 表示
query = tr["queries"]          # q_proj(S_action)；没有 q_proj 的任务中为空
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

这里保存的不是每个 tick 的完整全神经元 `D × D` 大矩阵，而是选中活跃神经元后的精确子矩阵。被选中的原始神经元编号保存在 `active_neuron_indices.npy`。

## query / S_action 是否存在

ImageNet 任务存在 `S_action` 和 `q_proj(S_action)`。query 是用已经记录下来的 `S_action` 通过训练好的 `q_proj` 重新计算得到的；没有改动模型权重，也没有改变 forward 逻辑。

## 可视化文件

- `result.gif`：每个 tick 的任务预测/输出变化。
- `query.gif` / `query_overview.png`：query 随 tick 的变化。
- `synchronization_triptych.gif` 或 `S_action_S_out_S_full.gif`：把 `S_action`、`S_out`、活跃神经元 `S_full` 放在一起观察。

## 本次最小代码改动

- 扩展了 `ctm_analysis/probe.py`、`ctm_analysis/visualize.py` 和 `scripts/run_ctm_demo_analysis.py`，用于额外保存 query 数组和 GIF。
- 官方模型文件和 checkpoint 没有改动。
