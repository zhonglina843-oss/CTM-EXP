# 代码说明：ImageNet validation 带 label 分析

主脚本：`scripts/run_ctm_imagenet_labeled_analysis.py`。

这个脚本复用现有 CTM helper：

- `ctm_analysis/probe.py`：运行模型并记录 `S_action`、`S_out`、query、post-activation。
- `ctm_analysis/visualize.py`：沿用 CTM 风格生成 query/S 矩阵 GIF 和 PNG。
- `ctm_analysis/convergence.py`：复用原来的同步表示收敛计算；本脚本额外计算带 ground-truth label 的正确性 tick。

读取详细样本原始矩阵的示例：

```python
from pathlib import Path
import numpy as np

sample_dir = Path('samples/val-0000-label091-coucal')
tr = np.load(sample_dir / 'traces.npz', allow_pickle=True)
S_action = tr['sync_action']
S_out = tr['sync_out']
query = tr['queries']
post = tr['post_activations']
correct = tr['correct_by_tick']
S_full = np.load(sample_dir / 'S_full_active_all_ticks.npz')['matrices']
active_ids = np.load(sample_dir / 'active_neuron_indices.npy')
```

`S_full[t]` 是固定活跃神经元子集上的精确累计 Gram matrix：`post[:t, active_ids].T @ post[:t, active_ids]`。
本次设置只对 `detail_candidate=True` 的 25 个样本保存完整 raw 矩阵和 GIF，其余样本只保存统计行，避免文件过大。

本次没有改动模型 forward 或 checkpoint；新增代码只负责读取带 label 的子集、计算 correctness tick、组织输出。
