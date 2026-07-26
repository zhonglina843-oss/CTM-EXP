# CTM 实验结果总入口

建议阅读顺序：

1. 先看 `overall_summary.md`，了解所有任务的总体收敛 tick、矩阵收敛 tick 和是否存在 query。
2. 再进入每个任务文件夹，阅读对应的 `summary.md`。
3. 如果需要复现读取矩阵或可视化流程，再看该任务的 `code_notes.md`。

每个任务目录通常包含：

- `convergence_summary.csv`：逐样本统计表，适合后续画图或重新分析。
- `aggregate_convergence.png`：该任务整体收敛情况可视化。
- `samples/`：详细样本目录，包含原始矩阵、query、GIF/PNG/PDF 可视化。
- `manifest.json`：模型、checkpoint、输出定义等元信息。

说明：为了让矩阵结果可读、可传输，`S_full` 保存的是“活跃神经元子矩阵”的精确结果，而不是每个 tick 的完整全神经元 `D × D` 大矩阵。每个样本目录中的 `active_neuron_indices.npy` 记录了被选中的原始神经元编号。
