# CTM 所有任务实验结果总览

## 任务级统计

下表把每个任务的总体结果放在一起。tick 均为从 1 开始计数；`median_*` 表示该任务样本的中位数。

| 任务 | 统计样本数 | 详细样本数 | 主要结果 | 正确/稳定 tick 中位数 | S_action 收敛 tick 中位数 | S_out 收敛 tick 中位数 | S_full 收敛 tick 中位数 | 是否有 query |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| imagenet | 20 | 20 | 网站样本无 ground-truth label，因此统计最终类别稳定性 | label_stable_tick median=6 | 50.0 | 49 | 50 | True |
| parity | 64 | 3 | final_accuracy median=1 | full_correct_sustained_tick median=67 | 61.0 | 75 | 73 | True |
| maze | 32 | 3 | final_navigation_solved median=0; final_step_accuracy median=1 | navigation_solved_tick median=0 | 71.0 | 72 | 74 | True |
| qamnist | 64 | 2 | final_correct median=1; most_certain_correct median=1 | answer_correct_sustained_tick median=122 | 126.0 | 130 | 129 | True |
| sorting | 64 | 2 | final_exact median=1; final_token_accuracy median=1 | first_exact_sustained_tick median=47 | 空 | 50 | 50 | False |

## 目录结构

每个任务文件夹包含：

- `summary.md`：该任务的人类可读总结。
- `code_notes.md`：如何读取原始矩阵、query，以及如何对应到可视化脚本。
- `convergence_summary.csv`：逐样本收敛/正确性统计。
- `manifest.json`：checkpoint、模型配置、矩阵定义等元信息。
- `aggregate_convergence.png`：任务级收敛图。
- `samples/`：详细样本的原始结果和 GIF/PNG/PDF。

## 重要说明：S_full 的含义

官方 CTM 代码直接记录/使用的是 `S_action` 和 `S_out` 两类同步表示，并没有为所有神经元对都维护一个完整的、带学习衰减参数的 `Sync matrix`。

这里导出的 `S_full` 是诊断用矩阵：从 post-activation 轨迹重建的累计 Gram matrix。为了让结果文件大小可控、图像可解释，每个 tick 保存的是精确的活跃神经元子矩阵，而不是完整全神经元 `D × D` 矩阵。

每个详细样本目录里都有：

- `S_full_active_all_ticks.npz`：每个 tick 的活跃神经元 `S_full` 子矩阵。
- `active_neuron_indices.npy`：这些活跃神经元在原模型中的原始编号。
- `active_neuron_statistics.npz`：用于选择/排序活跃神经元的统计量。

## 代码快照

`code_snapshot/` 保存了本轮分析用到的脚本和共享 helper 代码。它的目的不是替代原始 CTM 仓库，而是让这次结果对应的读取、导出、可视化逻辑可追溯。
