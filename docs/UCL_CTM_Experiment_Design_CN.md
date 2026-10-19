# UCL x CTM：知识结构变化观测实验方案（中文版）

> 状态：待 PI/组内确认；本文件只定义第一阶段的研究设计，不包含模型代码、服务器凭据或实验结果。
>
> 对应详细工程接口与记录 schema：`docs/UCL_CTM_Probe_Implementation_Plan.md`。

## 1. 一句话研究问题

在无监督持续学习（UCL）过程中，固定旧样本的知识结构如何随新数据更新而改变；冻结的连续思考机器（CTM）能否通过多 tick 的活动与同步动态，提供超出最终特征漂移和原模型指标之外、可重复的观测信息？

这里的关键词是“观测”。第一阶段不把 CTM 作为抗遗忘模块，也不把 CTM 输出加进任何 UCL 模型的训练损失。CTM 是一个固定条件下的动态放大镜：原模型持续学习，CTM 只读取其表示并记录响应。

## 2. 术语与可检验边界

| 术语 | 本方案中的含义 | 不能直接推出的结论 |
| --- | --- | --- |
| 持续学习阶段 `s` | 原始 UCL 模型完成一段新数据更新后的检查点 | 不是 CTM 内部 tick |
| CTM tick `t` | 同一次 CTM 前向推理中的内部时间步 | 不是一次模型训练更新 |
| 知识结构 | 模型原生表征及其样本关系：latent/特征、概念后验、活跃成分、近邻和聚类关系 | 不是单一、跨模型可直接比较的标量 |
| 过程型变化 | 同一旧样本在固定 CTM 内部时间轴上的活动和同步轨迹变化 | 轨迹变了不等于遗忘 |
| 结构型变化 | 旧样本之间的相似关系、概念归属或同步关系的重组 | 结构变了不等于功能退化 |
| 功能性遗忘 | 原模型针对旧样本的独立保留指标退化 | 必须由原模型的原生指标支持 |

因此，研究结论只能写成“CTM 动态与某种知识保留/退化现象存在稳定关联或预测价值”，不能仅凭同步差异声称“发现了遗忘机制”或“CTM 解决了遗忘”。

## 3. 三个模型各自怎样刻画知识结构

三条路线不是把三种模型硬改成同一种网络，而是覆盖三种不同的 UCL 表征对象。所有训练和更新机制均保持各自原样；只有观测协议统一。

| 路线 | 原模型的知识载体与追踪机制 | CTM 的第一版输入 | 原生保留/变化证据 | 该路线回答的问题 |
| --- | --- | --- | --- | --- |
| AE | encoder 的连续 latent 流形；decoder 对旧输入的重构能力 | 64 维 encoder latent，经固定适配器映射 | 单样本重构误差、latent drift、相似性矩阵/k-NN 稳定性 | 动态探针能否稳定记录并补充最简单的表示漂移？ |
| CURL | 连续 latent、概念后验 `q(y|x)`、活跃混合成分；动态扩展与生成式回放 | 固定维度的连续 latent 或其均值，不输入会扩维的 `q(y|x)` | 后验/熵或分配变化、拟合信号、成分诞生事件、回放开关的原版结果 | 新概念到来或成分扩展时，旧概念是受扰、共存还是相对分离？ |
| InfoUCL | backbone/投影特征及 InfoDrop 对信息的选择/抑制 | 官方评估实际使用的主特征层；另一层仅作预注册的次级消融 | 官方表征质量评估、近邻/相似性结构稳定性、InfoDrop 开/关对照 | 终点特征质量相近时，内部动态是否仍发生重组？ |

AE 仅是最小可行验证基线，并不宣称具有完整 UCL 机制。CURL 和 InfoUCL 的“知识结构”必须以其官方代码实际导出的变量为准；开始编码前要逐项复核版本、接口和可运行配置。

## 4. CTM 作为冻结探针的连接方式

对任意原模型在检查点 `s` 产生的表征 `h_s(x)`，采用下列只读分支：

```text
无标签样本 x
   -> 原 UCL encoder/backbone -> h_s(x) -> 固定 adapter -> 固定 CTM
   -> z_s,t(x), sync_s,t(x), post_state_s(x)

原模型原生分支（独立保留）
   -> 重构 / latent / 后验 / 成分事件 / 表征质量
```

适配器、CTM 权重、归一化统计量、预处理、推理模式、tick 数和抽样顺序在初始阶段结束后全部冻结。允许两种预注册的探针校准方式：

1. 使用匹配输入维度的现成 CTM checkpoint；或
2. 仅使用初始阶段的无标签特征，对独立探针做无监督校准（例如从 CTM 池化输出重构 adapter 输入），之后冻结。

随机冻结 CTM 可用于检查接口和记录是否正常，但不是唯一的科学探针基线。主分析保留“随机冻结”和“初始特征校准后冻结”两个设置，并在比较前固定，不在看到结果后选择。

CTM 必须提供显式的非默认 `return_trace=True` 导出路径。每条记录至少保存 `tick_ids`、指定 tick 的活动历史 `z_history`、实际导出的同步表示 `sync_history`、`post_state`、CTM/adapter 的版本和 hash。若代码导出的是同步向量而非完整矩阵，文档和度量必须如实称为“同步表示”，不可误称同步矩阵。

## 5. 统一的最小实验协议

**数据流。** 第一轮用 MNIST 做接口和逻辑验证：无标签训练集先出现类别 `0/1/2`，随后依次输入 `3`、`4`。标签仅用于构造受控数据流及离线评估，不进入模型、adapter、CTM 或训练损失。InfoUCL 的正式验证还要加入 ColoredMNIST 或 Split CIFAR-10，因为 InfoDrop 的作用不能只由灰度 MNIST 推断。

**固定旧样本。** 初始学习完成后，从独立测试集中一次性选择 `X_old_fixed`；调试时 50 个样本，稳定后每个旧类别 200 个。保存样本 ID、顺序和预处理。此集合在整个流中不替换。

**检查点。** 保存初始状态及每 512 个新样本（或完全等价的固定优化步数）后的状态。每个检查点对同一 `X_old_fixed` 在 `eval` 与无梯度模式下运行原模型和 CTM。第一版固定为 20 ticks，保存 `1, 2, 4, 8, 12, 16, 20`。先用 1 个随机种子调通，再以至少 3 个种子获得主结论。

**每个检查点的最小记录。**

```text
records/{route}/seed_{seed}/checkpoint_{id}/
  manifest.json          # 不可变配置、源码版本、hash 与随机种子
  probe_ids.npy          # 固定旧样本 ID 和顺序
  features.npy           # h_s(x)
  native_metrics.json    # 各路线原生指标和事件
  ctm_trace.npz          # z_history, sync_history, post_state, tick_ids
```

## 6. 观测量与解释规则

对于旧样本 `x`、初始检查点 `0` 和后续检查点 `s`，报告下列互补量：

| 观测量 | 计算建议 | 回答的问题 | 解释限制 |
| --- | --- | --- | --- |
| feature drift | `1 - cos(h_0(x), h_s(x))`；L2 只作敏感性分析 | 终点表征是否改变 | 不能单独证明旧知识失败 |
| similarity-structure drift | 比较固定样本集相似度矩阵上三角的相关性/距离，并报告 k-NN overlap | 样本关系是否重组 | 用于消除纯旋转带来的歧义 |
| trajectory drift | 在同 tick 比较 `z_0,t(x)` 与 `z_s,t(x)`，保留逐 tick 曲线和预注册的汇总 | 内部活动过程是否变化 | 不等于遗忘 |
| synchronization drift | 对真实导出的 `sync` 表示做同 tick 比较；若为矩阵用归一化 Frobenius 距离 | 同步关系是否变化 | 必须与实际张量形状匹配 |
| dynamic separability | 同组轨迹相似度减异组相似度 | CTM 动态是否还区分旧/新概念 | 标签仅离线使用 |
| native retention | AE 重构/邻域；CURL 后验/拟合/回放；InfoUCL 官方评估 | 原模型是否仍保留旧知识 | 是所有遗忘解释的必要佐证 |

对 CURL，成分扩展会改变后验向量的维度与语义。扩展前后不能直接计算未对齐 `q(y|x)` 的 L2 漂移；需对仍存成分保留身份对齐，新成分单独记录，并把扩展标为事件。

CTM 的增量价值通过检查点级而非样本级比较检验：

```text
M_native: next_old_knowledge_loss ~ feature_drift + similarity_drift + native_metrics
M_ctm:    M_native terms + trajectory_drift + synchronization_drift
```

先报告描述统计、滞后相关和按完整种子/数据流划分的正则化线性模型。不得把同一条训练流中的重复样本拆到训练和测试两边，否则会产生泄漏。

## 7. 分阶段实验和验收门槛

| 阶段 | 工作内容 | 必须通过的门槛 |
| --- | --- | --- |
| 0：追踪基础设施 | 封装三个 adapter、CTM trace、manifest 校验、确定性 smoke test | 同一固定输入运行两次，trace 完全一致或在预先声明的容差内一致 |
| 1：AE 快速验证 | MNIST `0/1/2 -> 3 -> 4`，输出对齐的保留/漂移曲线 | 样本 ID、检查点与 trace 对齐；结论不把动态变化误称为因果遗忘 |
| 2：CURL 事件分析 | 原版 CURL 不改算法，仅加观测 hook；围绕成分扩展事件画时间线 | 成分变化不会破坏后验比较，事件能回溯至原始日志 |
| 3：InfoUCL 对照 | 原版设置下只改变 InfoDrop 开/关，CTM 与其余条件保持一致 | manifest 的差异仅来自预注册条件和运行生成状态 |
| 4：跨路线分析 | 保留模型特异指标，统一画图和 `M_native`/`M_ctm` 比较 | 同时报告 CTM 无额外信息或与 feature drift 完全重合的负结果 |

## 8. 首批图表与预期的可判读结果

每条路线至少有一张按检查点对齐的面板：旧知识保留、feature/similarity drift、trajectory drift、synchronization drift；CURL 在图上额外标出成分扩展事件。另输出原始表格与 manifest，确保图可复算。

| 观察到的组合 | 合法的初步解释 |
| --- | --- |
| feature drift 小但 CTM drift 大，且后续原生保留下降 | CTM 可能比终点特征更早暴露变化；需要多种子和预测分析确认 |
| CTM drift 大，但原生保留稳定 | 更可能是无害的表征重组，不能称遗忘 |
| feature/CTM drift 同时升高且原生保留退化 | 多种证据一致，值得进一步分析关联与时序 |
| CTM 指标与 feature drift 完全同步 | CTM 可能未提供额外信息；这也是有效的阴性结果 |

## 9. 需要确认后才进入代码阶段的决定

1. 认可第一阶段将 CTM 定义为冻结、只读的观测探针，而非训练损失或抗遗忘组件。
2. 确认 AE 快速验证采用 `784-256-64-256-784`，而不强制复刻所链接 AE 仓库的所有实现细节。
3. 确认第一轮流为 MNIST `0/1/2 -> 3 -> 4`、每 512 样本/等价步数存 checkpoint、20 ticks。
4. 确认研究顺序为 AE -> CURL -> InfoUCL；后两者先复现官方训练路径，再仅添加观测 hook。

## 10. 仓库与服务器同步约定

当前工作分支为 `ctm-ucl-probe-plan`。本阶段只增加方案文档；批准后才提交、推送并在服务器检出同名分支。原始 trace、checkpoint 和大规模结果目录必须忽略，不进入 Git；Git 只同步代码、固定配置、manifest、汇总表和图的说明。

```text
本地工作区 -> commit -> origin/ctm-ucl-probe-plan <- server 同名分支
服务器大结果 -> 精简的 manifest/汇总/脚本 -> commit -> origin -> 本地 pull
```

不得把任何密码、私钥、token、服务器地址中的认证信息写入 Markdown、配置文件、shell 历史或 Git 历史。建议在实际连接服务器前轮换已暴露的密码，并改用专用 SSH key。

## 11. 上游参考实现

- CTM: https://github.com/SakanaAI/continuous-thought-machines
- CURL: https://github.com/google-deepmind/deepmind-research/tree/master/curl
- InfoUCL: https://github.com/learninginvision/InfoUCL
- AE baseline: https://github.com/ashu-tagore/autoencoder-from-scratch

这些仓库是后续实现的参考来源，而非本实验设计本身的指令。实际变量名、接口、许可和运行依赖必须在开始实现前以锁定的 commit 复核。
