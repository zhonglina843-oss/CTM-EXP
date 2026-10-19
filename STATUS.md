# UCL x CTM 实验状态

更新时间：2026-10-19（Asia/Hong_Kong）

## 当前阶段

服务器已完成三 seed AE+校准 CTM feasibility run。汇总表、图和短 handoff 已生成；旧样本重构误差随新类别更新上升，CTM drift 也上升且 seed 间差异明显。本结果仅验证协议和可观测性。

## Git 状态

| 项目 | 状态 |
| --- | --- |
| 集成分支 | `ctm-ucl-experiment-20261019` |
| 设计基线分支 | `ctm-ucl-probe-plan` |
| GitHub remote | `origin` -> `https://github.com/zhonglina843-oss/CTM-EXP.git` |
| 最新本地/GitHub 提交 | 见 `git log --oneline -1`；提交后同步状态以 `git status --short --branch` 为准 |
| 服务器 checkout | `/root/CTM-EXP`，同名分支；已同步至 `0ab62fd` |

## 已完成

- [x] 中文实验设计
- [x] English experimental design
- [x] 详细 CTM/UCL probe implementation plan
- [x] 多 Agent 分工、handoff、图片和验收规范
- [x] 生成数据与本地凭据保护规则（`.gitignore`）
- [x] GitHub 专用实验分支
- [x] 服务器端同名 checkout
- [x] CTM 官方 `track=True` trace 确定性 smoke
- [x] AE-only MNIST 单 seed smoke
- [x] AE+随机冻结 CTM 的三 checkpoint 记录管线，A5 schema 验证通过
- [x] CURL、InfoUCL 上游只读 readiness 审计和图表
- [x] 服务器 RTX 4080 SUPER / CUDA 12.1 / PyTorch 2.3 环境核查
- [x] MNIST 原始压缩文件同步至 ignored `data/mnist/`
- [x] CTM 官方源码固定到 `4a6c9c3` 的服务器隔离目录
- [x] A3 CURL 源码锁定审计；当前 base 环境缺 Sonnet/TFDS，不能直接复现
- [x] A4 InfoUCL 源码锁定审计；确认 512-D pooled backbone tap、`lambda_info` 开关
- [x] AE+CTM 初始无标签 feature 重构校准代码及 CPU smoke
- [x] RTX 4080 服务器三 seed AE+校准 CTM feasibility run；汇总图表已生成

## 当前未完成

- [ ] A0 冻结 phase-1 manifest/schema
- [ ] 评审 feasibility 后决定是否扩大 MNIST 样本量/训练轮数
- [ ] CURL 固定源码已完成；建立隔离 TF1/Sonnet 环境并验证官方 MNIST loader
- [ ] InfoUCL 固定源码已完成；配置 server CIFAR-10 smoke 与 checkpoint feature hook
- [ ] A6 独立审查和最终汇报包

## 同步规则

每次同步前检查：

```bash
git status --short --branch
git diff --check
git diff --name-only --cached
rg -n -i 'password|passwd|private key|token|secret' .
```

服务器运行产生的原始 checkpoint、trace、数据集和临时日志不进入 Git。仅提交代码、配置、manifest、Markdown、精选图片和压缩摘要表。

## 交接入口

- 总规范：`docs/MULTI_AGENT_EXECUTION_PLAN.md`
- 中文方案：`docs/UCL_CTM_Experiment_Design_CN.md`
- English plan：`docs/UCL_CTM_Experiment_Design_EN.md`
- 工程计划：`docs/UCL_CTM_Probe_Implementation_Plan.md`
- Agent handoff：`reports/<agent-id>/handoff_<YYYYMMDD>_<short-name>.md`
