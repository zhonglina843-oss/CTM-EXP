# UCL x CTM 实验状态

更新时间：2026-10-19（Asia/Hong_Kong）

## 当前阶段

设计与多 Agent 协作规范已建立；CTM trace、AE 原生路线和 AE+CTM 记录管线已完成工程 smoke。正式多 seed/校准 probe 实验尚未开始。

## Git 状态

| 项目 | 状态 |
| --- | --- |
| 集成分支 | `ctm-ucl-experiment-20261019` |
| 设计基线分支 | `ctm-ucl-probe-plan` |
| GitHub remote | `origin` -> `https://github.com/zhonglina843-oss/CTM-EXP.git` |
| 最新本地/GitHub 提交 | 见 `git log --oneline -1`；提交后同步状态以 `git status --short --branch` 为准 |
| 服务器 checkout | `/root/CTM-EXP`，同名分支；本轮本地改动待推送后同步 |

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

## 当前未完成

- [ ] A0 冻结 phase-1 manifest/schema
- [ ] 选择/训练 phase-1 CTM probe 校准方式，并做多 seed AE+CTM 验证
- [ ] CURL 锁定上游 commit、建立兼容环境、复现官方 MNIST 入口
- [ ] InfoUCL 锁定上游 commit、核实 feature tap/InfoDrop 开关并复现基线
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
