# UCL x CTM 实验状态

更新时间：2026-10-19（Asia/Hong_Kong）

## 当前阶段

设计与多 Agent 协作规范已建立；尚未开始模型实现或正式训练。

## Git 状态

| 项目 | 状态 |
| --- | --- |
| 集成分支 | `ctm-ucl-experiment-20261019` |
| 设计基线分支 | `ctm-ucl-probe-plan` |
| GitHub remote | `origin` -> `https://github.com/zhonglina843-oss/CTM-EXP.git` |
| 最新本地/GitHub 提交 | 见 `git log --oneline -1`；不得仅凭本表推断未推送状态 |
| 服务器 checkout | `/root/CTM-EXP`，同名分支 |

## 已完成

- [x] 中文实验设计
- [x] English experimental design
- [x] 详细 CTM/UCL probe implementation plan
- [x] 多 Agent 分工、handoff、图片和验收规范
- [x] 生成数据与本地凭据保护规则（`.gitignore`）
- [x] GitHub 专用实验分支
- [x] 服务器端同名 checkout

## 当前未完成

- [ ] A0 冻结 phase-1 manifest/schema
- [ ] A1 核验 CTM trace 与真实 synchronization shape
- [ ] A5 建立 records reader 和指标骨架
- [ ] A2 完成 AE 单 seed 快速验证
- [ ] G1/G2 通过后再启动 CURL 与 InfoUCL 路线
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
