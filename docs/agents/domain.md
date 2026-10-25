# Domain Docs

工程类技能在探查本仓库代码时，应如何消费这里的领域文档。

## Before exploring, read these

- 仓库根目录的 **`GLOSSARY.md`**
- **`docs/adr/`**：读与你要动的区域相关的 ADR

如果这些文件不存在，**安静地继续**。不要报告它们缺失，也不要主动建议创建。`/domain-modeling` 技能（经 `/grill-with-docs` 与 `/improve-codebase-architecture` 到达）会在术语或决定真正落定时按需创建它们。

## Layout

**本仓库是 single-context（单上下文）布局**：全仓库共用一套用语，根目录一份 `GLOSSARY.md`，架构决定放在 `docs/adr/`。

这一结论在初始化时与用户确认过，也已记录在 `docs/agents/workspace.yaml` 的 `layout: single`。

本仓库没有 monorepo 信号（无 `pnpm-workspace.yaml`、无 workspaces 字段、无 `packages/*`），因此不使用多上下文布局。

## File structure

```
/
├── GLOSSARY.md
├── docs/adr/
│   ├── 0001-....md
│   └── 0002-....md
└── src/
```

**当前状态**：`GLOSSARY.md` 与 `docs/adr/` 尚未创建。按上面的规则，这属于正常情况——第一次真正写下术语或架构决定时再建，不要为了初始化而预先创建空文件。

## Use the glossary's vocabulary

当你的产出要命名一个领域概念时（issue 标题、重构提案、假设、测试名），使用 `GLOSSARY.md` 中定义的说法。不要漂移到该文件明确回避的同义词。

如果你需要的概念还不在术语表里，这是一个信号：要么你在发明项目不用的语言（重新考虑），要么这是一个真实的缺口（记下来，交给 `/domain-modeling`）。

## Flag ADR conflicts

如果你的产出与某个既有 ADR 相矛盾，明确指出来，而不是悄悄覆盖：

> _与 ADR-0007（…）冲突，但值得重开，因为……_
