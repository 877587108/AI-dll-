# Issue tracker: Local Markdown

本仓库的 issue 与 spec 以 markdown 文件形式存放在 `.scratch/` 下。

## Conventions

- 一个功能一个目录：`.scratch/<feature-slug>/`
- spec 是 `.scratch/<feature-slug>/spec.md`
- 实现 issue 一个文件一张票，放在 `.scratch/<feature-slug>/issues/<NN>-<slug>.md`，从 `01` 编号；**不要**把所有票合并成一个文件
- 分诊状态写在每个 issue 文件靠上的 `Status:` 行（角色字符串见 `triage-labels.md`）
- 评论与讨论历史追加到文件末尾的 `## Comments` 标题下

## When a skill says "publish to the issue tracker"

在 `.scratch/<feature-slug>/` 下新建文件（目录不存在就创建）。

## When a skill says "fetch the relevant ticket"

读取对应路径的文件。用户通常直接给出路径或票号。

## Wayfinding operations

供 `/wayfinder` 使用。**地图**是一个文件，每张票对应一个**子**文件。

- **Map**：`.scratch/<effort>/map.md`（Notes / Decisions-so-far / Fog 正文）
- **Child ticket**：`.scratch/<effort>/issues/NN-<slug>.md`，从 `01` 编号，正文写问题。`Type:` 行记录票型（`research`/`prototype`/`grilling`/`task`）；`Status:` 行记录 `claimed`/`resolved`
- **Blocking**：靠上的 `Blocked by: NN, NN` 行。它列出的每个文件都 `resolved` 时，这张票才解除阻塞
- **Frontier**：扫 `.scratch/<effort>/issues/`，找出未关闭、未阻塞、未认领的文件；编号小的优先
- **Claim**：设 `Status: claimed`，加一行 `Assignees: <你的名字>`，在做任何工作前保存。这个名字会被原样读回——本地没有登录用户，票文件是它唯一能存在的地方
- **Resolve**：在 `## Answer` 标题下追加答案，设 `Status: resolved`，然后把一条上下文指针（要点 + 链接）追加到 `map.md` 的 Decisions-so-far
