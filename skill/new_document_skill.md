---
name: new_document_skill
description: 为 CANN Recipes Docs 仓库新增或更新技术文档，选择技术分类，按 zh/en/figures 组织文档与图片，生成 static-meta 静态字段并执行校验。用户要求新增文章、迁移中英文技术文档、补充 static-meta、扩展文档字段或枚举值时使用。
---

# CANN Recipes 新增文档

在仓库中新增或更新技术文档时，按以下流程执行。静态字段的定义、枚举、标签和填写规则以 [static_meta_infos.md](../utils/references/static_meta_infos.md) 为准，不在本 Skill 中重复维护。需要将中文技术文档翻译为英文时，继续引用 [英文文档翻译 Skill](./translate_english_document_skill.md)；翻译专属要求以该 Skill 为准。

## 1. 确认文档目录

### 1.1 技术分类

先按内容选择技术分类，再建立模型或专题目录：

```text
infer/
├─ llm/<model>/                 # 大语言模型推理
├─ multimodal/<model>/          # 多模态生成与理解
└─ recommendation/<model>/      # 推荐系统推理

train/
├─ pretrain/<model>/             # 预训练
└─ rl/<model>/                   # 强化学习

embodied_ai/
├─ 3d_vision/<model>/            # 3D 视觉
├─ manipulation/<model>/         # 机械臂操作
├─ navigation/<model>/           # 导航
└─ world_model/<model>/          # 世界模型

cann_features/                   # CANN 原子特性
```

`<model>` 使用小写英文字母与下划线的组合，优先采用稳定的模型名或技术主题名。不要按作者、日期或临时任务名创建分类目录。

### 1.2 中英文档与图片附件

把最末级模型或专题目录作为一个“内容目录”，在其中使用固定的三目录结构：

```text
<content-directory>/
├─ zh/                                   # 中文技术文档
│  └─ optimization.md
├─ en/                                   # 英文技术文档
│  └─ optimization.md
└─ figures/                              # 中英文文档引用的图片
   ├─ architecture.png
   ├─ performance_zh.png
   └─ performance_en.png
```

目录分布规则：

- 中文技术文档只放在 `zh/`，英文技术文档只放在同级 `en/`。
- 同一篇中英文技术文档使用相同文件名，并保持相同的下级路径，便于建立语言对应关系。
- 图片统一放在同级 `figures/`，不要散落在 `zh/`、`en/` 或仓库根目录。
- 中英文共用的图片只保留一份；图片含语言文字时，用 `_zh`、`_en` 后缀区分。
- `zh/` 和 `en/` 中的文档使用 `../figures/<file>` 引用图片。例如 `![架构](../figures/architecture.png)`。
- 新增内容目录时建立 `zh/`、`en/`、`figures/` 三个同级目录；暂时没有英文版本时，允许 `en/` 为空。
- 不在 `figures/` 中放技术文档、脚本、日志或其他非图片文件。

模型和专题文档在 `<model>/` 下使用该结构。`cann_features/` 不再增加模型层，直接使用：

```text
cann_features/
├─ zh/
├─ en/
└─ figures/
```

`cann_features/**` 文档只允许在 `static-meta` 中填写 `sidebarTitle` 和 `cannFeatures`。`cannFeatures` 必须是只含一个自身主题特性 ID 的数组；同一特性 ID 可以对应多篇文档，用于后续一对多映射，不能把关联或依赖特性作为额外标签。

## 2. 读取静态字段规范

生成或修改 `static-meta` 前，读取 [static_meta_infos.md](../utils/references/static_meta_infos.md) 中与任务相关的内容：

- 新增技术文档：读取第 2～4 章和第 6.1 节。
- 扩展已有字段的枚举值：读取第 6.2 节。
- 新增字段或枚举类：读取第 6.3 节。
- 运行校验：读取第 6.4 节。

不要凭模型常识补充字段值。只使用正文能够明确支持的技术信息。`frameworks` 仅限上层专用框架，不考虑底层基础框架；添加前考证其概念层级。

## 3. 新增或更新文档

### 3.1 文档处理

- 阅读完整正文，确认 H1、技术场景和目标目录。
- 保留正文原有结构、图片、链接和代码示例。
- 按 [新增文档并生成 static-meta](../utils/references/static_meta_infos.md#61-新增文档并生成-static-meta) 的 checklist 生成静态字段。
- 将唯一的 `static-meta` 放在文档末尾。
- 非必填公共字段没有明确依据时直接省略，不使用 `none` 或空数组占位。
- 检查当前路径是否要求场景子类扩展字段。
- 新增文档时，检查仓库主页 `README.md`，在对应技术分类的“样例列表”中增加该模型或专题的文档入口。
- 搬运原文档时，若正文出现“本仓库”“本项目”或其他类似的指代，必须根据上下文或来源改为对应的第三方仓库或项目名称；禁止使用“本 xx”继续指代 `cann-recipes-infer`、`cann-recipes-train`、`cann-recipes-embodied-ai` 等外部仓库。

对于 `cann_features/**` 文档，`static-meta` 只能包含 `sidebarTitle` 和单值 `cannFeatures`，且 `cannFeatures` 必须代表当前文档自身主题。

### 3.2 链接优先级

添加或迁移链接前，先按文件名、标题和主题在本仓库中查找目标。按以下优先级选择链接：

- 本仓库中存在目标技术文档时，必须使用从当前文档出发的相对路径，不得改用 `cann-recipes-infer`、`cann-recipes-train` 或 `cann-recipes-embodied-ai` 的 GitCode 文档链接。
- 优先链接同语言文档；同语言版本不存在时，可以链接本仓库中的另一语言版本，并在链接文字中注明语言。
- 将原项目文档搬运到本仓库时，必须同步改写其中已有的引用路径和链接名称。对于未随文档一并迁移、仍指向原项目的源码、配置或脚本引用，链接文字必须只保留文件名（例如 `xxx.py`、`config.yaml`），不得出现原项目多层目录；链接地址必须同步改为原 GitCode 项目的网页链接，并保留正确的仓库、分支/版本和文件路径（优先使用对应文件页面），不得保留原项目相对路径、本机路径或编辑器专用 URL。
- 原项目文档链接必须先判断目标类型：已迁移到本仓库的文档改为本仓库相对路径；未迁移的文档、源码、配置、脚本和发布版本说明分别改为原项目对应的 GitCode 网页页面。
- 正文中提到的外部源码文件、类、函数、配置文件、配置项或可执行脚本，只要读者需要据此定位、修改或运行，就必须提供可直接跳转的 GitCode 页面链接。文件引用链接到对应文件页面，函数或类引用优先链接到包含该符号的文件并保留行号/代码片段定位；无法确认准确符号位置时，至少链接到对应文件页面，并明确说明需要在该文件中查找，不得只留下裸文件名或让读者自行搜索。
- 本仓库没有对应内容时，才使用外部官方文档或来源仓库 URL；不得使用本机绝对路径、`file://`、`file+.vscode-resource`、编辑器预览 URL 或无法复现的临时下载 URL。
- 图片优先引用同级 `figures/` 中的本地文件；优先使用项目中存在的图片文件，避免使用外部图片；若必须使用外部图片，应先确认允许长期访问且返回图片内容，并将图片链接与图片跳转链接分开处理。
- 链接文字应描述目标内容，不得把原项目目录层级、仓库内部前缀或临时路径作为展示文字；同一目标在正文中保持名称一致。

### 3.3 逐一检查链接

完成文档修改后，检查该文档中的每一个 Markdown 链接、图片引用以及 HTML `href`、`src`：

- 相对文件路径必须从当前文档所在目录解析，并指向真实存在的文件；检查路径大小写、文件名和扩展名。
- 检查普通 Markdown 链接、引用式链接、自动链接、HTML `href`/`src` 以及链接定义；代码块中的命令、脚本参数和示例 URL 若会被读者复制使用，也必须检查是否仍包含原项目路径或本机路径。
- 带 `#anchor` 的链接必须指向目标文档中真实存在的标题锚点；保留 `?` 查询参数或行号片段时，必须确认其仍适用于迁移后的 GitCode 页面。
- `zh/`、`en/` 和 `figures/` 之间的相对层级必须正确，不能根据仓库根目录臆算路径。
- 外部 URL 必须实际访问并确认不是 404、登录页、重定向占位页或失效页面。
- GitCode 源码链接使用文件页面 URL；外部图片只有在返回图片 Content-Type 时才能用于 `src`，否则改用本地图片或文件页面跳转链接。
- 对正文中的代码文件名、类名、函数名、配置文件名、配置项和脚本命令进行全文检查；凡是用于指导读者定位、修改或运行的外部实现引用，都必须有对应的可访问源码或配置页面链接，且链接文字应为文件名或符号名，不得只写未链接的反引号文本。
- 迁移技术文档后，重新检查全文所有链接，不只检查本次新增的链接。

可以先使用以下命令列出待检查项，但不能只凭搜索结果判断链接有效：

```powershell
rg -n "\]\(|^\s*\[[^]]+\]:|<https?://|src=|href=" <document-path>
```

逐项检查全部通过后才能交付；发现失效相对路径时，先在本仓库寻找正确目标并修复相对路径，确认本仓库不存在目标后才改用 GitCode URL。

### 3.4 更新主页样例列表

新增文档时，必须同步检查仓库根目录的 `README.md`，并按文档所在目录更新对应分类下的“样例列表”：

- `infer/llm/**`、`infer/multimodal/**`、`infer/recommendation/**` 分别更新“推理”下对应子分类的样例列表。
- `train/pretrain/**`、`train/rl/**` 分别更新“训练”下对应子分类的样例列表。
- `embodied_ai/3d_vision/**`、`embodied_ai/manipulation/**`、`embodied_ai/navigation/**`、`embodied_ai/world_model/**` 分别更新“具身智能”下对应子分类的样例列表。
- `cann_features/**` 更新“CANN 原子特性”样例列表。

使用模型或专题的正式名称作为链接文字，链接到本仓库中新建的中文文档，并使用从 `README.md` 出发的有效相对路径。保持当前列表的 Markdown 格式和分类顺序；若该模型或专题已存在条目，则检查并更新已有入口，不得添加重复项。新增条目后逐一验证链接目标真实存在。

## 4. 扩展字段或枚举

- 已有字段增加选项：按 [6.2 扩展已有字段](../utils/references/static_meta_infos.md#62-扩展已有字段新增枚举值) 操作，只新增枚举值及对应标签。
- 新增字段：按 [6.3 新增字段](../utils/references/static_meta_infos.md#63-新增字段扩展字段枚举类) 操作，扩展 `MetadataField`；受控字段再新增专属 `StrEnum`。

先修改 `static_meta_infos.md`，再同步 Python 枚举、标签和校验器。不要在单篇文档中临时创造字段或枚举值。

## 5. 校验

在仓库根目录运行：

```powershell
python utils\scripts\validate_metadata.py
```

修改 Python 文件后，再运行：

```powershell
python -m py_compile utils\scripts\metadata_enums.py utils\scripts\metadata_labels.py utils\scripts\validate_metadata.py
```

修复全部错误后再交付。不要通过删除必填字段、放宽规则或跳过文档来绕过校验。

## 6. 完成检查

- 文档位于正确的场景目录。
- 文档有且只有一个 H1 和一个末尾 `static-meta`。
- `sidebarTitle` 简短、有效且同目录不重复。
- 所有静态字段均有正文依据，并使用规范中的机器 ID。
- 场景子类字段符合目录范围。
- 新增文档对应的模型或专题已加入 `README.md` 中正确分类的样例列表，且入口不重复、链接有效。
- 显式封面引用真实存在。
- 所有相对文件链接和标题锚点均有效。
- 本仓库已有的技术文档没有链接到 infer、train 或 embodied-ai 外部仓库。
- 所有保留的外部 URL 均已逐一验证可访问。
- 静态字段校验通过。

