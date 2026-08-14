# CANN Recipes Docs 项目说明

## 项目范围

- 本仓库维护 CANN Recipes 的技术文档和 CANN 原子特性文档。
- 文档分类包括 `infer/`、`train/`、`embodied_ai/` 和 `cann_features/`。
- 不要把源码、模型权重、数据集、构建产物或会话归档提交到本仓库。

## 文档目录

- 模型或专题文档使用 `<content-directory>/zh/`、`<content-directory>/en/` 和同级 `figures/`。
- 中文文档放在 `zh/`，英文文档放在同级 `en/`；对应中英文文档使用相同文件名和下级路径。
- `cann_features/` 不增加模型层，直接使用 `cann_features/zh/`、`cann_features/en/` 和 `cann_features/figures/`。
- 如果新增中文文档时 `zh/` 不存在，先创建 `zh/` 并将中文文档放入其中，再生成对应英文文档。
- 图片统一放在同级 `figures/`，文档从 `zh/` 或 `en/` 使用 `../figures/<file>` 引用。

## 文档规范

- 新增或更新文档前阅读 `skill/new_document_skill.md`。
- 中文文档翻译为英文时阅读 `skill/translate_english_document_skill.md`，正文重点参考其中链接的官方 CANN translation skill。
- 翻译必须保持原文内容、结构、代码、表格、图片和链接语义一致，不得删减或扩写。
- 英文文档优先使用官方英文网页链接；没有对应英文页面时保留中文链接并标记 `(Chinese)`。
- 新增模型或专题后，必须在根目录 `README.md` 对应分类的样例列表中增加不重复且有效的中文文档入口。

## `static-meta`

- 每篇文档必须有且只有一个位于文档末尾的 `static-meta` 块。
- 字段、枚举和填写规则以 `utils/references/static_meta_infos.md` 为准；不要在单篇文档中创建未定义字段或枚举值。
- `cann_features/**` 文档只允许 `sidebarTitle` 和单值 `cannFeatures`。
- 中英文同名文档同时存在时，除 `sidebarTitle` 外的 `static-meta` 字段和值必须完全一致。

## 验证命令

在仓库根目录执行：

```powershell
python utils\scripts\validate_metadata.py
python -m py_compile utils\scripts\metadata_enums.py utils\scripts\metadata_labels.py utils\scripts\validate_metadata.py
```

修改后检查 Markdown 相对链接、标题锚点、图片路径和外部 URL。不要通过删除字段、放宽校验或跳过文档来绕过错误。

## 变更边界

- 保持修改聚焦于用户请求，保留无关的用户改动。
- 不自动执行 `git commit`、`git push` 或其他远程写操作。
- 提交前排除 `__pycache__/`、临时文件和其他生成物，除非用户明确要求纳入。
