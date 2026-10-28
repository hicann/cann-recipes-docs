# 首页项目列表 JSON 模板

[`CANN_homepage.json`](CANN_homepage.json) 是[CANN首页](https://gitcode.com/cann)展示模型实践项目的源数据。文件顶层是一个 JSON 对象，`CANN_homepage` 是模块列表，每个模块的项目嵌套在 `projects` 数组中。

## 数据格式

文件使用 UTF-8 编码和标准 JSON 格式。顶层 `CANN_homepage` 是包含 5 个模块对象的数组，可以直接解析整个文件；不要添加注释或尾随逗号。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `CANN_homepage` | 数组 | 首页展示的 5 个模块 |
| `module_name` | 非空字符串 | 模块标签，如“训练” |
| `module_order` | 整数 | 模块显示顺序，从 1 开始 |
| `repository_url` | 非空字符串 | 模块的仓库链接 |
| `projects` | 数组 | 当前模块的项目，允许 0 到 4 项 |
| `projects[].name` | 非空字符串 | 项目卡片标题 |
| `projects[].description` | 非空字符串 | 项目卡片描述，建议用一两句话 |
| `projects[].url` | 非空字符串 | 点击卡片后打开的链接；填写实际可访问的链接 |
| `projects[].order` | 整数 | 同模块内的显示顺序，从 1 开始 |

## 填写约束

- 整个文件恰好有 5 个模块，`module_name` 不重复，`module_order` 使用不重复的 1 到 5。
- 每个模块允许 0 到 4 个项目；同模块内 `order` 从 1 连续编号且不重复。
- 所有项目必须上传真实项目 URL，避免无法跳转。
- 首页按 `module_order` 和 `order` 升序展示，不依赖模块对象或项目数组的原始顺序。

## 模型调优样例列表

[`model_tuning.json`](model_tuning.json) 保存首页“模型调优”模块使用的样例列表。顶层 `model_tuning` 是数组，当前包含 3 个模型调优样例；每条记录包含名称、显示顺序和跳转链接。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `model_tuning` | 数组 | 模型调优样例列表 |
| `model_tuning[].name` | 非空字符串 | 样例名称 |
| `model_tuning[].order` | 整数 | 显示顺序，从 1 开始且不重复 |
| `model_tuning[].url` | 非空字符串 | 样例详情页链接 |

样例按 `order` 升序展示；新增样例时应同步更新顺序，并填写可访问的链接。
