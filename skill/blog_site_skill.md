# CANN Recipes 文档站点技术说明

## 1. 文档范围

本文档说明 `utils/site` 当前实现的站点结构、内容生成、前端加载链路、本地运行方式、已实现能力和修改约束。描述以当前代码为准，不包含仓库中不存在的服务、缓存或部署工作流。

## 2. 当前文件及定位

| 文件或目录 | 用处 | 定位 |
|------------|------|------|
| `utils/site/index.html` | 单页应用界面、导航、搜索、筛选、报告渲染和交互逻辑 | 站点前端入口 |
| `utils/site/build_content.py` | 扫描本地文档，解析标题、`static-meta` 和正文文本，生成索引 | 构建期索引工具 |
| `utils/site/serve.py` | 本地提供站点文件及仓库 Markdown、图片 | 本地开发服务器 |
| `utils/site/content/index.json` | 分类、子分类、模型、报告路径和元数据 | 导航 manifest |
| `utils/site/content/search-index.json` | 报告纯文本搜索内容 | 全文搜索索引 |
| `utils/site/content/homepage_samples.json` | 首页展示的领域和模型样例配置 | 首页文档列表配置 |
| `utils/site/docs/implemented_features.md` | 保存较详细的功能记录 | 功能记录，不参与运行时逻辑 |
| `utils/site/cann_logo.png` | 站点 Logo | 前端静态资源 |
| `utils/site/404.html` | 静态托管环境的未知路径回退页 | 部署兼容文件 |

`content/` 包含导航索引、全文搜索索引和首页模型样例配置。

## 3. 分类和文档布局

当前分类均来自主仓 `cann/cann-recipes-docs`：

| 分类键 | 扫描目录 | 结构 |
|--------|----------|------|
| `Infer` | `infer/` | 子分类 → 模型 → `zh`/`en` |
| `Train` | `train/` | 子分类 → 模型 → `zh`/`en` |
| `Embodied_AI` | `embodied_ai/` | 子分类 → 模型 → `zh`/`en` |
| `CANN Features` | `cann_features/` | 直接扫描 `zh`/`en`，无模型层 |

分类键、索引键、显示文字和前端配置必须保持一致。

## 4. 内容生成

`build_content.py` 只扫描本地目录：

- 递归发现四类文档的子分类、模型和 Markdown 报告。
- 读取 `static-meta`，生成筛选字段、枚举和值标签。
- 提取中英文标题、侧边栏短标题、源路径和搜索文本。
- 写入 `content/index.json` 与 `content/search-index.json`。

在仓库根目录执行：

```powershell
python utils\site\build_content.py
```

## 5. 前端加载链路

```text
读取 index.json 和 search-index.json
        ↓
根据 manifest 生成分类、子分类、模型和报告导航
        ↓
点击报告后，从 sourcePath 同源读取 Markdown
        ↓
通过 marked.js 渲染 Markdown
        ↓
使用 highlight.js、KaTeX 和 Mermaid 增强代码、公式和图表
```

图片使用仓库中的相对 `figures/` 路径；前端在渲染前将相对路径解析为同源 URL。站点不依赖本地代理、第三方 CORS 代理或远程正文缓存。

## 6. 已实现功能

### 6.1 导航和报告阅读

- 支持分类、子分类、模型、报告四级导航，侧边栏可展开和折叠。
- 支持报告面包屑、报告目录、章节锚点、上一篇/下一篇和阅读进度条。
- 支持 Markdown 标题、列表、链接、引用、代码块、表格、图片、公式和 Mermaid 图表。
- 代码块支持语法高亮和复制；宽表格在正文区域内横向滚动。

### 6.2 搜索和筛选

- 按报告文件名、标题、模型和分类进行全文搜索。
- 支持内容范围多选，以及分类下的子分类组合筛选。
- 支持按量化、并行方式、算子、CANN 特性、硬件、框架等技术属性多选筛选。
- 同一属性内取并集，不同属性之间取交集，并支持 `none` 空值。
- 搜索结果显示当前范围、结果数量和响应式报告卡片。

### 6.3 首页、反馈和界面

- 首页根据 manifest 生成报告列表和分类入口；没有封面时使用分类占位封面。
- 真实报告提供内容纠错、Bug、功能建议和技术讨论四类反馈入口，并自动填充报告路径和阅读地址。
- 支持浅色/深色主题、中文/英文 UI 切换，以及英文文档缺失提示。
- 使用 Material Symbols 图标、响应式布局、长文本换行和图标控件无障碍标签。

## 7. 本地运行

在仓库根目录执行：

```powershell
python utils\site\build_content.py
python utils\site\serve.py --port 8001
```

访问 `http://127.0.0.1:8001`。服务器将站点映射到根路径，同时提供 `infer/`、`train/`、`embodied_ai/` 和 `cann_features/` 下的文档与图片。

## 8. 修改约束

新增模型文档后，必须将新模型与对应领域当前首页展示的模型进行对比；如新模型更适合作为最新样例，应更新 `utils/site/content/homepage_samples.json`，确保 Infer、Train、Embodied_AI 各领域至少保留一个首页模型样例，并重新生成站点索引。

1. 保持单文件 SPA 结构，除非需求明确要求拆分前端。
2. `build_content.py` 只能依赖 Python 标准库，只读取当前仓库文件。
3. 分类键和源目录必须同步更新，不能只修改显示文字。
4. 报告正文、代码文件和实现符号引用应提供可定位的 GitCode 链接；源码链接不能伪装成站点文档路径。
5. 相对图片链接必须解析到实际 `figures/` 路径，不能把源码路径或外部仓库路径误当成本仓库文件。
6. 不要提交源码、模型权重、数据集、缓存目录或临时构建产物。
7. 修改后重新生成两个索引，并检查 JSON、Markdown 链接、图片路径和浏览器控制台错误。

## 9. 验证清单

| 检查项 | 预期结果 |
|--------|----------|
| 执行 `build_content.py` | 无异常，两个索引文件可解析 |
| 访问 `http://127.0.0.1:8001` | 首页直接加载，无路径重定向 |
| 展开四个分类 | 分类和模型与 `content/index.json` 一致 |
| 打开中英文报告 | 标题、表格、代码、公式、图片正常 |
| 使用搜索和技术属性筛选 | 结果范围、数量和组合逻辑正确 |
| 点击目录、面包屑和前后篇 | 页面跳转和锚点正确 |
| 切换主题和语言 | UI 状态持久化，英文缺失时有提示 |

修改完成后删除 `__pycache__/` 和其他临时文件，不自动执行 `git commit` 或 `git push`。
