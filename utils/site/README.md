# CANN Recipes 文档站点

`utils/site` 是 CANN Recipes 文档仓库中的静态单页站点。站点使用
`index.html` 展示文档导航、搜索结果和 Markdown 正文，文档源目录位于仓库根目录。

## 当前功能

- 按 `Infer`、`Train`、`Embodied_AI` 和 `CANN Features` 四个分类浏览文档。
- 根据 `content/index.json` 动态生成分类、子分类、模型和报告导航。
- 默认使用中文界面，支持切换到英文；同时支持报告搜索、中文/英文文档版本切换、面包屑导航和报告目录。
- 使用 `marked.js` 渲染 Markdown，使用 `highlight.js` 高亮代码，使用 KaTeX 和 Mermaid 处理公式及图表。
- 支持代码复制、深色模式、响应式布局、侧边栏折叠和报告问题反馈。
- 文档和图片均从当前仓库同源路径读取，构建过程不下载额外的正文或图片缓存。

## 目录结构

```text
utils/site/
├── index.html                 # 单页应用及前端逻辑
├── build_content.py            # 生成导航和搜索索引
├── serve.py                   # 本地开发服务器
├── content/
│   ├── index.json              # 分类、模型和报告元数据
│   ├── search-index.json       # 报告全文搜索索引
│   └── homepage_samples.json   # 首页展示的领域和模型样例
├── cann_logo.png              # 站点 Logo
└── 404.html                   # 静态托管环境的回退页面
```

站点技术规范位于仓库根目录的 `skill/blog_site_skill.md`。

## 文档来源

构建脚本从 GitCode 主仓 `cann/cann-recipes-docs` 的以下目录发现文档：

| 分类 | 仓库目录 | 目录布局 |
|------|----------|----------|
| Infer | `infer/` | `子分类/模型/zh`、`子分类/模型/en` |
| Train | `train/` | `子分类/模型/zh`、`子分类/模型/en` |
| Embodied_AI | `embodied_ai/` | `子分类/模型/zh`、`子分类/模型/en` |
| CANN Features | `cann_features/` | `zh`、`en`（无模型层） |

图片通常放在模型或分类目录下的 `figures/` 中。中文文档放在 `zh/`，英文文档放在同级 `en/`，文件名和相对路径保持对应。

## 本地运行

在仓库根目录执行：

```powershell
python utils\\site\\build_content.py
python utils\\site\\serve.py --port 8001
```

浏览器访问 `http://127.0.0.1:8001`。`serve.py` 会将站点目录映射到根路径，同时提供仓库根目录中的 Markdown 文档和图片，因此本地运行不需要额外代理服务。

## 静态部署

部署时将 `utils/site` 目录及其所需的 `content/` 文件发布到静态 Web 服务器，若站点部署在子路径下，应确保服务器正确提供 `index.html`、`content/` 和仓库文档目录，并保留相对资源路径。

## 开发注意事项

- `build_content.py` 仅使用 Python 标准库，不要引入额外依赖。
- 新增分类时必须同步更新 `build_content.py` 的 `REPO_CONFIG`，并重新生成两个索引文件。
- 不要手工维护与 manifest 冲突的模型列表；前端硬编码数据只用于 manifest 不可用时的回退。
- 修改前端后，应验证首页、分类导航、报告渲染、搜索、语言切换和无网络时的已生成索引。
