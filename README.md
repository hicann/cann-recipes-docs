<h1 align="center">CANN-RECIPES-DOCS</h1>

<p align="center">
  CANN Recipes 技术文档中心<br>
  汇聚推理、训练、具身智能与 CANN 原子特性优化文档，沉淀可复用的模型优化经验与工程实践
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Docs-Online%20Preview-orange.svg" alt="Online Preview"></a>
  <img src="https://img.shields.io/badge/Content-CANN%20Recipes-blue.svg" alt="CANN Recipes">
</p>

<p align="center">
  <a href="#-概览">📖 概览</a> ·
  <a href="#-内容分类">🗂️ 内容分类</a> ·
  <a href="#-使用方法">🧭 使用方法</a> ·
  <a href="#-项目结构">📁 项目结构</a> ·
  <a href="#-参与贡献">🤝 参与贡献</a>
</p>

---

## 📖 概览

本仓库是 CANN Recipes 的统一技术文档仓，主要承载从各 recipe 代码仓中沉淀出来的模型优化文档、场景实践文档和 CANN 原子特性说明。

文档内容面向以下目标：

| 目标 | 说明 |
|---|---|
| 沉淀优化经验 | 记录模型适配、性能优化、并行切分、量化、算子优化等实践过程。 |
| 统一内容入口 | 集中维护**推理、训练、具身智能**等方向的技术文档，提供 CANN 原子特性文档。 |
| 支持结构化检索 | 通过文档末尾的 `static-meta` 静态字段，为分类筛选、侧栏标题和封面选择提供基础信息。 |


> 在线页面仅作为文档阅读和检索入口，仓库核心仍是技术文档内容。

---

## 🗂️ 内容分类

| 目录 | 内容概要 | 关联代码仓 |
|---|---|---|
| [infer/](infer/) | **推理**：涵盖大语言模型、多模态生成、推荐系统等推理优化文档。 | [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) |
| [train/](train/) | **训练**：涵盖预训练、强化学习、长序列训练等训练优化文档。 | [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train) |
| [embodied_ai/](embodied_ai/) | **具身智能**：涵盖 3D 视觉、机械臂操作、导航、世界模型等优化文档。 | [cann-recipes-embodied-ai](https://gitcode.com/cann/cann-recipes-embodied-ai) |
| [cann_features/](cann_features/) | **CANN 原子特性**：涵盖 MultiStream、SuperKernel、Prefetch、NPUGraph 等跨场景公共能力文档。 | - |

### 推理

| 应用场景 | 目录 | 示例内容 |
|---|---|---|
| 大语言模型 | `infer/llm/` | DeepSeek、Qwen、Kimi、LongCat、GPT-OSS 等模型推理优化。 |
| 多模态生成 | `infer/multimodal/` | 图像生成、视频生成、多模态模型推理优化。 |
| 推荐系统 | `infer/recommendation/` | 推荐模型推理部署与性能优化。 |

#### 样例列表

**大语言模型**

- [DeepSeek-R1](infer/llm/deepseek_r1/zh/deepseek_r1_decode_optimization.md)

**多模态生成**


**推荐系统**


### 训练

| 应用场景 | 目录 | 示例内容 |
|---|---|---|
| 预训练 | `train/pretrain/` | 大模型预训练优化方案。 |
| 强化学习 | `train/rl/` | RL 训练、长序列训练、解码策略优化等。 |

#### 样例列表

**预训练**


**强化学习**


### 具身智能

| 应用场景 | 目录 | 示例内容 |
|---|---|---|
| 3D 视觉 | `embodied_ai/3d_vision/` | 3D 生成、几何重建、渲染优化等。 |
| 机械臂操作 | `embodied_ai/manipulation/` | 策略模型、机械臂控制模型优化。 |
| 导航 | `embodied_ai/navigation/` | 机器人导航模型优化。 |
| 世界模型 | `embodied_ai/world_model/` | 世界模型推理与训练优化。 |

#### 样例列表

**3D 视觉**


**机械臂操作**


**导航**


**世界模型**


### CANN 原子特性

| 应用场景 | 目录 | 示例内容 |
|---|---|---|
| 原子特性 | `cann_features/` | MultiStream、SuperKernel、Prefetch、NPUGraph 等跨场景公共能力文档。 |

#### 样例列表

- [NPU 多流原理](cann_features/zh/multi_stream_principles.md)
- [NPU 图模式优化原理](cann_features/zh/npu_graph_optimization.md)
- [NPU Prefetch 原理](cann_features/zh/prefetch_principles.md)
- [SuperKernel 原理](cann_features/zh/super_kernel.md)

---

## 📁 项目结构

<details>
<summary>点击展开目录说明</summary>

```text
.
├─ infer/                        # 推理类优化实践文档
│  ├─ llm/                       # 大语言模型推理文档
│  ├─ multimodal/                # 多模态推理文档
│  └─ recommendation/            # 推荐系统文档
├─ train/                        # 训练类优化实践文档
│  ├─ pretrain/                  # 预训练文档
│  └─ rl/                        # 强化学习训练文档
├─ embodied_ai/                     # 具身智能优化实践文档
├─ cann_features/                # CANN 原子特性文档
├─ skill/
│  ├─ new_document_skill.md      # 新增文档 Skill
│  └─ translate_english_document_skill.md # 英文文档翻译 Skill
├─ utils/
│  ├─ references/
│  │  └─ static_meta_infos.md    # 静态字段规范
│  └─ scripts/
│     ├─ metadata_enums.py       # 静态字段枚举
│     ├─ metadata_labels.py      # 中英文标签映射
│     └─ validate_metadata.py    # 静态字段校验
├─ CONTRIBUTION.md               # 贡献指南
├─ LICENSE                       # Apache 2.0 许可证
└─ README.md                     # 项目说明
```

</details>

---

## 🤝 参与贡献

欢迎各种形式的文档贡献，包括新增优化文档、改进现有内容、补充英文翻译和修复问题。

请参阅 [贡献指南](CONTRIBUTION.md) 了解文档目录、静态字段和校验流程。

若有疑问或共建意向，可以参与[Recipes SIG 例会](https://gitcode.com/cann/community/blob/master/CANN/sigs/recipes/README.md)与我们交流~

---

<p align="center">
  <sub>CANN Recipes 技术文档中心</sub>
</p>
