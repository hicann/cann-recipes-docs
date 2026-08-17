# CANN Recipes Docs 文档静态字段规范

## 1. 概述

每篇技术文档通过少量、稳定、可校验的静态字段描述其核心技术属性。静态字段用于 Blog 的侧栏标题、分组筛选和封面；不在文档正文中重复展示。

本规范将静态字段分为两类：公共字段和场景子类扩展字段。

公共字段适用于所有技术文档，目前仅允许维护以下 8 个字段：

- `sidebarTitle`
- `quantization`
- `parallelism`
- `operator`
- `cannFeatures`
- `hardware`
- `frameworks`
- `cover`

场景子类扩展字段只允许在指定目录下使用，其他目录不得填写：

- `llmSpeculativeInference`：仅限 `infer/llm/**`
- `multimodalDitCache`：仅限 `infer/multimodal/**`

### 1.1 相关目录

静态字段相关文件集中在 `utils/` 下，新增文档流程单独放在 `add-cann-recipes-document/` 下：

```text
.
├─ utils/
│  ├─ references/
│  │  └─ static_meta_infos.md    # 字段说明、取值和填写规则
│  └─ scripts/
│     ├─ metadata_enums.py       # 字段名和枚举 ID
│     ├─ metadata_labels.py      # 枚举 ID 对应的中英文名称
│     └─ validate_metadata.py    # static-meta 校验
└─ add-cann-recipes-document/
   └─ SKILL.md                   # 新增文档流程
```

### 1.2 枚举类

`metadata_enums.py` 使用 `StrEnum` 定义允许写入 `static-meta` 的值。枚举值是稳定的机器 ID，使用小写形式，不直接写中文名称。

```python
class Parallelism(StrEnum):
    DATA_PARALLEL = "data-parallel"
    TENSOR_PARALLEL = "tensor-parallel"
    CONTEXT_PARALLEL = "context-parallel"
```

文档中填写枚举的 `value`：

```json
"parallelism": ["tensor-parallel", "context-parallel"]
```

校验器通过 `enum_values(Parallelism)` 取得全部合法值。如果需要增加枚举，先在本文对应章节补充定义，再修改枚举类。

### 1.3 标签字典

`metadata_labels.py` 使用字典保存字段和枚举值的中英文名称。字典键必须和 `metadata_enums.py` 中的机器 ID 一致。

```python
FIELD_LABELS = {
    "parallelism": {"en": "Parallelism", "zh": "并行切分"},
}

VALUE_LABELS = {
    "parallelism": {
        "tensor-parallel": {"en": "Tensor Parallel", "zh": "张量并行"},
    },
}
```

前端筛选使用机器 ID 做匹配，再按当前语言读取名称。例如中文页面读取 `VALUE_LABELS["parallelism"]["tensor-parallel"]["zh"]`，显示“张量并行”。文档的 `static-meta` 中仍然只填写 `tensor-parallel`。

### 1.4 整体使用路径

```text
static_meta_infos.md 定义字段
    -> metadata_enums.py 定义机器 ID
    -> metadata_labels.py 定义中英文名称
    -> 技术文档填写 static-meta
    -> validate_metadata.py 校验
    -> 构建脚本生成索引并交给前端使用
```

新增字段或枚举时按这条路径依次修改，避免规范、代码和页面展示不一致。校验器只检查现有内容，不会自动补写缺失字段或推断枚举值。

文档 H1、目录层级、文件名、来源仓库和全文搜索内容均由构建过程派生，禁止重复写入静态字段。

## 2. 通用规则

- 静态字段放在技术文档末尾，使用 HTML 注释包裹 JSON，方便 agent 从尾部扫描和校验。
- 除 `sidebarTitle` 外，所有机器可读值均使用小写 ASCII `kebab-case`；数值精度可使用 `int8`、`fp8` 等行业通用写法。
- 数组字段使用受控枚举；数组中不得重复，空数组应直接省略。
- `quantization`、`parallelism`、`operator`、`cannFeatures`、`hardware` 和 `frameworks` 均为非必填字段。若正文未提供明确的信息或依据，无需填写对应字段，不使用 `none` 或空数组占位。
- `cann_features/**` 目录下的文档是 CANN 原子特性专题文档，只允许填写 `sidebarTitle` 和 `cannFeatures` 两个字段，且两个字段均为必填。
- `cann_features/**` 文档的 `cannFeatures` 只能填写一个与当前文档主题一致的 CANN 特性 ID，不得把前置依赖或关联特性混入本字段；同一个特性 ID 可以对应多篇文档，用于后续一对多映射。
- 不允许 `null`、空字符串、未知顶层字段或不在枚举中的值。
- 场景子类扩展字段只在对应目录下填写；不属于该目录的文档必须省略该字段，不填写 `["none"]`。
- 每篇文档最多有一个 `static-meta` 块。只有字段校验成功，才允许提交改动。

```markdown
# DeepSeek-R1 Decode 阶段推理优化

正文内容……

<!-- static-meta
{
  "sidebarTitle": "Decode 阶段推理优化",
  "quantization": ["int8"],
  "parallelism": ["tensor-parallel", "context-parallel"],
  "operator": ["ascendc", "tilelang"],
  "cannFeatures": ["multi-stream", "prefetch"],
  "hardware": ["atlas-a3"],
  "frameworks": ["cann-recipes"],
  "cover": {"type": "auto"}
}
-->
```

## 3. 字段总览

### 3.1 公共字段

| 字段 | 类型 | 必填 | 用途 |
|---|---|---:|---|
| `sidebarTitle` | string | 是 | 侧边栏和紧凑导航标题 |
| `quantization` | string[] | 否 | 按量化数据类型筛选 |
| `parallelism` | string[] | 否 | 按并行切分类型筛选 |
| `operator` | string[] | 否 | 按算子开发/优化方式筛选 |
| `cannFeatures` | string[] | 否 | 按固定 CANN 原子特性筛选 |
| `hardware` | string[] | 否 | 已验证或明确适配的硬件 |
| `frameworks` | string[] | 否 | 直接相关的软件框架 |
| `cover` | object | 否 | 控制封面来源；缺省等价于 `auto` |

除 `sidebarTitle` 外，公共字段均为非必填字段。只有正文提供明确的信息或依据时才填写；无法从文档确认时，直接省略该字段。

`cann_features/**` 是例外：文档必须同时填写 `sidebarTitle` 和单值 `cannFeatures`，并且不得填写其他公共字段。`cannFeatures` 表示该文档自身的原子特性主题；同一主题可被多个中英文文档共同引用和筛选。

### 3.2 应用场景子类扩展字段

| 字段 | 适用目录 | 类型 | 必填规则 | 用途 |
|---|---|---|---|---|
| `llmSpeculativeInference` | `infer/llm/**` | string[] | 目录内填写；目录外禁止 | 按 LLM 投机推理方案筛选 |
| `multimodalDitCache` | `infer/multimodal/**` | string[] | 目录内填写；目录外禁止 | 按多模态 DiT Cache 方案筛选 |

新增场景字段规则：

- 字段名使用 `<子类><技术主题>` 的 camelCase，例如 `llmSpeculativeInference`。
- 字段必须服务于该子类下可复用的筛选能力，不为单篇文章新增字段。
- 字段值必须集中放入 `utils/scripts/metadata_enums.py`，展示标签集中放入 `utils/scripts/metadata_labels.py`，不得在文档中临时扩展。
- 前端只在用户进入或筛选到对应子类后展示该字段，避免其他场景出现无意义筛选项。

### 3.3 中文标签规范

前端中文页面不得直接展示机器字段名或枚举 ID，应使用本节维护的中文标签。专有英文名词、技术缩写、框架名、硬件名保持英文不变，例如 `SuperKernel`、MTP、AscendC、NPUGraph、Atlas A3。

字段中文标签如下：

| 字段 | 中文标签 | 说明 |
|---|---|---|
| `sidebarTitle` | 侧栏标题 | 仅作为文档侧栏短标题，不作为筛选项 |
| `quantization` | 量化 | 按权重数据类型筛选 |
| `parallelism` | 并行切分 | 按模型、序列或数据切分方式筛选 |
| `operator` | 算子种类 | 按算子开发或优化方式筛选 |
| `cannFeatures` | CANN 特性 | 按固定 CANN 原子特性筛选 |
| `hardware` | 硬件 | 按已验证或明确适配硬件筛选 |
| `frameworks` | 框架 | 按直接相关框架或工具链筛选 |
| `cover` | 封面 | 控制技术文章封面来源 |
| `llmSpeculativeInference` | 投机推理 | 仅用于 `infer/llm/**` |
| `multimodalDitCache` | DiT Cache | 仅用于 `infer/multimodal/**` |

中文枚举值见后续各章节的中文标签。

维护规则：

- 中文标签只用于前端展示、筛选面板和属性卡片，不写入每篇文档的 `static-meta`。
- 新增或修改枚举值时，必须同步更新本节中文标签、`utils/scripts/metadata_enums.py` 和 `utils/scripts/metadata_labels.py`。
- 如果枚举值本身是专有英文名词或行业通用缩写，中文标签保持英文；如果枚举值是通用概念，中文标签使用简洁中文。

## 4. 字段定义与使用

### 4.1 `sidebarTitle`

侧边栏和紧凑导航使用的中文短标题，不使用枚举。

- 必填，建议 4～16 个汉字或等效长度，最长不超过 24 个字符。
- 描述文章的核心技术点或优化目标，避免重复完整 H1。
- 原则上省略模型名、NPU、CANN、昇腾和硬件型号；无法区分时可保留必要术语。
- 避免“实践”“说明”“指南”“文档”等低信息量后缀。
- 同一目录下不得重复。

### 4.2 `quantization`

`quantization` 按文档实际使用、转换或优化的**权重数据类型**分类，不描述激活数据类型、缓存数据类型、校准算法或单独的中间计算精度。

| ID | 英文名称 | 中文名称 |
|---|---|---|
| `bf16` | BF16 | BF16 |
| `int8` | INT8 | INT8 |
| `int4` | INT4 | INT4 |
| `float8` | Float8 | Float8 |
| `fp8` | FP8 | FP8 |
| `mxfp8` | MXFP8 | MXFP8 |
| `hif8` | HiF8 | HiF8 |
| `mxfp4` | MXFP4 | MXFP4 |

- 仅在正文明确说明该类型用于模型权重、权重量化或权重检查点时填写；若未明确说明，省略 `quantization` 字段。
- 仅出现在激活、KV Cache、中间张量或示例代码中的精度不作为本字段取值。

### 4.3 `parallelism`

`parallelism` 按模型、序列或数据的**切分类型**分类，不描述并行度数值、Rank 数或集群规模。

| ID | 英文名称 | 中文名称 |
|---|---|---|
| `data-parallel` | Data Parallel | 数据并行 |
| `tensor-parallel` | Tensor Parallel | 张量并行 |
| `pipeline-parallel` | Pipeline Parallel | 流水并行 |
| `context-parallel` | Context Parallel | 上下文并行 |
| `sequence-parallel` | Sequence Parallel | 序列并行 |
| `expert-parallel` | Expert Parallel | 专家并行 |
| `zero` | ZeRO | ZeRO |

- 只填写正文中有明确实现、调优或性能分析的切分类型。
- 若正文未提供明确的并行切分信息，省略 `parallelism` 字段。

### 4.4 `operator`

`operator` 按文档使用的**算子开发后端**分类，例如 AscendC、TileLang、PyPTO 和 AutoFuse；不按 Attention、MatMul 等计算算子名称分类。

| ID | 英文名称 | 中文名称 |
|---|---|---|
| `ascendc` | AscendC | AscendC |
| `tilelang` | TileLang | TileLang |
| `pypto` | PyPTO | PyPTO |
| `autofuse` | AutoFuse | AutoFuse |

- 仅在正文包含对应方式的独立实现、性能分析或优化方案时填写。
- 若正文未提供明确的算子开发或优化方式，省略 `operator` 字段。
- Attention、MatMul 等计算算子名称不作为 `operator` 枚举值；如后续确有筛选需求，应新增独立字段，不与本字段混用。

### 4.5 `cannFeatures`

`cannFeatures` 仅使用以下固定原子特性总类别。新增或改名必须先修改本规范和 `utils/scripts/metadata_enums.py`，不得在文档中自行扩展。

| ID | 英文名称 | 中文名称 | 含义 |
|---|---|---|---|
| `multi-stream` | MultiStream | 多流 | 多流并行与通信计算重叠 |
| `superkernel` | SuperKernel | SuperKernel | SuperKernel 融合与执行优化 |
| `prefetch` | Prefetch | 预取 | 数据预取能力 |
| `npugraph` | NPUGraph | NPUGraph | NPU 图模式与图优化能力 |

- `multi-stream` 是 MultiStream 的规范 ID；不使用 `multistram`、`multi_stream` 等拼写变体。
- 普通技术文档只填写实际作为核心方案或优化点的特性，若正文未提供明确的 CANN 原子特性信息，省略 `cannFeatures` 字段。
- `cann_features/**` 文档必须填写且只能填写一个自身主题特性 ID；不要因为正文使用了多流、图模式等辅助机制就增加多个 ID。

### 4.6 `hardware`

`hardware` 表示报告已验证或明确适配的硬件，而非理论支持范围。

| ID | 英文名称 | 中文名称 |
|---|---|---|
| `atlas-a2` | Atlas A2 / Ascend 910B | Atlas A2 / Ascend 910B |
| `atlas-a3` | Atlas A3 | Atlas A3 |
| `ascend-950` | Ascend 950 | Ascend 950 |

若正文未提供明确的硬件适配或验证信息，省略 `hardware` 字段。

### 4.7 `frameworks`

`frameworks` 仅填写对方案实现有直接影响的框架或工具链。

| ID | 英文名称 | 中文名称 |
|---|---|---|
| `cann-recipes` | CANN Recipes | CANN Recipes |
| `torchtitan` | TorchTitan | TorchTitan |
| `megatron` | Megatron | Megatron |
| `mindspeed` | MindSpeed | MindSpeed |
| `verl` | verl | verl |
| `vllm` | vLLM | vLLM |
| `mindspore` | MindSpore | MindSpore |

- `frameworks` 仅限上层专用框架，不考虑底层基础框架；添加前考证其概念层级。
- 若正文未提供明确的框架或工具链信息，省略 `frameworks` 字段。

### 4.8 `cover`

```json
{"type": "auto"}
```

支持的类型：

| 类型 | 含义 | 附加字段 |
|---|---|---|
| `auto` | 自动选择封面。依次尝试文档第一张图片（Markdown 或 HTML 格式）、第一个 Mermaid 图表，均不存在时使用分类占位封面。 | 无 |
| `image` | 使用指定图片作为封面，不再自动选择文档中的其他图片或图表。 | 必须提供 `source`，填写相对于当前文档的图片路径或允许的 HTTPS URL。 |
| `mermaid` | 使用文档中的指定 Mermaid 图表作为封面。 | 必须提供从 `1` 开始的 `index`，表示第几个 Mermaid 图表。 |
| `placeholder` | 强制使用分类占位封面，不读取文档中的图片或 Mermaid 图表。 | 无 |

构建阶段必须验证指定图片或 Mermaid 图表存在且可用。

### 4.9 `llmSpeculativeInference`

`llmSpeculativeInference` 仅用于 `infer/llm/**` 文档，描述 LLM 推理文档是否涉及投机推理以及具体方案。

| ID | 英文名称 | 中文名称 | 含义 |
|---|---|---|---|
| `mtp` | MTP | MTP | Multi-Token Prediction 投机推理 |
| `dspark` | DSpark | DSpark | DSpark 投机推理方案 |
| `dflash` | DFlash | DFlash | DFlash 投机推理方案 |
| `none` | None | 未涉及投机推理 | 文档不涉及该类方案 |

规则：

- 只在 `infer/llm/**` 文档中填写。
- 如果文档同时对比或实现多个投机推理方案，可以多选，若没有明确说明投机推理方案则忽略此字段。

示例：

```json
"llmSpeculativeInference": ["mtp", "dflash"]
```

### 4.10 `multimodalDitCache`

`multimodalDitCache` 仅用于 `infer/multimodal/**` 文档，描述多模态生成或理解任务中 DiT 结构相关的缓存复用方案。取值与 `cann-recipes-infer` 启动 YAML 中 `dit_cache.method` 的方法名一致。

| ID | 英文名称 | 中文名称 | 含义 |
|---|---|---|---|
| `None` | None | None | 不启用 DiT Cache（基线） |
| `FBCache` | FBCache | FBCache | Full Block Cache：缓存整个 DiT block 输出，按 L1 距离阈值决定是否复用 |
| `TeaCache` | TeaCache | TeaCache | 用 timestep embedding 估计输出差异、多项式缩放校正的 timestep-aware 缓存 |
| `TaylorSeer` | TaylorSeer | TaylorSeer | 用泰勒展开预测当前步输出、一跳多步的缓存 |

规则：

- 只在 `infer/multimodal/**` 文档中填写；当前仓库目录使用 `multimodal`，不使用 `multimodel` 拼写。
- 只有正文明确提到某个 DiT Cache 方法时才填写对应 ID；多方法并存时多选。
- 正文只是模糊涉及缓存、没有明确提到任何方法时，填 `None`。

示例：

```json
"multimodalDitCache": ["FBCache", "TeaCache"]
```

## 5. 构建校验与前端使用

构建脚本必须校验字段名、类型、数组重复值、枚举值和封面引用。任一校验失败时停止发布，保留上一版可用站点。

| 页面或模块 | 使用字段 |
|---|---|
| 侧边栏与紧凑导航 | `sidebarTitle` |
| 分类筛选 | 场景范围仅包含 `Infer`、`Train`、`Embodie_AI`；技术属性包含 `quantization`、`parallelism`、`operator`、`cannFeatures`、`hardware`、`frameworks` |
| 报告页属性区 | `quantization`、`parallelism`、`operator`、`cannFeatures`、`hardware`、`frameworks` |
| 封面 | `cover` |

场景子类扩展字段仅在路径匹配时参与筛选和报告页属性展示。例如用户筛选 `infer/llm` 后，前端再展示 `llmSpeculativeInference`；用户筛选 `infer/multimodal` 后，前端再展示 `multimodalDitCache`。

## 6. 文档增加静态字段 Checklist

### 6.1 新增文档并生成 `static-meta`

- [ ] 根据内容选择 `infer/`、`train/`、`embodied_ai/` 或 `cann_features/` 下的正确目录。
- [ ] 阅读完整正文，确认 H1 和核心技术内容，不只根据文件名或目录推断字段值。
- [ ] 生成简短且同目录不重复的 `sidebarTitle`，长度不超过 24 个字符。
- [ ] 只填写正文有明确依据的非必填公共字段；没有明确信息时直接省略，不填 `none` 或空数组。
- [ ] 若文档位于 `cann_features/**`，只保留 `sidebarTitle` 与单值 `cannFeatures`；`cannFeatures` 必须代表文档自身主题，不能混入关联特性。
- [ ] 检查文档路径是否要求场景子类扩展字段，并按第 3.2 节和第 4.9～4.10 节填写。
- [ ] 根据需要设置 `cover`；未显式填写时按 `auto` 处理。
- [ ] 在技术文档末尾生成唯一的 `static-meta`，保证 JSON 使用双引号且没有尾逗号。

### 6.2 扩展已有字段：新增枚举值

如果只是扩展已经存在的字段，只需要在该字段现有的枚举类中增加枚举值，不需要新增字段或创建新的枚举类。

- [ ] 先在本文对应字段章节中补充机器 ID、英文名称、中文名称和使用边界。
- [ ] 在 `utils/scripts/metadata_enums.py` 中找到该字段现有的 `StrEnum`，增加枚举成员和枚举值。
- [ ] 枚举 `value` 使用稳定的小写机器 ID，枚举成员名使用大写下划线形式。
- [ ] 在 `utils/scripts/metadata_labels.py` 的 `VALUE_LABELS` 中增加相同机器 ID 的中英文名称。
- [ ] 检查使用该枚举的技术文档，按内容更新 `static-meta`。

### 6.3 新增字段：扩展字段枚举类

新增字段时，必须先扩展字段枚举类 `MetadataField`。如果新字段使用受控枚举值，还需要创建该字段自己的 `StrEnum` 枚举类，并接入标签和校验器。

- [ ] 确认字段是可复用的技术属性，不为单篇文档创建专用字段。
- [ ] 在第 3 章登记字段名、类型、是否必填、适用目录和用途，并在第 4 章增加字段定义与取值说明。
- [ ] 在 `utils/scripts/metadata_enums.py` 的 `MetadataField` 中增加字段枚举成员和字段名。
- [ ] 如果字段使用受控枚举值，在同一文件中新增对应的 `StrEnum` 枚举类；普通字符串或对象字段无需新增取值枚举类。
- [ ] 在 `utils/scripts/metadata_labels.py` 的 `FIELD_LABELS` 中增加字段名称；有枚举值时同步增加 `VALUE_LABELS`。
- [ ] 在 `utils/scripts/validate_metadata.py` 中导入新的枚举类，并增加类型、枚举和目录作用域校验。
- [ ] 如字段参与筛选或展示，同步更新内容索引构建和前端消费逻辑。
- [ ] 如果字段为必填，补齐适用范围内已有技术文档的 `static-meta`。

### 6.4 执行校验

- [ ] 在仓库根目录执行静态字段校验：

```powershell
python utils\scripts\validate_metadata.py
```

- [ ] 修改 Python 枚举、标签或校验器后执行语法检查：

```powershell
python -m py_compile utils\scripts\metadata_enums.py utils\scripts\metadata_labels.py utils\scripts\validate_metadata.py
```

- [ ] 确认校验结果没有未知字段、非法枚举、重复值、目录越界、重复 `sidebarTitle` 或无效封面引用。
- [ ] 如果修改会影响站点索引或筛选逻辑，重新执行内容构建并检查中英文展示。
