# 贡献指南

本项目欢迎开发者参与技术文档建设和改进。社区行为准则、CLA 协议和通用贡献流程请参见 [cann-community](https://gitcode.com/cann/community)。

## 第一步：新增或修改文档

根据文档内容选择对应目录，例如：

- 推理文档放在 `infer/` 下。
- 训练文档放在 `train/` 下。
- 具身智能文档放在 `embodied_ai/` 下。
- CANN 原子特性文档放在 `cann_features/` 下。

## 第二步：增加静态字段

参考 [static_meta_infos.md](utils/references/static_meta_infos.md) 和 [新增文档 Skill](skill/new_document_skill.md)，在技术文档末尾补充 `static-meta`。

将中文技术文档翻译为英文时，参考 [英文文档翻译 Skill](skill/translate_english_document_skill.md)，确保正文无增删、链接优先替换为对应英文页面，并保持中英文 `static-meta` 标签一致。

静态字段的意义是把文档内容转成稳定、可校验、可复用的结构化信息，便于后续做分类筛选、侧栏短标题、封面选择和内容索引。

当前公共字段：

- `sidebarTitle`
- `quantization`
- `parallelism`
- `operator`
- `cannFeatures`
- `hardware`
- `frameworks`
- `cover`

当前增量应用场景字段：

- `llmSpeculativeInference`：仅用于 `infer/llm/**`
- `multimodalDitCache`：仅用于 `infer/multimodal/**`

## 第三步：校验静态字段

修改文档静态字段后，在仓库根目录运行：

```powershell
python utils\scripts\validate_metadata.py
```
上传代码前，必须保证校验通过。