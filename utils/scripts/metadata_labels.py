# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LABEL_LANGUAGES = ("en", "zh")


FIELD_LABELS = {
    "sidebarTitle": {"en": "Sidebar title", "zh": "侧栏标题"},
    "quantization": {"en": "Quantization", "zh": "量化"},
    "parallelism": {"en": "Parallelism", "zh": "并行切分"},
    "operator": {"en": "Operator", "zh": "算子种类"},
    "cannFeatures": {"en": "CANN Features", "zh": "CANN 特性"},
    "hardware": {"en": "Hardware", "zh": "硬件"},
    "frameworks": {"en": "Frameworks", "zh": "框架"},
    "cover": {"en": "Cover", "zh": "封面"},
    "llmSpeculativeInference": {"en": "Speculative inference", "zh": "投机推理"},
    "multimodalDitCache": {"en": "DiT Cache", "zh": "DiT Cache"},
}


VALUE_LABELS = {
    "quantization": {
        "bf16": {"en": "BF16", "zh": "BF16"},
        "int8": {"en": "INT8", "zh": "INT8"},
        "int4": {"en": "INT4", "zh": "INT4"},
        "float8": {"en": "Float8", "zh": "Float8"},
        "fp8": {"en": "FP8", "zh": "FP8"},
        "mxfp8": {"en": "MXFP8", "zh": "MXFP8"},
        "hif8": {"en": "HiF8", "zh": "HiF8"},
        "mxfp4": {"en": "MXFP4", "zh": "MXFP4"},
    },
    "parallelism": {
        "data-parallel": {"en": "Data Parallel", "zh": "数据并行"},
        "tensor-parallel": {"en": "Tensor Parallel", "zh": "张量并行"},
        "pipeline-parallel": {"en": "Pipeline Parallel", "zh": "流水并行"},
        "context-parallel": {"en": "Context Parallel", "zh": "上下文并行"},
        "sequence-parallel": {"en": "Sequence Parallel", "zh": "序列并行"},
        "expert-parallel": {"en": "Expert Parallel", "zh": "专家并行"},
        "zero": {"en": "ZeRO", "zh": "ZeRO"},
    },
    "operator": {
        "ascendc": {"en": "AscendC", "zh": "AscendC"},
        "tilelang": {"en": "TileLang", "zh": "TileLang"},
        "pypto": {"en": "PyPTO", "zh": "PyPTO"},
        "autofuse": {"en": "AutoFuse", "zh": "AutoFuse"},
    },
    "cannFeatures": {
        "multi-stream": {"en": "MultiStream", "zh": "多流"},
        "superkernel": {"en": "SuperKernel", "zh": "SuperKernel"},
        "prefetch": {"en": "Prefetch", "zh": "预取"},
        "npugraph": {"en": "NPUGraph", "zh": "NPUGraph"},
    },
    "hardware": {
        "atlas-a2": {"en": "Atlas A2 / Ascend 910B", "zh": "Atlas A2 / Ascend 910B"},
        "atlas-a3": {"en": "Atlas A3", "zh": "Atlas A3"},
        "ascend-950": {"en": "Ascend 950", "zh": "Ascend 950"},
    },
    "frameworks": {
        "cann-recipes": {"en": "CANN Recipes", "zh": "CANN Recipes"},
        "mindspore": {"en": "MindSpore", "zh": "MindSpore"},
        "torchtitan": {"en": "TorchTitan", "zh": "TorchTitan"},
        "megatron": {"en": "Megatron", "zh": "Megatron"},
        "onnx": {"en": "ONNX", "zh": "ONNX"},
        "atb": {"en": "ATB", "zh": "ATB"},
    },
    "cover.type": {
        "auto": {"en": "Auto cover", "zh": "自动封面"},
        "image": {"en": "Image cover", "zh": "图片封面"},
        "mermaid": {"en": "Mermaid cover", "zh": "Mermaid 封面"},
        "placeholder": {"en": "Placeholder cover", "zh": "占位封面"},
    },
    "llmSpeculativeInference": {
        "mtp": {"en": "MTP", "zh": "MTP"},
        "dspark": {"en": "DSpark", "zh": "DSpark"},
        "dflash": {"en": "DFlash", "zh": "DFlash"},
        "none": {"en": "None", "zh": "未涉及投机推理"},
    },
    "multimodalDitCache": {
        "dit-block-cache": {"en": "DiT Block Cache", "zh": "DiT Block Cache"},
        "attention-cache": {"en": "Attention Cache", "zh": "Attention Cache"},
        "feature-cache": {"en": "Feature Cache", "zh": "Feature Cache"},
        "teacache": {"en": "TeaCache", "zh": "TeaCache"},
        "magcache": {"en": "MagCache", "zh": "MagCache"},
        "none": {"en": "None", "zh": "未涉及 DiT Cache"},
    },
}


def metadata_label_manifest() -> dict[str, dict]:
    """Return labels in the JSON shape consumed by the static site."""

    return {
        "fields": FIELD_LABELS,
        "values": VALUE_LABELS,
    }
