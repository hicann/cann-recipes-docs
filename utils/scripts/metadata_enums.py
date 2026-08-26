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

from enum import StrEnum


class MetadataField(StrEnum):
    """Allowed top-level fields inside a static-meta block."""

    SIDEBAR_TITLE = "sidebarTitle"
    QUANTIZATION = "quantization"
    PARALLELISM = "parallelism"
    OPERATOR = "operator"
    CANN_FEATURES = "cannFeatures"
    HARDWARE = "hardware"
    FRAMEWORKS = "frameworks"
    COVER = "cover"
    LLM_SPECULATIVE_INFERENCE = "llmSpeculativeInference"
    MULTIMODAL_DIT_CACHE = "multimodalDitCache"


class Quantization(StrEnum):
    """Weight data types and quantized weight formats."""

    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    FLOAT8 = "float8"
    FP8 = "fp8"
    MXFP8 = "mxfp8"
    HIF8 = "hif8"
    MXFP4 = "mxfp4"


class Parallelism(StrEnum):
    """Model, sequence, and data partitioning types."""

    DATA_PARALLEL = "data-parallel"
    TENSOR_PARALLEL = "tensor-parallel"
    PIPELINE_PARALLEL = "pipeline-parallel"
    CONTEXT_PARALLEL = "context-parallel"
    SEQUENCE_PARALLEL = "sequence-parallel"
    EXPERT_PARALLEL = "expert-parallel"
    ZERO = "zero"


class Operator(StrEnum):
    """Operator development and optimization approaches."""

    ASCENDC = "ascendc"
    TILELANG = "tilelang"
    PYPTO = "pypto"
    AUTOFUSE = "autofuse"


class CannFeature(StrEnum):
    """Fixed CANN atomic feature categories."""

    MULTI_STREAM = "multi-stream"
    SUPERKERNEL = "superkernel"
    PREFETCH = "prefetch"
    NPUGRAPH = "npugraph"


class Hardware(StrEnum):
    """Hardware platforms covered by the metadata specification."""

    ATLAS_A2 = "atlas-a2"
    ATLAS_A3 = "atlas-a3"
    ATLAS_310P = "atlas-310p"
    ASCEND_950 = "ascend-950"


class Framework(StrEnum):
    """Frameworks and toolchains directly relevant to a report."""

    CANN_RECIPES = "cann-recipes"
    MINDSPORE = "mindspore"
    TORCHTITAN = "torchtitan"
    MEGATRON = "megatron"
    MINDSPEED = "mindspeed"
    VERL = "verl"
    VLLM = "vllm"


class CoverType(StrEnum):
    AUTO = "auto"
    IMAGE = "image"
    MERMAID = "mermaid"
    PLACEHOLDER = "placeholder"


class LlmSpeculativeInference(StrEnum):
    """Speculative inference approaches for infer/llm reports."""

    MTP = "mtp"
    DSPARK = "dspark"
    DFLASH = "dflash"
    NONE = "none"


class MultimodalDitCache(StrEnum):
    """DiT cache methods for infer/multimodal reports."""

    NO_CACHE = "None"
    FB_CACHE = "FBCache"
    TEACACHE = "TeaCache"
    TAYLOR_SEER = "TaylorSeer"


def enum_values(enum_class: type[StrEnum]) -> set[str]:
    """Return the serialized values of a metadata enum."""

    return {item.value for item in enum_class}
