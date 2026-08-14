# Decode-Stage Inference Performance Optimization for DeepSeek-R1 on an Atlas A3 Cluster
## Overview
This document describes parallel deployment and performance optimization strategies for the decode stage of DeepSeek-R1 on Atlas A3 series products. It explains how to perform large-scale Expert Parallel (EP) inference on Atlas A3 series products and introduces other parallel techniques and optimization solutions.

## Performance Optimization
### General Optimization
The non-MoE part of DeepSeek-R1 has a structure similar to Llama. For general optimizations, refer to the changes in [Llama](https://gitcode.com/Ascend/torchair/tree/7.3.0/npu_tuned_model/llm/llama), such as fixing the KV Cache size, optimizing cos/sin, fusing AddRMSNorm, and fully optimizing the LM Head computation workload.

### MLA (Multi-Head Latent Attention) Low-Rank Compression Optimization
#### Enabling Fused Operators
The low-rank compression method described in the [DeepSeek paper](https://arxiv.org/pdf/2405.04434) reduces the memory occupied by the KV cache and improves inference efficiency. After MLA low-rank compression is implemented, enable fused kernels to optimize performance. For the related implementation, refer to the [`forward_page_attention_mla_prolog`](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/deepseek_r1/models/modeling_deepseek.py) function in the `DeepseekV3Attention` class.

- MLA prolog computation performance optimization: Enable the [npu_mla_prolog_v3](https://www.hiascend.com/document/detail/zh/Pytorch/2600/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_mla_prolog_v3.md)(Chinese) fused kernel to replace computations before attention. These computations include the Q, K, and V linear layers, rotary position encoding (ROPE), RmsNorm, and KV Cache updates.
- Attention performance optimization: Enable the [npu_fused_infer_attention_score](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_fused_infer_attention_score.md)(Chinese) fused kernel to accelerate MLA computation.

### Implementing Expert Parallel (EP) and Enabling Fused Operators for the MoE Module

If the original MoE implementation is used in the MoE module, a for loop processes `expert_num` FFN experts, which is inefficient. CANN provides a series of fused operators for token routing, expert computation, and other operations in the MoE layer to improve computation efficiency.

- Router computation optimization: Use the [torch_npu.npu_moe_gating_top_k](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_gating_top_k.md)(Chinese) operator to sort the router computation results and select the top-k experts.
- High-performance expert computation: Use the [torch_npu.npu_grouped_matmul](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_grouped_matmul.md)(Chinese) operator to process the computations of multiple experts simultaneously, improving computation and data movement efficiency.
- High-performance communication routing across multiple cards: Enable the [torch_npu.npu_moe_distribute_dispatch_v2](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_distribute_dispatch_v2.md)(Chinese) and [torch_npu.npu_moe_distribute_combine_v2](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_moe_distribute_combine_v2.md)(Chinese) operators to implement communication across multiple cards under EP. Before using these operators, refer to their documentation to check whether environment variables such as `HCCL_BUFFSIZE` are properly configured and to understand their use cases and constraints.

### Merging MLP Linear Layer Computations

The original `DeepseekV3MLP` implementation contains three matmul operations: `gate_proj`, `up_proj`, and `down_proj`. Merge the `gate_proj` and `up_proj` computations to obtain `gate_up_proj` and improve overall computation efficiency.

### TorchAir Graph Mode

The decode scenario generally has strict latency requirements. In PyTorch `eager_mode`, operators may encounter a severe host delivery bottleneck. Enable graph mode through torch.compile to optimize performance. TorchAir (Torch Ascend Intermediate Representation) extends the graph mode capabilities of Ascend Extension for PyTorch (torch_npu). It provides an Ascend device-friendly torch.compile graph mode backend to accelerate and optimize PyTorch network inference in graph mode on Ascend NPUs.

TorchAir provides the max-autotune and reduce-overhead modes. For details, refer to the [documentation](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00003.html)(Chinese). This sample supports both modes in the decode scenario. Enable a mode by configuring `exe_mode` in the YAML file. `ge_graph` corresponds to max-autotune, and `acl_graph` corresponds to reduce-overhead.

### Supporting Multi-Token Prediction (MTP)
The sample implements MTP speculative inference based on the MTP method described in the DeepSeek paper. When the scenario has not reached the computation bottleneck, MTP computation can provide effective inference acceleration. Set the `next_n` parameter to enable MTP.


### Enabling AIV Expansion for Collective Communication
Use the Vector Core compute units on the device to accelerate communication operations. For details, refer to the [HCCL_OP_EXPANSION_MODE environment variable](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0096.html)(Chinese):

```shell
export HCCL_OP_EXPANSION_MODE=AIV
```

### SuperKernel Optimization
When `ge_graph` graph mode is enabled for decode, the sample supports SuperKernel optimization of the model computation graph within a user-defined scope. For details about SuperKernel, refer to the [official documentation](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00035.html)(Chinese). In this sample, set `enable_superkernel` to enable the feature. The sample can optimize all decode layers within one SuperKernel scope, reducing task scheduling wait time and scheduling overhead to improve overall performance. To change the SuperKernel scope, modify [`modeling_deepseek.py`](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/deepseek_r1/models/modeling_deepseek.py) as follows:
```python
from executor.utils import superkernel_scope

switch = enable_superkernel
scope = f"scope_name"
options = f"scope_options"
with superkernel_scope(switch, scope, options):
    your modules
```
Here, `scope` indicates the SuperKernel name of the current fusion scope, and `options` indicates the custom SuperKernel compilation options. For details about the descriptions and usage scopes of `scope` and `options`, refer to the [SuperKernel documentation](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00035.html)(Chinese).

## Benchmark

This practice benchmarks the performance of the DeepSeek-R1 W8A8C8 quantized version on Atlas A3.
|Quant Mode| Global Batch Size | Seq Length | Chips | TPOT (ms) | Throughput (tokens/p/s) |
|-------| ----------------- | ---------- | ----- | --------- | ----------------------- |
|W8A8C8 |    6144           | 4096       | 64   | 44.9      |2138                     |

> Note: The performance data is collected with the MTP1 and perfect eplb configurations. On average, 0.7 tokens are accepted from one draft token.
---
## Appendix
[Environment Deployment and Sample Execution](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/deepseek_r1/README.md)(Chinese)

<!-- static-meta
{
  "sidebarTitle": "Decode Optimization",
  "quantization": ["int8"],
  "parallelism": ["expert-parallel"],
  "cannFeatures": ["superkernel"],
  "hardware": ["atlas-a3"],
  "frameworks": ["cann-recipes"],
  "llmSpeculativeInference": ["mtp"]
}
-->
