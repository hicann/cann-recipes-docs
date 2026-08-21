# Qwen Dense Inference Performance Optimization on Atlas A2/A3
## Overview
This document primarily describes NPU inference optimization strategies for Qwen2/Qwen3 Dense (non-MoE) models. It implements BF16 inference on Atlas A2/A3 training/inference series products.

## Tensor Parallel (TP) Optimization
### Attention TP Optimization
#### Partitioning Strategy
The tensor partitioning strategy for Attention can be divided into QKV head partitioning and linear layer partitioning.

When QKV heads are partitioned, the multi-head computation mechanism of attention makes tensor partitioning convenient. Each head is computed independently first, and the results are then concatenated. Assume that the attention layer of the model needs to partition `num_heads` query heads by the partition count `attn_tp_size`. `num_heads` must be divisible by `attn_tp_size`, and the number of query heads placed on each card is `num_heads_per_rank = num_heads // attn_tp_size`. The numbers of key and value heads are equal and may be less than or equal to the number of query heads (they are smaller in GQA scenarios). To ensure that at least one key and one value head are placed on each card, calculate the number of key or value heads placed on each card as follows:
`num_key_value_heads_per_rank = max(num_key_value_heads // attn_tp_size, 1)`.

For the `o_proj` linear layer, use row partitioning.

#### Computation Decomposition
This optimization strategy first merges the Q, K, and V linear-layer computations into one Matmul computation (`merged_qkv_proj`) to improve computation performance. After splitting the output of `merged_qkv_proj` into Q, K, and V, normalize Q and K (Qwen3 enables QK-Norm, while Qwen2 does not perform this step), apply rotary position encoding, perform attention computation, and finally generate the output through the o_proj layer.

### MLP TP Optimization
#### Partitioning Strategy
Apply column partitioning to `gate_proj` and `up_proj` of the MLP layer and row partitioning to `down_proj`. Also merge the `gate_proj` and `up_proj` linear-layer computations into `gate_up_proj`.

#### Computation Decomposition
The MLP layer contains three matmul operations: gate_proj, up_proj, and down_proj. The specific computation is x = down( SiLU(gate(x)) * up(x) ). This optimization merges the tensor-partitioned gate_proj and up_proj into one gate_up_proj Matmul computation. It splits the output into two blocks along the intermediate dimension, performs SiLU activation and element-wise multiplication on them, and finally generates the output through down_proj.

Merging gate_proj and up_proj reduces the number of kernel launches and improves overall computation efficiency.

## Residual Fusion Optimization
In the original transformers implementation, the residual connection and RMSNorm of each DecoderLayer run separately:
```
residual = hidden_states
hidden_states = RMSNorm(hidden_states)
hidden_states = Attention(hidden_states)
hidden_states = residual + hidden_states        ← independent residual add
```

This optimization enables the [torch_npu.npu_add_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000140.html)(Chinese) fused operator to merge residual addition and RMSNorm into one kernel call:
```
hidden_states, residual = npu_add_rms_norm(residual, hidden_states, weight, eps)
```

Each DecoderLayer contains two residual connections, before Attention and before MLP. After fusion, each layer eliminates two independent tensor addition operators and their memory read and write overhead. The residual is passed between layers as a parameter and is no longer independently added inside each layer.

## Enable Fused Operators
### RmsNorm Operator Optimization
Enable the [torch_npu.npu_rms_norm](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000140.html)(Chinese) operator to replace the original manually implemented variance + rsqrt computation and improve model inference performance.

### RoPE Operator Optimization
Enable the [torch_npu.npu_apply_rotary_pos_emb](https://www.hiascend.com/document/detail/zh/Pytorch/60RC2/apiref/apilist/ptaoplist_000140.html)(Chinese) fused operator to replace the original `rotate_half` + manual multiplication and addition implementation. Also precompute and cache cos/sin to avoid dynamic computation in each forward operation.

### FlashAttention Fused Operator Optimization
Enable the [torch.ops.npu.npu_fused_infer_attention_score_v2](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00086.html)(Chinese) FlashAttention fused operator for inference scenarios, which supports both full computation and incremental computation.

## Enable Graph Mode
Using a static graph improves inference performance. Override the `graph_compile` function of `ModelRunner` in [`model_runner.py`](https://gitcode.com/cann/cann-recipes-infer/blob/master/executor/model_runner.py) to compile the model into a static graph.

## W8A8 INT8 Quantization

Qwen3-8B supports W8A8 INT8 quantization, with per-channel static quantization for weights and per-token dynamic quantization for activations. Quantization covers four Linear layers: Attention `merged_qkv_proj` / `o_proj` and MLP `gate_up_proj` / `down_proj`.

After quantization is enabled, MLP computation switches to a dedicated fused-operator path. `gate_up_proj` outputs an int32 tensor and per-token scale, which are passed to [torch_npu.npu_dequant_swiglu_quant](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_dequant_swiglu_quant.md)(Chinese). The operator performs dequantization, SwiGLU activation, and requantization in one operation. The output int8 tensor and corresponding scale are directly received by `down_proj` for subsequent matrix multiplication. This implementation enables the ge_graph static graph mode to use an existing fused operator and avoids the fallback path caused by unavailable composite operators such as `AddRmsNormDynamicQuant`.

Quantized model weights are loaded in compressed-tensors format. The `mm_quant_mode` field controls the quantization logic. The W8A8 quantization path is enabled only when the field value is `w8a8int8`; otherwise, the original BF16 inference path remains unchanged.

## Enable AIV Expansion for Collective Communication
Use the device's Vector Cores to accelerate collective communication operations. For details, see the [HCCL_OP_EXPANSION_MODE environment variable](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/maintenref/envvar/envref_07_0096.html)(Chinese):

```shell
export HCCL_OP_EXPANSION_MODE=AIV
```

## Appendix
[Environment Deployment and Sample Execution](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/qwen/README.md)(Chinese)

<!-- static-meta
{
  "sidebarTitle": "Inference Optimization",
  "quantization": ["bf16", "int8"],
  "parallelism": ["tensor-parallel"],
  "hardware": ["atlas-a2", "atlas-a3"],
  "frameworks": ["cann-recipes"],
  "llmSpeculativeInference": ["none"]
}
-->
