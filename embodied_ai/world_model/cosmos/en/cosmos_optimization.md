# Cosmos Ascend NPU Multi-Card Parallel Optimization Guide

## 1. Optimization Overview

This optimization systematically enhances the multi-card parallel inference capability of the Cosmos family of world foundation models on the Ascend NPU platform, mainly covering two models:
- **Cosmos-Transfer2.5-2B**: video style transfer multi-control network model
- **Cosmos-Predict2.5-2B**: video generation world foundation model

The optimization focuses on **enabling multi-card parallelism**, including CFG (Classifier-Free Guidance) parallelism, context parallelism, and NPU device management, to achieve distributed, efficient inference in a multi-card Ascend environment.

In addition, targeted optimizations were also performed for NPU characteristics, including Flash Attention replacement, RMSNorm fused operator adaptation, and Rotary position encoding optimization.

---

## 2. Enabling Multi-Card Parallelism

### 2.1 Multi-Card Parallelism of Cosmos on NPU
The current Cosmos-Predict2.5 and Cosmos-Transfer2.5 can perform multi-card parallel inference normally on the NPU by running the [`npu_adapt.sh`](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/world_model/cosmos-transfer2.5/npu_adapt.sh) script.


### 2.2 CFG Parallelism Fix

Cosmos-Transfer2.5 natively supports video-to-video style transfer with multiple control modalities (depth maps, semantic segmentation, edge detection, etc.). To improve large-scale inference efficiency, the following parallel strategies need to be implemented:

1. **CFG parallelism (Classifier-Free Guidance Parallelism)**: splits the NPUs into two groups to handle the conditional and unconditional denoising tasks separately, improving scalability for large-scale clusters
2. **Context parallelism**: allocates long-sequence video frames across devices, supporting ultra-long video generation

### 2.3 Core Modifications

#### 2.3.1 Configuration Layer: Modify [config.py](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/8b0e6af4b3bed40408c5762e528cb4e2a233f278/cosmos_transfer2/config.py)

Add new parallel control parameters to the `SetupArguments` data class:

```python
# Add a new parameter to the SetupArguments data class
enable_cfg_parallel: bool = False
"""Enable Classifier-Free Guidance parallelism for better scaling across more NPUs. 
Splits NPUs into two groups for conditional/unconditional denoising."""
```

#### 2.3.2 Inference Layer: Refactor the `Control2WorldInference.__init__` Method

**File modified**: [inference.py](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/8b0e6af4b3bed40408c5762e528cb4e2a233f278/cosmos_transfer2/inference.py)

**Patch file**: [`inference_patch.py`](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/world_model/cosmos-transfer2.5/adaptor_patches/inference_patch.py)

**Key code changes**:

```python
# Original code (official version)
self.device_rank = 0
process_group = None
if args.context_parallel_size > 1:
    from megatron.core import parallel_state
    distributed.init()
    parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
    process_group = parallel_state.get_context_parallel_group()

# Optimized code (Ascend-adapted version)
self.device_rank = 0
cfg_parallel = args.enable_cfg_parallel  # Added: read the CFG parallelism flag
process_group = None
if args.context_parallel_size > 1:
    from megatron.core import parallel_state
    
    distributed.init()
    
    # Decide the context parallelism scale based on cfg_parallel
    if cfg_parallel:
        # CFG parallelism mode: split the total number of cards in half, half for condition and half for unconditional
        parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size // 2)
    else:
        # Standard mode: use all cards for context parallelism
        parallel_state.initialize_model_parallel(context_parallel_size=args.context_parallel_size)
    
    process_group = parallel_state.get_context_parallel_group()
```

**Logic description**:
1. **CFG parallelism mode** (`enable_cfg_parallel=True`):
   - Assuming a total of 8 cards, `context_parallel_size=4`
   - 4 cards process the conditional denoising branch, and 4 cards process the unconditional denoising branch
   

2. **Standard parallelism mode** (`enable_cfg_parallel=False`):
   - All 8 cards are used for context parallelism


3. **Passing the cfg_parallel flag**:
   ```python
   self.inference_pipeline = ControlVideo2WorldInference(
       ...
       cfg_parallel=cfg_parallel,  # Passed to the downstream pipeline
   )
   ```
---

## 3. NPU Operator Performance Optimization

### 3.1 Flash Attention (FA) Replacement

#### 3.1.1 Optimization Description

Use the `npu_fusion_attention` fused operator in torch_npu to replace the FlashAttention operator implementation in the source code. For detailed instructions on `npu_fusion_attention`, see the [Ascend community documentation](https://www.hiascend.com/document/detail/en/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_fusion_attention.md).

#### 3.1.2 Implementation

(1) In Cosmos-Predict2.5-2B, the `torch_npu` interface call method is used:
```python
attn_output_bnsd = torch_npu.npu_fusion_attention(
    query_bnsd, key_bnsd, value_bnsd, head_num, input_layout="BNSD", 
    pse=None,
    atten_mask=self.atten_mask_npu,
    scale=scale,
    pre_tokens=2147483647,
    next_tokens=2147483647,
    keep_prob=1,
    sparse_mode=2
)[0]
```
(2) In Cosmos-Transfer2.5-2B, the native SDPA interface call is used:
```python
attn_output_bnsd = F.scaled_dot_product_attention(
    query_bnsd, 
    key_bnsd, 
    value_bnsd, 
    attn_mask=None, 
    dropout_p=0.0,
    is_causal=True
)
```

#### 3.1.3 Optimization Location

- **Files**:
  - [cosmos-predict2.5/cosmos_predict2/_src/reason1/networks/qwen2_5_vl.py](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/173b0fe980ab572f17c13258b775271f897b90ed/cosmos_predict2/_src/reason1/networks/qwen2_5_vl.py)
  - [cosmos-transfer2.5/cosmos_transfer2/_src/reason1/networks/qwen2_5_vl.py](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/8b0e6af4b3bed40408c5762e528cb4e2a233f278/cosmos_transfer2/_src/reason1/networks/qwen2_5_vl.py)


### 3.2 RMSNorm Operator Optimization

#### 3.2.1 Optimization Description

Use the built-in `npu_rms_norm` fused operator in torch_npu to replace the custom implementation in the source code. For detailed instructions on `npu_rms_norm`, see the [Ascend design documentation](https://www.hiascend.com/document/detail/en/Pytorch/710/apiref/torchnpuCustomsapi/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md).

#### 3.2.2 Implementation

(1) Original implementation:
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```
(2) Optimized implementation:
```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = torch_npu.npu_rms_norm(x, self.weight.float(), epsilon=self.eps)[0]
        return output
```

#### 3.2.3 Optimization Location

- **Files**:
  - [cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/173b0fe980ab572f17c13258b775271f897b90ed/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py)
  - [cosmos-transfer2.5/cosmos_transfer2/_src/predict2/networks/minimal_v4_dit.py](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/8b0e6af4b3bed40408c5762e528cb4e2a233f278/cosmos_transfer2/_src/predict2/networks/minimal_v4_dit.py)

### 3.3 Rotary Fused Operator Adaptation

#### 3.3.1 Optimization Description

Use the built-in `npu_rotary_mul` fused operator in torch_npu to replace the `apply_rotary_pos_emb` imported from `transformer_engine` in the source code. For detailed instructions on `npu_rotary_mul`, see the [Ascend design documentation](https://www.hiascend.com/document/detail/en/Pytorch/730/ptmoddevg/trainingmigrguide/performance_tuning_0030.html).

#### 3.3.2 Implementation

```python
def apply_rotary_pos_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    radians = freqs.transpose(0, 1) 
    cos = torch.cos(radians) 
    sin = torch.sin(radians) 
    res_rot = torch_npu.npu_rotary_mul(x, cos, sin)
    return res_rot
```

#### 3.3.3 Optimization Location

- **Files**:
  - [cosmos-predict2.5/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/173b0fe980ab572f17c13258b775271f897b90ed/cosmos_predict2/_src/predict2/networks/minimal_v4_dit.py)
  - [cosmos-transfer2.5/cosmos_transfer2/_src/predict2/networks/minimal_v4_dit.py](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/8b0e6af4b3bed40408c5762e528cb4e2a233f278/cosmos_transfer2/_src/predict2/networks/minimal_v4_dit.py)

## 4. Summary

This optimization successfully implements the multi-card parallel inference capability and optimization of the Cosmos family of models on the Ascend NPU platform:

### 4.1 Multi-Card Parallel Optimization

1. **Cosmos-Transfer2.5**:
   - Adds the `enable_cfg_parallel` parameter, supporting flexible combinations of CFG parallelism and context parallelism
   - Dynamically modifies the initialization logic through `inference_patch.py`, without intrusive source code modifications

2. **Cosmos-Predict2.5**:
   - Dynamically applies NPU adaptation patches through the monkey patch mechanism

### 4.2 General Features

- Supports multi-card distributed inference launched by `torchrun`
- Flexible parallel strategy configuration

### 4.3 Fused Operator Optimization

- **Flash Attention**: uses `npu_fusion_attention` to replace the standard Flash Attention
- **RMSNorm**: uses the `npu_rms_norm` fused operator to improve normalization performance
- **Rotary position encoding**: uses `npu_rotary_mul` to accelerate rotary position encoding computation

<!-- static-meta
{
  "sidebarTitle": "Multi-Card Parallel",
  "hardware": ["atlas-a3"],
  "parallelism": ["context-parallel"],
  "frameworks": ["megatron"],
  "cover": {"type": "auto"}
}
-->
