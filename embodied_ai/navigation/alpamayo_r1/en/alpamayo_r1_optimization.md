# Alpamayo-R1 VLA Model Inference: Ascend Adaptation and Performance Optimization Guide

This document summarizes the optimization strategies for the Alpamayo-R1 model on the Ascend NPU:

## 1. Flash Attention (FA) Replacement

### Optimization Description
This sample uses the npu_fusion_attention fused operator built into torch_npu to replace the small-operator implementation in the source code. For details on npu_fusion_attention, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/en/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_fusion_attention.md).

### Implementation
Setting the following parameter in the model configuration calls the npu_fusion_attention fused operator built into torch_npu:

```python
config.attn_implementation = 'flash_attention_2'
```

### Optimization Location
- File: [base_model.py](https://github.com/NVlabs/alpamayo/blob/e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073/src/alpamayo_r1/models/base_model.py)
- Modification: set the attention implementation method at model initialization

---

## 2. Boolean Indexing Optimization in the Transformers Library

### Optimization Description
The original implementation uses boolean indexing operations, which are inefficient. By optimizing the indexing computation logic, unnecessary boolean operations and type conversions are avoided.

### Implementation

**Original implementation:**
```python
if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
```

**Optimized implementation:**
An independent `patched_get_placeholder_mask` function is created to optimize the mask generation logic for image and video tokens:
```python
if image_features is not None:
    n_masked_elements = special_image_mask.sum().item()
    if n_masked_elements != image_features.numel():
        raise ValueError(
            f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.shape[0]}"
        )
```

### Optimization Location
- New file: [qwen3vl_optimization.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/navigation/alpamayo-r1/patches/ar-1.patch)
- New: the `patched_get_placeholder_mask` function
- New: the `apply_qwen3vl_patches()` function used to apply the patch

---

## 3. RMSNorm Operator Optimization

### Optimization Description
This sample uses the npu_rms_norm fused operator built into torch_npu to replace the small-operator implementation in the source code. For details on npu_rms_norm, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/en/Pytorch/710/apiref/torchnpuCustomsapi/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md). The `npu_rms_norm` fused operator is enabled in the `RmsNorm` class in [action_in_proj.py](https://github.com/NVlabs/alpamayo/blob/e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073/src/alpamayo_r1/models/action_in_proj.py):

### Implementation

**Original implementation:**
```python
def _norm(self, x):
    """Normalize the input tensor."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

def forward(self, x):
    """Normalize the input tensor."""
    output = self._norm(x.float()).type_as(x)
    return output * self.weight
```

**Optimized implementation:**
```python
def forward(self, x):
    input_dtype = x.dtype
    weight_dtype = self.weight.dtype

    if input_dtype != weight_dtype:
        weight_dtype = self.weight.to(input_dtype)
    else:
        weight = self.weight

    # npu_rms_norm returns
    output = torch_npu.npu_rms_norm(x, weight, self.eps)[0]
    return output
```

### Optimization Location
- File: `src/alpamayo_r1/models/action_in_proj.py`
- Class: `RmsNorm`
- Modification: replace the original implementation with `torch_npu.npu_rms_norm`

---

## 4. ADD Operator Optimization: Using the Fused Operator torch.addcmul

### Optimization Description
During the integration process of the diffusion model, an operation such as `x = x + dt * v` is encountered. Using the fused operator `torch.addcmul` provided by the Ascend NPU, the addition and multiplication can be fused into a single operator, reducing computation steps and memory access.

### Implementation

**Original implementation:**
```python
x = x + dt * v
```

**Optimized implementation:**
```python
x = torch.addcmul(x, dt, v)
```

### Optimization Location
- File: [flow_matching.py](https://github.com/NVlabs/alpamayo/blob/e0e2ac32ebd9c3ad8a314099aa5af5c3aaa28073/src/alpamayo_r1/diffusion/flow_matching.py)
- Function: the `flow_matching_euler` integration step
- Modification: replace the original addition and multiplication combination with `torch.addcmul`

---

## 5. Concat Optimization: Pre-allocation + Index Assignment

### Optimization Description
When handling the concatenation of multiple features, the original implementation uses `torch.cat` for multiple concatenations, which causes multiple memory allocations and data copies. The optimization is to pre-allocate a tensor of the complete size and then assign values directly through indexing, avoiding the concatenation operation.

### Implementation

**Original implementation:**
```python
action_feats = torch.cat([s(x[:, :, i]) for i, s in enumerate(self.sinus)], dim=-1)
timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
timestep_feats = timestep_feats.repeat(1, T, 1)
x = torch.cat((action_feats, timestep_feats), dim=-1)
```

**Optimized implementation:**
```python
batch_feats = torch.empty(B, T, self.num_input_feats, device=x.device, dtype=x.dtype)
offset = 0
for i, s in enumerate(self.sinus):
    feat = s(x[:, :, i])
    batch_feats[:, :, offset:offset+feat.size(-1)] = feat
    offset += feat.size(-1)
timestep_feats = self.timestep_fourier_encoder(timesteps[..., -1])
batch_feats[:, :, offset:] = timestep_feats.repeat(1, T, 1)
x = batch_feats
```

### Optimization Location
- File: `src/alpamayo_r1/models/action_in_proj.py`
- Class: `PerWaypointActionInProjV2`
- Method: `forward`
- Modification: pre-allocate `batch_feats` and use index assignment instead of `torch.cat`

---

## Optimization Summary

| Optimization item | Optimization type | Main technique | Performance benefit |
|--------|----------|----------|----------|
| FA replacement | Attention optimization | torch_npu.npu_fusion_attention | Reduces memory usage, improves speed, supports longer sequences |
| Index operator replacement | Operator optimization | Optimized boolean indexing logic | Reduces redundant computation, improves indexing efficiency |
| RMSNorm optimization | Operator fusion | torch_npu.npu_rms_norm | Reduces kernel launches, optimizes memory access |
| ADD operator optimization | Operator fusion | torch.addcmul | Reduces memory read/write, reduces kernel launches |
| Slice+Concat optimization | Memory optimization | Pre-allocation + index assignment | Reduces memory allocation and copying, improves cache efficiency |

<!-- static-meta
{
  "sidebarTitle": "Inference Optimization",
  "hardware": ["atlas-a2"],
  "cover": {"type": "auto"}
}
-->
