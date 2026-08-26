# daVinci-MagiHuman Model Inference Optimization on NPU

The daVinci-MagiHuman model is a multimodal video generation model that provides text-to-video and text-plus-image-to-video capabilities. This document mainly introduces the NPU-based inference adaptation and optimization of the daVinci-MagiHuman model, which was verified on the Ascend Atlas A2/A3 environments.

## NPU Global Adaptation

At the script entry, that is, [sample_video.py](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/davinci-magihuman/sample_video.py), add the following code:

```python
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
```

## Importing the torch_npu Package on NPU

At the top of Python files that call torch_npu interfaces (such as `torch_npu.npu_fused_infer_attention_score`, `torch_npu.npu_rotary_mul`, etc.), import the torch_npu package:

```python
import torch
# Added
import torch_npu
```

## NPU npu_fused_infer_attention_score Operator Adaptation

This sample uses the npu_fused_infer_attention_score fused operator built into torch_npu to replace the FlashAttention operator. For details of this operator, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fused_infer_attention_score.md) (Chinese).

The npu_fused_infer_attention_score operator is used in [vae2_2_module.py](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/davinci-magihuman/inference/model/vae2_2/vae2_2_module.py) and [dit_module.py](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/davinci-magihuman/inference/model/dit/dit_module.py).

Using the `flash_attn_func` function in `models/davinci-magihuman/inference/model/dit/dit_module.py` as an example, the call is as follows:

```python
def flash_attn_func(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    b, s, n, d = query.shape
    _, _, kv_n, _ = key.shape
    scale = 1.0 / (d ** 0.5)

    original_dtype = query.dtype

    query = query.to(torch.bfloat16).contiguous()
    key = key.to(torch.bfloat16).contiguous()
    value = value.to(torch.bfloat16).contiguous()

    output, _ = torch_npu.npu_fused_infer_attention_score(
        query, key, value,
        num_heads=n,
        num_key_value_heads=kv_n,
        scale=scale,
        input_layout="BSND",
        sparse_mode=0
    )
    return output.to(original_dtype)
```

## NPU npu_rotary_mul Operator Adaptation

This sample uses the npu_rotary_mul fused operator built into torch_npu to replace the small-operator implementation in the source code. For details of npu_rotary_mul, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_rotary_mul.md) (Chinese).

The npu_rotary_mul operator is used in `models/davinci-magihuman/inference/model/dit/dit_module.py`.

Using the `apply_rotary_emb_npu` function in `models/davinci-magihuman/inference/model/dit/dit_module.py` as an example, the call is as follows:

```python
def apply_rotary_emb_npu(x, cos, sin):
    """
    Apply rotary embedding using npu_rotary_mul with pre-expanded cos/sin.
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (1, seqlen, 1, rotary_dim) - pre-expanded via cat([d, d], dim=-1)
    """
    ro_dim = cos.shape[-1]
    if ro_dim == x.shape[-1]:
        return torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode='half')
    x_rot = torch_npu.npu_rotary_mul(x[..., :ro_dim], cos, sin, rotary_mode='half')
    return torch.cat([x_rot, x[..., ro_dim:]], dim=-1)
```

## NPU npu_rms_norm Operator Adaptation

This sample uses the npu_rms_norm fused operator built into torch_npu to replace the small-operator implementation in the source code. For details of npu_rms_norm, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md) (Chinese).

The npu_rms_norm operator is used in `models/davinci-magihuman/inference/model/dit/dit_module.py`.

Using the `forward_single_expert` function in `models/davinci-magihuman/inference/model/dit/dit_module.py` as an example, the call is as follows:

```python
    def forward_single_expert(self, x: torch.Tensor, modality_dispatcher: Optional[ModalityDispatcher] = None, residual: Optional[torch.Tensor] = None):
        gamma = (self.weight + 1).contiguous()
        if residual is not None:
            gamma = gamma.to(torch.bfloat16)
            out, _, new_residual = torch_npu.npu_add_rms_norm(
                residual.to(gamma.dtype), x.to(gamma.dtype), gamma, self.eps)
            return out, new_residual
        out = torch_npu.npu_rms_norm(x, gamma, self.eps)[0]
        return out
```

## NPU npu_clipped_swiglu Operator Adaptation

This sample uses the npu_clipped_swiglu fused operator built into torch_npu to replace the small-operator implementation in the source code. For details of npu_clipped_swiglu, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_clipped_swiglu.md) (Chinese).

The npu_clipped_swiglu operator is used in `models/davinci-magihuman/inference/model/dit/dit_module.py`.

Using the `swiglu7` function in `models/davinci-magihuman/inference/model/dit/dit_module.py` as an example, the call is as follows:

```python
def swiglu7(x, alpha: float = 1.702, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x_f32 = x.to(torch.float32).contiguous()
    out = torch_npu.npu_clipped_swiglu(x_f32, alpha=alpha, limit=limit, bias=1.0, interleaved=True)
    return out.to(out_dtype)
```

## NPU npu_fast_gelu Operator Adaptation

This sample uses the npu_fast_gelu fused operator built into torch_npu to replace the small-operator implementation in the source code. For details of npu_fast_gelu, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_fast_gelu.md) (Chinese).

The npu_fast_gelu operator is used in `models/davinci-magihuman/inference/model/dit/dit_module.py`.

Using the `gelu7` function in `models/davinci-magihuman/inference/model/dit/dit_module.py` as an example, the call is as follows:

```python
def gelu7(x, limit: float = 7.0, out_dtype: Optional[torch.dtype] = None):
    out_dtype = x.dtype if out_dtype is None else out_dtype
    x = x.to(torch.float32)
    x.clamp_(max=limit)
    x = torch_npu.npu_fast_gelu(x)
    return x.to(out_dtype)
```

## NPU npu_add_rms_norm Operator Adaptation

This sample uses the npu_add_rms_norm fused operator built into torch_npu to replace the small-operator implementation in the source code. For details of npu_add_rms_norm, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/latest/apiref/torchnpuCustomsapi/docs/zh/custom_APIs/torch_npu/torch_npu-npu_add_rms_norm.md) (Chinese).

The npu_add_rms_norm operator is used in `models/davinci-magihuman/inference/model/dit/dit_module.py`.

Using the `forward_single_expert` function in `models/davinci-magihuman/inference/model/dit/dit_module.py` as an example, the call is as follows:

```python
    def forward_single_expert(self, x: torch.Tensor, modality_dispatcher: Optional[ModalityDispatcher] = None, residual: Optional[torch.Tensor] = None):
        gamma = (self.weight + 1).contiguous()
        if residual is not None:
            gamma = gamma.to(torch.bfloat16)
            out, _, new_residual = torch_npu.npu_add_rms_norm(
                residual.to(gamma.dtype), x.to(gamma.dtype), gamma, self.eps)
            return out, new_residual
        out = torch_npu.npu_rms_norm(x, gamma, self.eps)[0]
        return out
```

## Offload Optimization

The daVinci-MagiHuman open-source code runs on H100 GPU (80 GB memory) devices, but the single-card memory of the Ascend Atlas A2/A3 NPU is only 64 GB. Without offload, running the sr_540p/sr_540p_ti2v/sr_1080p/sr_1080p_ti2v modes would cause OOM (out of memory), so offload optimization was performed.

First, the implementation of the `CPUOffloadWrapper` class in [cpu_offload_wrapper.py](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/davinci-magihuman/inference/common/cpu_offload_wrapper.py) was modified, adding the `_save_cpu_copies`, `prepare_resident_backup`, `disable_offload`, and `enable_offload` functions, and modifying the `_run_with_optional_offload` function.

Second, in [video_generate.py](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/davinci-magihuman/inference/pipeline/video_generate.py), the `prepare_resident_backup`, `disable_offload`, and `enable_offload` functions of the `CPUOffloadWrapper` class are called.

For detailed changes, compare the differences of these two files between cann-recipes-infer and the [daVinci-MagiHuman](https://github.com/GAIR-NLP/daVinci-MagiHuman) open-source repository.

<!-- static-meta
{
  "sidebarTitle": "Inference Optimization",
  "hardware": ["atlas-a2", "atlas-a3"],
  "multimodalDitCache": ["None"],
  "cover": {"type": "auto"}
}
-->
