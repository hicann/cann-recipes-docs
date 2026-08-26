# SANA-Video Model Inference Optimization on NPU
This document mainly introduces the NPU-based inference optimization strategies for the SANA-Video model. The related optimization logic is located in [npu_patches.py](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/sana-video/patches/npu_patches.py).

## 1×1 conv2d Optimization
In `patch_conv1x1_matmul()`, the 1×1 conv2d detection and conversion logic is added to `ConvLayer.forward`:

```python
if self.conv.kernel_size == (1, 1):
    B, Ci, H, W = x.shape
    Co, *_ = self.conv.weight.shape
    x_trans = x.view(B, Ci, -1).permute(0, 2, 1).contiguous()
    kernel_trans = self.conv.weight.view(Co, Ci)
    x = x_trans @ kernel_trans.T
    if self.conv.bias is not None:
        x += self.conv.bias
    x = x.permute(0, 2, 1).contiguous().view(B, Co, H, W)
else:
    x = self.conv(x)
```

## Temporal conv2d hw-axis Swap Optimization
In `patch_temporal_conv_swap()`, `t_conv_swap` is added to `GLUMBConvTemp` and called in `forward`:

```python
self.t_conv_swap = nn.Conv2d(
    out_feature,
    out_feature,
    kernel_size=(1, t_kernel_size),
    stride=1,
    padding=(0, t_padding),
    bias=False,
)

x_reshaped = x.view(batch_size, time_steps, channels, height * width).permute(0, 2, 1, 3)
x_out = x_reshaped + self.t_conv_swap(x_reshaped.transpose(-1, -2)).transpose(-1, -2)
```

In [inference_sana_video.py](https://gitcode.com/cann/cann-recipes-infer/blob/master/models/sana-video/inference_video_scripts/inference_sana_video.py), after loading the model parameters, synchronize the swapped weights once:

```python
for block in model.blocks:
    block.mlp.t_conv_swap.weight = nn.Parameter(block.mlp.t_conv.weight.permute(0, 1, 3, 2))
```

## npu_rms_norm Operator Adaptation
This sample uses the `npu_rms_norm` fused operator built into `torch_npu` to replace the original small-operator implementation. For details of `npu_rms_norm`, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/%EF%BC%88beta%EF%BC%89torch_npu-npu_rms_norm.md) (Chinese). The `npu_rms_norm` fused operator is enabled in `patch_rms_norm()`:

```python
def forward(self, x):
    if _npu_available:
        return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]
    return original_forward(self, x)
```

## npu_rotary_mul Operator Adaptation
This sample uses the `npu_rotary_mul` fused operator built into `torch_npu` to replace the original rotary small-operator implementation. For details of `npu_rotary_mul`, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_rotary_mul.md) (Chinese). The `npu_rotary_mul` fused operator is enabled in `patch_rotary_mul()`:

```python
def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
    if _npu_available:
        cos = freqs.real.unsqueeze(-1).expand(-1, -1, -1, -1, 2).flatten(-2)
        sin = freqs.imag.unsqueeze(-1).expand(-1, -1, -1, -1, 2).flatten(-2)
        x_out = torch_npu.npu_rotary_mul(hidden_states.permute(0, 1, 3, 2), cos, sin, "interleave")
        return x_out.permute(0, 1, 3, 2)
    else:
        x_rotated = torch.view_as_complex(hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
        x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return x_out.type_as(hidden_states)

q_rotated = apply_rotary_emb(q, rotary_emb)
k_rotated = apply_rotary_emb(k, rotary_emb)
```

## npu_fusion_attention Operator Adaptation
This sample uses the `npu_fusion_attention` fused operator built into `torch_npu` to replace the original attention small-operator implementation. For details of `npu_fusion_attention`, see the [Ascend Community Documentation](https://www.hiascend.com/document/detail/zh/Pytorch/710/apiref/torchnpuCustomsapi/context/torch_npu-npu_fusion_attention.md) (Chinese). The `npu_fusion_attention` fused operator is enabled in `patch_fusion_attention()`:

```python
if mask is not None and mask.ndim == 2:
    mask = (1 - mask.to(q.dtype)) * -10000.0
    mask = mask[:, None, None].repeat(1, self.num_heads, num_tokens, 1)

if _npu_available:
    bool_mask = None if mask is None else mask < -1000
    x = torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        q.shape[1],
        input_layout="BNSD",
        pse=None,
        atten_mask=bool_mask,
        scale=1.0 / math.sqrt(q.shape[-1]),
        pre_tockens=2147483647,
        next_tockens=2147483647,
        keep_prob=1,
    )[0]
else:
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

x = x.transpose(1, 2).contiguous()
```

<!-- static-meta
{
  "sidebarTitle": "Inference Optimization",
  "hardware": ["atlas-a2", "atlas-a3"],
  "multimodalDitCache": ["None"],
  "cover": {"type": "auto"}
}
-->
