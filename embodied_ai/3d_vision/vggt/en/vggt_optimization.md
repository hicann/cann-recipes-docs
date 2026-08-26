# NPU VGGT Model Inference Optimization Practices
This document mainly introduces the NPU-based inference optimization strategies for the VGGT model, including the following optimization points:
- Cos/Sin operator input optimization
- Rotary position encoding computation optimization
  - Enabling the NPU npu_rotary_mul fused operator
  - Redundant computation elimination
- DPT head position encoding computation optimization
- Enabling the NPU npu_add_layer_norm fused operator
- Converting weight format to BF16
- Early conversion of 2D convolution kernels to the private format
- Enabling the NPU npu_fused_infer_attention_score fused operator
- Enabling Ulysses and Ring attention sequence-parallel inference
---
## Performance Optimization
### Cos/Sin Operator Optimization
**Optimization reason:** In the original network structure code, the input data type of the Cos and Sin operators is double, causing these two operators to be dispatched to the AI CPU for execution, resulting in low operator performance.

**Optimization method:** Modify the data type of the omega variable in the make_sincos_pos_embed function in the [utils.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/3d_vision/vggt/vggt/heads/utils.py) file from torch.double to torch.bfloat16.
```python
'''Replacement part
def _make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()
'''
#After replacement
def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.bfloat16, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb
```

### Rotary Position Encoding Optimization
#### Enabling the npu_rotary_mul Fused Operator
- **Optimization reason:** In the original network code, the rotary operation is implemented via `(token * cos) + (self._rotate_features(tokens) * sin)`, and the host side needs to dispatch multiple small operators.
- **Optimization method:** Modify the _apply_1d_rope function in the [rope.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/3d_vision/vggt/vggt/layers/rope.py) file, and use `npu_rotary_mul` to replace the original operator implementation.
```python
'''Replacement part
def __apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
  # Embed positions with frequency components
  cos = F.embedding(positions, cos_comp)[:, None, :, :]
  sin = F.embedding(positions, sin_comp)[:, None, :, :]

  # Apply rotation
  return (tokens * cos) + (self._rotate_features(tokens) * sin)
'''
#After replacement
def _apply_1d_rope(
        self, tokens: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
  # Apply rotation
  return torch_npu.npu_rotary_mul(tokens, cos, sin, rotary_mode='half')
```

#### Redundant Operation Removal
- **Optimization reason:** In the rope module implementation of the VGGT Attention network, the cos and sin values need to be recomputed for every computation, resulting in redundant computation, specifically as follows: 
    - In the _apply_1d_rope function in the `vggt/layers/rope.py` file, the cos and sin variables are computed separately via `cos = F.embedding(positions, cos_comp)[:, None, :, :]` and `sin = F.embedding(positions, sin_comp)[:, None, :, :]`.
    - The forward function in the `vggt/layers/rope.py` file needs to compute the maximum value of the input variable positions via the max function.
    - Every time the rope computation is performed on the q and k variables, the cos and sin need to be computed in both the vertical dimension and the horizontal dimension, and the maximum value of positions also needs to be computed.
- **Optimization method:** The input of the rotary position encoding computation depends on the positions variable, and the positions variable is related to the height and width of the input image. Therefore, by establishing a dictionary keyed by (height, width) to cache the computation results of the cos and sin variables and the maximum value of positions, repeated computation can be reduced for images of the same size under different inputs.
```python
'''Replacement part
def __compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  cache_key = (dim, seq_len, device, dtype)
  if cache_key not in self.frequency_cache:
      # Compute frequency bands
      exponents = torch.arange(0, dim, 2, device=device).float() / dim
      inv_freq = 1.0 / (self.base_frequency**exponents)

      # Generate position-dependent frequencies
      positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
      angles = torch.einsum("i,j->ij", positions, inv_freq)

      # Compute and cache frequency components
      angles = angles.to(dtype)
      angles = torch.cat((angles, angles), dim=-1)
      cos_components = angles.cos().to(dtype)
      sin_components = angles.sin().to(dtype)
      self.frequency_cache[cache_key] = (cos_components, sin_components)

  return self.frequency_cache[cache_key]
'''
# After replacement
def _compute_frequency_components(
        self, dim: int,  input_positions: torch.tensor, device: torch.device, dtype: torch.dtype,
        height: None, width: None, batch_size: None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
  if height == None or width == None:
      seq_len = int(input_positions.max()) + 1
  else:
      hw_key = (height, width)
      if hw_key not in self.max_position_cache:
        self.max_position_cache[hw_key] = int(input_positions.max()) + 1
      seq_len = self.max_position_cache[hw_key]
  cache_key = (dim, seq_len, device, dtype)
  if cache_key not in self.frequency_cache:
      # Compute frequency bands
      exponents = torch.arange(0, dim, 2, device=device).float() / dim
      inv_freq = 1.0 / (self.base_frequency**exponents)

      # Generate position-dependent frequencies
      positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
      angles = torch.einsum("i,j->ij", positions, inv_freq)

      # Compute and cache frequency components
      angles = angles.to(dtype)
      angles = torch.cat((angles, angles), dim=-1)
      cos_components = angles.cos().to(dtype)
      sin_components = angles.sin().to(dtype)
      self.frequency_cache[cache_key] = (cos_components, sin_components)

  cos_components, sin_components = self.frequency_cache[cache_key]

  if height == None or width == None or batch_size == None:
      vertical_cos = F.embedding(input_positions[...,0], cos_components)[:, None, :, :]
      vertical_sin = F.embedding(input_positions[...,0], sin_components)[:, None, :, :]
      horizontal_cos = F.embedding(input_positions[...,1], cos_components)[:, None, :, :]
      horizontal_sin = F.embedding(input_positions[...,1], sin_components)[:, None, :, :]
      return vertical_cos, vertical_sin, horizontal_cos, horizontal_sin

  sub_cache_key = (height, width)
  if sub_cache_key not in self.cos_sin_cache:
      # Embed positions with frequency components
      vertical_cos = F.embedding(input_positions[...,0], cos_components)[:, None, :, :]
      vertical_sin = F.embedding(input_positions[...,0], sin_components)[:, None, :, :]
      horizontal_cos = F.embedding(input_positions[...,1], cos_components)[:, None, :, :]
      horizontal_sin = F.embedding(input_positions[...,1], sin_components)[:, None, :, :]
      self.cos_sin_cache[sub_cache_key] = (vertical_cos, vertical_sin, horizontal_cos, horizontal_sin)
  vertical_cos, vertical_sin, horizontal_cos, horizontal_sin = self.cos_sin_cache[sub_cache_key]
  return vertical_cos, vertical_sin, horizontal_cos, horizontal_sin

```
### DPT Head Position Encoding Computation Optimization
- **Optimization reason:** In the DPT head implementation of the VGGT network, the position encoding needs to be recomputed for the input on every computation, resulting in redundant computation.
- **Optimization method:** Since the position encoding result depends on the size of the input image and the length of the tokens, a dictionary is used to cache and save the position encoding results in advance, avoiding redundant computation and reducing the amount of computation.
```python
'''Replacement part
def __apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
  patch_w = x.shape[-1]
  patch_h = x.shape[-2]
  pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
  pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
  pos_embed = pos_embed * ratio
  pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
  return x + pos_embed
'''
#After replacement
def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
   if (W, H, x.shape) not in self.pos_embed_cache:
       patch_w = x.shape[-1]
       patch_h = x.shape[-2]
       pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
       pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
       pos_embed = pos_embed * ratio
       pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
       self.pos_embed_cache[(W, H, x.shape)] = pos_embed
   pos_embed = self.pos_embed_cache[(W, H, x.shape)]
   return x + pos_embed
```
### Add+LayerNorm Fused Operator
- **Optimization reason:** Replace the two small operators Add and LayerNorm with a fused large operator to reduce the number of operator dispatches and the memory reads/writes of intermediate results, improving performance.
- **Optimization method:** Use the fused operator `npu_add_layer_norm` to replace the original separate execution of Add+LayerNorm.
```python
#Before replacement (the two operators execute separately)
def original_layernorm_forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
    if residual is None:
        return torch_npu.npu_layer_norm_eval(x, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        residual = residual + x  # Operator 1: Add operation (executed separately)
        y = torch_npu.npu_layer_norm_eval(residual, self.normalized_shape, self.weight, self.bias, self.eps)  # Operator 2: LayerNorm (executed separately)
        return y, residual

#After replacement (fused operator, Add+LayerNorm merged into one operator)
def vggt_layernorm_forward(self, x: torch.Tensor, residual: Optional[torch.Tensor] = None):
    if residual is None:
        return torch_npu.npu_layer_norm_eval(x, self.normalized_shape, self.weight, self.bias, self.eps)
    else:
        y, _, _, residual = torch_npu.npu_add_layer_norm(residual, x, self.weight, self.bias, self.eps, additional_output = True)
        return y, residual

nn.LayerNorm.forward = vggt_layernorm_forward
```
> **Note:** Both before and after the optimization, the NPU-compatible `npu_layer_norm_eval` is used as the underlying LayerNorm implementation; the only difference is whether the Add operation is fused (2 operators → 1 fused operator).
### Using BF16 Weights
- **Optimization reason:** Currently, the VGGT network weights use the float data type. Consider converting the VGGT network weights to bfloat16.
- **Optimization method:** After loading the model, convert the model weights to bfloat16. Using this approach yields a 6.62% performance gain. Compared with fp32, the accuracy of the camera pose estimation task drops from 0.919 to 0.911, with an accuracy loss within 0.5%.
<!-- - Performance gain after quantization compared with the original model, and the corresponding accuracy loss -->

### Using INT8 Weights
- **Optimization reason:** Currently, the Linear layers of the VGGT network use W8A8 quantization with controllable accuracy (loss within 1%). Consider converting the VGGT network weights to int8 offline.
- **Optimization method:** Use dynamic per-token quantization for the activations of some Linear layers and static per-channel quantization for the weights. The fp32 (original) model size is 4.9 GB, the bf16 model size is 2.46 GB, and the int8 model size is 2.16 GB. Compared with bf16, the accuracy of the camera pose estimation task drops from 0.911 to 0.907, with an accuracy loss within 0.5%.
- **Enable method:** Disabled by default. To enable it, set enableW8A8 to True.

### Early Conversion of Convolution Kernels to the Private Format
- **Optimization reason:** For 2D convolution operations on the NPU, the convolution kernels need to be converted to the `Fractal_Z` format in advance via the Transdata operator, incurring data format conversion overhead during inference.
- **Optimization method:** After loading the model weights, convert the data format of the 2D convolution kernels to `Fractal_Z` in advance, thereby avoiding the introduction of conversion overhead.
```python
#After replacement
def cast_model_weight(model):
    def __format_cast(module, class_name):
        if issubclass(class_name, torch.nn.Conv2d):
            if module.groups > 1:
                return
            if hasattr(module, "weight") and module.weight is not None and \
                "weight" in dict(module.named_parameters()):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 4)
        return module
    def cast_weight(module):
        current_class = module.__class__
        module = __format_cast(module, current_class)

        if not module.children:
            return
        for sub_module in module.children():
            if isinstance(sub_module, torch.nn.Module):
                sub_module = cast_weight(sub_module)
        return module
    for _, module in model.named_modules():
        module = cast_weight(module)
    return model
# At inference time
model = VGGT()
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint)
model = model.to(dtype)
model.to(device).eval()
model = cast_model_weight(model)#Call the cast_model_weight function to convert the convolution kernel data format in advance before inference
predictions = model(images)
```

### NPU fused_infer_attention_score Operator Adaptation
- **Optimization reason:** The npu_fused_infer_attention_score fused operator is a FlashAttention operator provided by Ascend Extension for PyTorch that adapts to both incremental and full inference scenarios. On Ascend environments, it delivers better inference performance than SDPA.
- **Optimization method:** Implement operator separation for the Frame-Global two-layer architecture. Frame Attention handles short intra-frame sequences and keeps using PyTorch SDPA; Global Attention handles long cross-frame sequences and switches to the NPU FIA fused operator.
```python
# Original: uniformly uses SDPA
x = F.scaled_dot_product_attention(q, k, v, dropout_p=...)

# Optimized: dispatch based on is_global_attention
if self.is_global_attention:
    # Global uses FIA
    x = torch_npu.npu_fused_infer_attention_score(
        q, k, v, num_heads=num_heads, scale=scale,
        input_layout="BNSD", pre_tokens=65535, 
        next_tokens=65535, inner_precise=0
    )[0]
else:
    # Frame uses SDPA
    x = F.scaled_dot_product_attention(q, k, v, ...)
```

### Sequence-Parallel Inference Adaptation
- **Optimization reason:** The VGGT network natively supports only single-card inference, with all computation completed on one card, failing to effectively leverage the multi-card advantage of Atlas A2; multi-card parallelism splits the sequence across ranks, effectively improving the inference performance of the entire network.
- **Optimization method:**

  1. Ulysses parallelism splits the num_heads dimension across multiple cards and aggregates the sequence dimension through all-to-all communication, so that each rank sees the full sequence but processes only part of the attention heads.
  
  2. Ring parallelism splits the sequence across multiple cards and hides communication overhead through the overlap strategy: it uses the LSE (log-sum-exp) information returned by the NPU FIA operator to support numerically stable merging of chunked attention results.
```python
# Ulysses parallelism (head dimension split)
# Swap the head and sequence dimensions via all-to-all communication
# Input: [B, shard_s, H, D] - each rank holds part of the sequence and the full heads
# Output: [B, S, shard_hc, D] - each rank holds the full sequence and part of the heads
def _all_to_all_head_to_seq(self, input_):
    input_t = input_.reshape(bs, shard_s, world_size, shard_hc, d)
    input_t = input_t.permute(2, 1, 0, 3, 4).contiguous()
    dist.all_to_all_single(output, input_t, group=self.ulysses_pg)
```
```python
# Ring parallelism (sequence dimension split)
# Ring Attention with Overlap: communicate and compute simultaneously
def _ring_attention_overlap(self, params):
    # Step 1: Asynchronously launch the allgather of K/V
    k_handle = dist.all_gather_into_tensor(k_gathered, k, async_op=True)
    v_handle = dist.all_gather_into_tensor(v_gathered, v, async_op=True)
    # Step 2: Compute local attention (Q @ local_K/V)
    out_local, lse_local = self._compute_attention_with_lse(q, k, v)
    # Step 3: Wait for communication to complete and compute cross attention (Q @ other_K/V)
    k_handle.wait()
    out_others, lse_others = self._compute_attention_with_lse(q, k_others, v_others)
    # Step 4: Merge results using LSE
    out_merged = self._merge_two_outputs(out_local, lse_local, out_others, lse_others)
```
- **Additional impact:** After adapting sequence parallelism to the VGGT network, the following is the overall network accuracy
  |Parallelism|AUC@30|
  |:---:|:---:|
  |Single card|91.10|
  |Ulysses=2|91.12|
  |Ulysses=4|91.10|
  |Ulysses=8|90.86|
  |Ring=2|91.12|
  |Ring=4|91.09|
  |Ring=8|90.88|

## Performance Optimization Metrics
This solution uses an 8-card Atlas 800I A2 inference product. Using the sample data provided by VGGT (`examples/kitchen`), which contains 25 images, the performance metrics are as follows
|Enabled method|Inference time (ms)|
|:---:|:---:|
|Cos/Sin operator optimization|1324.83|
|Rotary position encoding optimization|1239.55|
|DPT head computation optimization|1211.26|
|npu_add_layer_norm fused operator|1208.17|
|Weight format to BF16|1128.18|
|Early conversion to the private format|1121.09|

After adopting sequence-parallel multi-card inference, the performance improvements are as follows
|Parallelism|Inference improvement|
|:---:|:---:|
|Ulysses=2|1.75x|
|Ulysses=4|3.43x|
|Ulysses=8|6.47x|
|Ring=2|1.82x|
|Ring=4|3.48x|
|Ring=8|6.42x|

<!-- static-meta
{
  "sidebarTitle": "Inference Optimization",
  "parallelism": ["sequence-parallel"],
  "quantization": ["int8"],
  "hardware": ["atlas-a2"],
  "cover": {"type": "auto"}
}
-->
