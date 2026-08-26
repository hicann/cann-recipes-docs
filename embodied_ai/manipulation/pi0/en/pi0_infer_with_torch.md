# Pi0 VLA Large Model Inference: Ascend Adaptation and Performance Optimization Guide
<br>


## Pi0 Ascend Adaptation
Based on torch_npu, the original code can be automatically adapted to the Ascend NPU. At the same time, replacing the float64 data type not supported by the NPU with the float32 data type in the code enables the inference to be quickly adapted and connected while ensuring that inference accuracy meets the requirements. Some NPU migration import statements are shown below:
```bash
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
<br>

## Pi0 Inference Optimization
### Optimization Goals
On the Ascend single-node single-card computing platform, implement inference performance optimization for the pi0 model to shorten inference time. The tested input and output information is as follows:
- Input: dual-view RGB images + robot body state + language text instruction
- Output: 50x6 joint angle sequence

### Optimization Process
- pi0 original out-of-the-box inference profiling analysis:
  
  Combining torch_npu.profiler and the MindStudio Insight computation pipeline visualization tool, we obtain the pipeline graph and operator computation statistics chart of pi0 during inference. Attention should be paid to the proportion of compute time versus idle time throughout the pi0 inference process. In the pipeline graph of pi0's original out-of-the-box inference, the idle time proportion is excessively large, from which it can be concluded that this is a host-bound problem, i.e., the device side keeps waiting for compute instructions to be dispatched.
  
  At the same time, attention should also be paid to the proportion of operators in AI_VECTOR_CORE and AI_CORE during inference. In pi0's original out-of-the-box inference process, the proportion occupied by AI_CORE operators is relatively small. From this, it can be concluded that the proportion of large fused operators friendly to the Ascend NPU is small during computation, failing to fully leverage the compute performance of the NPU Cube (matrix unit). The original pi0 code's inference computation logic is not suitable for the Cube.

- pi0 inference code structure analysis:
  
  According to the pi0 inference code, the slow system calls PaliGemma's vision encoder to extract high-dimensional representations from a single frame image; the fast system is a Transformer-based Flow-Matching policy model responsible for generating action sequences through parallel denoising. In the code, the rotary encoding part, the FA part, the layer-norm(add) part, and the QKV computation part are all formed by concatenating small operators or connecting multiple steps in series, resulting in computation that is not friendly to the Ascend NPU. Therefore, starting from means such as code logic, Ascend-friendly fused operator replacement, and graph mode, it is of great significance to optimize pi0's inference performance on the Ascend NPU. The following is a brief introduction to several optimization strategies:
  
- Optimization strategy - rotary encoding sin/cos computation logic optimization:
  
  By modifying the rotary encoding computation logic in the forward function of `class PaliGemmaWithExpertModel(PreTrainedModel)` in [paligemma_with_expert.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/manipulation/pi0/infer_with_torch/paligemma_with_expert.py), changing the inputs and outputs of the apply_rope function, using the torch_npu.npu_rotary_mul fused operator, splitting the fused computation of Q/K_states, and moving the sin/cos computation outside the for loop for unified computation to reduce the number of repeated computations, optimization and acceleration of the rotary encoding part can be achieved. Part of the optimized code is shown below:
  ```
  def apply_rope(query_states, key_states, cos, sin):
      n_q = query_states.shape[2]
      n_k = key_states.shape[2]
      
      merged_states = torch.cat([query_states, key_states], dim=2)  # shape is [B, S, n_q + n_k, D]
      
      merged_rot = torch_npu.npu_rotary_mul(merged_states, cos, sin)
      
      q_rot, k_rot = merged_rot.split([n_q, n_k], dim=2)

      return q_rot, k_rot
  ```
  

- Optimization strategy - flash attention NPU native operator optimization:
  
  By replacing the FA part in `paligemma_with_expert.py` with the npu_prompt_flash_attention fused operator, the fast computing capability of the fused operator on the NPU is reused to improve the compute performance of this module. At the same time, attention should be paid to the differences between the original FA computation method in pi0 and the npu_prompt_flash_attention fused operator in the implementation of the mathematical formulas, especially the difference in the mathematical formula implementation for attention_mask (positions where the bool type is True are converted to negative infinity), and corresponding adaptation modifications should be made, moving some computation steps outside the loop for pre-computation, so that optimization and acceleration of the FA part can be achieved. Part of the optimized code is shown below:
  ```bash
  attention_mask = torch.logical_not(attention_mask).to(dtype=torch.int8, memory_format=torch.contiguous_format)
  attention_mask = attention_mask[:, None, :, :]

  ......

  def eager_attention_forward(
      self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
  ):
      att_output = torch_npu.npu_prompt_flash_attention(
          query_states,
          key_states.contiguous(),
          value_states.contiguous(),
          num_heads=self.num_att_heads_forward,
          input_layout="BSND",
          scale_value=self.scale_value,
          pre_tokens=65535,
          next_tokens=65535,
          atten_mask=attention_mask,
          num_key_value_heads=self.num_key_value_heads_forward
      )

      att_output = att_output.reshape(batch_size, -1, self.num_att_heads_forward * head_dim)

      return att_output
  ```


- Optimization strategy - layernorm part NPU native operator optimization:
  
  Replace the layernorm and add operations in the first residual computation with the npu_add_rms_norm fused operator, reusing the fast computing capability of the fused operator on the NPU to improve the compute performance of this module. At the same time, attention should be paid to the differences between the layernorm computation method in pi0 and the npu_add_rms_norm fused operator in the product coefficients of the mathematical formulas, and corresponding adaptation modifications should be made. In addition, the native layer.input_layernorm computation part in the forward function of `class PaliGemmaWithExpertModel(PreTrainedModel)` in `paligemma_with_expert.py` can be replaced with Ascend's native npu_rms_norm to achieve optimization and acceleration of the layernorm part. Part of the optimized code is shown below:
  ```bash
  hidden_states = torch_npu.npu_rms_norm(
      hidden_states,
      layer.input_layernorm.weight.add(self.ones_add),
      1e-6
  )[0]

  ......

  out_emb, _, after_first_residual = torch_npu.npu_add_rms_norm(
      layer.self_attn.o_proj(att_output[:, start:end]),
      hidden_states.to(torch.bfloat16),
      layer.post_attention_layernorm.weight.add(self.ones_add),
      1e-6
  )
  ```


- Optimization strategy - qkv computation part NPU native operator optimization:
  
  In the forward computation process of the pi0 model, convert the separate q/k/v computation into a computation using fused qkv weights, then split the fused computation result with split, saving cache swap-in/swap-out and using large matrix multiplication to improve Cube utilization. By performing corresponding optimizations on the q/k/v computation part in `paligemma_with_expert.py`, optimization and acceleration of the qkv computation part can be achieved. Part of the optimized code is shown below:
  ```bash
  # Fuse the q/k/v weights into a single qkv linear layer
  @torch.no_grad()
  def fuse_qkv_weights(self):
      """
      Concatenate the q/k/v weights of self_attn of each layer into a single qkv linear layer
      """
      for model in self.models:  # paligemma + gemma_expert
          for layer in model.layers:      
              attn = layer.self_attn
              w_q = attn.q_proj.weight
              w_k = attn.k_proj.weight
              w_v = attn.v_proj.weight

              w_fused = torch.cat([w_q, w_k, w_v], dim=0).bfloat16()

              # Create the qkv linear layer
              attn.qkv = nn.Linear(w_fused.shape[1], w_fused.shape[0], bias=False, device=w_q.device).to(torch.bfloat16)

              # Assign the concatenated weights to the qkv weights              
              attn.qkv.weight.data.copy_(w_fused)

  ......

  qkv = layer.self_attn.qkv(hidden_states)

  q_proj, k_proj, v_proj = qkv.split([q_out, kv_out, kv_out], dim=-1)
  ```


- Optimization strategy - graph mode optimization:
  
  The native out-of-the-box pi0 inference performance has a severe host-bound problem on the NPU. Therefore, to reduce the per-operator dispatch and synchronization overhead from the CPU to the NPU, combined with the TorchAir graph mode compilation module of torch_npu, the fast system and slow system parts of pi0 can be compiled as a whole graph as much as possible in [modeling_pi0.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/manipulation/pi0/infer_with_torch/modeling_pi0.py), thereby achieving overall computation optimization and acceleration. Part of the optimized code is shown below:
  ```bash
  import torch_npu
  import torchair as tng
  from torchair.configs.compiler_config import CompilerConfig

  config = CompilerConfig()
  config.experimental_config.frozen_parameter = True
  config.experimental_config.tiling_schedule_optimize = True

  npu_backend = tng.get_npu_backend(compiler_config=config)

  ......

  self.compiled_embed_and_model_forward = torch.compile(
      self.embed_and_model_forward,
      dynamic=False,
      fullgraph=True,
      backend=npu_backend
  )

  self.denoise_step_all_compile = torch.compile(
      self.denoise_step_all.forward,
      dynamic=False,
      fullgraph=True,
      backend=npu_backend
  )

  ......

  (
      prefix_pad_masks,
      dt,
      x_t,
      time,
      past_key_values
  ) = self.compiled_embed_and_model_forward(
      noise, images, img_masks, lang_tokens, lang_masks
  )

  for step in range(self.config.num_steps):
      self.denoise_step_all_compile(
          state,
          prefix_pad_masks,
          past_key_values,
          x_t,
          bsize,
          time,
          dt
      )
  ```


- Summary of optimization strategy results:
  
  Based on the above Ascend migration-optimization strategies, a profiling analysis based on MindStudio Insight was performed on a single inference process. The total proportion of Ascend NPU compute time increased to 94%, the compute time proportion of AI_CORE-type Ascend fused operators increased to 50%, the total compute time was 71.6 milliseconds, and the idle time was 4.58 milliseconds. The analysis results fully demonstrate that the migration-optimization strategies improve pi0's affinity for Ascend, fully leverage the fused operator advantages and compute power advantages of the Ascend NPU, and fully show the feasibility and ease of migrating the embodied AI pi0 model to Ascend.

<br>

<!-- static-meta
{
  "sidebarTitle": "Inference Optimization",
  "hardware": ["atlas-a2"],
  "cannFeatures": ["npugraph"],
  "cover": {"type": "auto"}
}
-->
