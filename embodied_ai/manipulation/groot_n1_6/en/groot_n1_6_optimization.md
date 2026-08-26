# GR00T N1.6 VLA Model Inference: Ascend Adaptation and Performance Optimization Guide

## GR00T N1.6 Ascend Adaptation
Based on torch_npu, the original code can be automatically adapted to the Ascend NPU. At the same time, replacing the flash attention related operators in the code with Ascend fused operators and replacing the float64 data type not supported by the NPU with the float32 data type enables the inference to be quickly adapted and connected while ensuring that inference accuracy meets the requirements. Some NPU migration import statements are shown below:
```bash
import torch_npu
from torch_npu.contrib import transfer_to_npu
```
<br>

## GR00T N1.6 Ascend Optimization
### Optimization Goals
On the Ascend A3 platform, implement single-card inference performance optimization for the GR00T N1.6 model to reduce inference time. The tested input and output information is as follows:
- Input: single-view RGB image + robot body state + language text instruction. By default, the gr1.PickNPlace dataset provided by demo_data is used, with 4 denoising steps.
- Output: each inference step outputs a 16x29 joint angle sequence, including joint information for both hands, both arms, waist, and other body parts.

### Optimization Process
- Out-of-the-box inference profiling analysis:
  
  Combining torch_npu.profiler and the MindStudio Insight computation pipeline visualization tool, we obtain the pipeline graph and operator computation statistics table of GR00T N1.6 during inference. Pay special attention to the proportion of compute time versus idle time throughout the GR00T N1.6 inference process, the proportion of each operator type, and other information. In the out-of-the-box inference pipeline graph of GR00T N1.6, the idle time proportion is excessively large, with compute accounting for less than 20%. The device side keeps waiting for compute instructions to be dispatched, indicating a host-bound problem.
  
  At the same time, attention should also be paid to the proportion of operators in AI_VECTOR_CORE and AI_CORE during inference. In the out-of-the-box inference process of GR00T N1.6, the proportion occupied by AI_CORE operators is relatively small. From this, it can be concluded that the proportion of large fused operators friendly to the Ascend NPU is small during computation, failing to fully leverage the compute performance of the NPU Cube unit. In addition, in the operator computation statistics table, operators such as transpose and cast account for relatively large proportions, reaching 16% and 10% respectively, which likewise fails to exploit the NPU's compute advantages.

- GR00T N1.6 inference structure analysis:
  
  The GR00T N1.6 network consists of two parts: a slow system and a fast system. The slow system uses Cosmos-Reason-2B, which is composed of four parts in series: a vision encoder, a multimodal projection adapter, an LLM base decoder, and a cross-modal bidirectional fusion layer. The fast system is a diffusion Transformer (DiT) model responsible for generating action sequences through parallel denoising. In the code, the rotary encoding part, the FA part, and the layer-norm(add) part are all formed by concatenating small operators or connecting multiple steps in series, resulting in computation that is not friendly to the Ascend NPU. Therefore, starting from means such as code logic optimization, Ascend fused operator replacement, and graph mode enablement, the inference optimization of GR00T N1.6 on the Ascend A3 can be accelerated. The following describes several optimization strategies:
  

- Optimization strategy - graph mode enablement:

  Because the out-of-the-box version has a host-bound problem, the entire GR00T N1.6 network or part of the network can be compiled and the instructions dispatched at once to reduce device waiting. The GR00T N1.6 network mainly consists of two parts: the backbone and the action head. Because the backbone contains many dynamic operations and cannot be compiled with TorchAir, the following code in [standalone_inference_script.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/manipulation/Isaac-GR00T/scripts/deployment/standalone_inference_script.py) uses torchair to enable graph mode for the `action head` part. After enablement, about two minutes of graph compilation time is added to the first inference step, but the speed of subsequent inference steps is greatly improved:
  ```
    if local_model_path is not None:
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=local_model_path,
        device="npu" if torch.npu.is_available() else "cpu",
    )

    if hasattr(policy.model, 'action_head') and hasattr(policy.model.action_head, 'model'):
        policy.model.action_head.model = torch.compile(
            policy.model.action_head.model,
            backend="npugraphs",
            fullgraph=True,
            dynamic=False,
        )
        logging.info(" npugraphs mode enabled for action_head (dynamic=True)")
  ```

- Optimization strategy - flash attention fused operator replacement:

  By using a monkey-patching approach in [modeling_siglip2_patch.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/manipulation/Isaac-GR00T/adaptor_patches/modeling_siglip2_patch.py) to use the `npu_fused_infer_attention_score` fused operator, the fast computing capability of the fused operator on the NPU is reused to improve the compute performance of this module.

  ```
  ...
        out, _ = torch.ops.npu.npu_fused_infer_attention_score(
            query=fa_args.q[:, :, i:end, :].contiguous(),
            key=fa_args.k[:, :, i:end, :].contiguous(),
            value=fa_args.v[:, :, i:end, :].contiguous(),
            num_heads=curr_heads,
            num_key_value_heads=curr_heads,
            input_layout=fa_args.input_layout,
            scale=fa_args.scale,
            atten_mask=None,
            sparse_mode=0,
        )
        outputs.append(out)

    # Concatenate results
    final_output = torch.cat(outputs, dim=2) if len(outputs) > 1 else outputs[0]
    
    # Remove padding, restore original dimension
    if head_dim_padded != head_dim:
        final_output = final_output[..., :head_dim]
    
    return final_output, None
  ```

- Optimization strategy - rotary encoding sin/cos computation logic optimization:
  
  By modifying the rotary encoding computation logic in the forward function of [modeling_siglip2_patch.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/manipulation/Isaac-GR00T/adaptor_patches/modeling_siglip2_patch.py), changing the inputs and outputs of the apply_rope function, using the torch_npu.npu_rotary_mul fused operator, splitting the fused computation of Q/K_states, and moving the sin/cos computation outside the for loop for unified computation to reduce the number of repeated computations, optimization and acceleration of the rotary encoding part can be achieved. Part of the optimized code is shown below:
  ```
    def apply_rope_npu(xq, xk, freqs_cis):
        """NPU-optimized RoPE implementation using npu_rotary_mul fused operator."""
        cos = freqs_cis.real.unsqueeze(-2)
        sin = freqs_cis.imag.unsqueeze(-2)
        xq_out = torch_npu.npu_rotary_mul(xq, cos, sin)
        xk_out = torch_npu.npu_rotary_mul(xk, cos, sin)
        return xq_out, xk_out
  ```


- Optimization strategy - layernorm part NPU native operator optimization:
  
  Replace the layernorm and add operations in the computation with the npu_add_rms_norm fused operator, reusing the fast computing capability of the fused operator on the NPU to improve the compute performance of this module. In addition, the native layer.input_layernorm computation part in the forward function of [modeling_siglip2_patch.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/manipulation/Isaac-GR00T/adaptor_patches/modeling_siglip2_patch.py) can be replaced with Ascend's native `npu_rms_norm` to achieve optimization and acceleration of the layernorm part. Part of the optimized code is shown below:
  ```bash
    class NpuRMSNorm(nn.Module):        
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, hidden_states):
            # NPU optimization: use fused operator
            if hidden_states.device.type == "npu" and hasattr(torch_npu, "npu_rms_norm"):
                # npu_rms_norm returns (output, rstd), we only need output
                result = torch_npu.npu_rms_norm(hidden_states, self.weight, self.variance_epsilon)
                return result[0] if isinstance(result, tuple) else result
            
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            return self.weight * hidden_states.to(input_dtype)
  ```


- Summary of optimization strategy results:
  
  Based on the above Ascend migration-optimization strategies, a profiling analysis based on MindStudio Insight was performed on a single inference process. The total proportion of compute time on A3 increased to 70%, and the total compute time was 82 ms, a significant improvement over the out-of-the-box performance of 280 ms.

<br>

<!-- static-meta
{
  "sidebarTitle": "Inference Optimization",
  "hardware": ["atlas-a3"],
  "cannFeatures": ["npugraph"],
  "cover": {"type": "auto"}
}
-->
