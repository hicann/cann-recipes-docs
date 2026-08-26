# NPU Hunyuan3D Model Inference Optimization Practices
This document mainly introduces the NPU-based inference optimization strategies for the Hunyuan3D model, including the following optimization points:

shapegen section:
- Enabling fused operators and high-performance compute operators
- PFA operator adaptation
- TorchAir graph mode adaptation
  - DiT graph mode adaptation
  - VAE-decoder graph mode adaptation
- CPU-to-NPU transfer optimization
- DiT-Cache step-level adaptation (step-level refers to the skip/predict step paradigm)

texgen section:
- Multithreaded parallel rasterization
- AI CPU operator migration
- Inpaint computation optimization

## shapegen Performance Optimization
### Fused Operator Optimization
**Optimization reason:** The RMSNorm operator is commonly found in LLM models such as LLaMA, LLaMA2, and Baichuan. Since PyTorch does not provide an interface for the RMSNorm operator, it usually appears as a custom class in the model, with the computation logic defined under the forward function, requiring many operators to run. The computation formula is
```python
x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

**Optimization method:** Replace the torch.rsqrt function in the [hunyuan3ddit.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/3d_vision/Hunyuan3D/hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py) file as follows
```python
'''Replacement part
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale

'''
#After replacement
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        return torch_npu.npu_rms_norm(x, self.scale, epsilon=self.eps)[0]

```

### Replacing with High-Performance Operators
#### Replacing with High-Performance Operators
- **Optimization reason:** In neural networks, GELU is an important activation function inspired by relu and dropout, which introduces the idea of stochastic regularization into activation. To improve the runtime performance of the GELU operator on the NPU, the industry has proposed versions such as FastGelu. The FasterGelu interface is a simplified version of FastGelu that can significantly improve computation performance. The principle is to use Taylor expansion to approximate the computation result.
- **Optimization method:** Modify the GELU function in the `hy3dgen/shapegen/models/denoisers/hunyuan3ddit.py` file, and use `torch_npu.npu_fast_gelu` to replace the operator implementation of `nn.functional.gelu`.
```python
'''Replacement part
class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(x, approximate=self.approximate)
'''
#After replacement
class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return torch_npu.npu_fast_gelu(x)
```

#### FIA Operator Adaptation and Optimization
- **Optimization reason:** The NPU operator corresponding to the officially used `scaled_dot_product_attention` is `npu_fusion_attention`, and `npu_fusion_attention` does not support graph mode. Therefore, if `npu_fusion_attention` is compiled into a static graph, it will be compiled into three core attention operators for computation, resulting in performance degradation.
    - `BatchMatMul`
    - `SoftmaxV2`
    - `BatchMatMul`
- **Optimization method:** Use `torch_npu.fused_infer_attention_score` to replace `torch_npu.npu_fusion_attention`. FIA adapts the FlashAttention operator for both incremental and full inference scenarios, supports graph compilation, and also solves the `Softmax` overflow problem and the memory contiguity problem. Perform the following optimization:
```python
'''Replacement part
scaled_dot_product_attention = nn.functional.scaled_dot_product_attention

def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    x = scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x
'''
# After replacement
def npu_fia(q, k, v, scale):
    attn_mask = None
    batch_size, num_head, seq_len, head_dim = q.shape
    out = torch_npu.fused_infer_attention_score(
        q, k, v, num_heads=num_head, input_layout="BNSD", scale=scale, atten_mask=attn_mask
    )[0]
    return out

def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    x = npu_fia(q, k, v, scale=(1 / math.sqrt(head_dim)))
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

```
### TorchAir Graph Mode Adaptation
- **Optimization reason:** In PyTorch's single-operator execution mode, operators may not only face the host bound of host-side dispatch, but may also miss unrecognized operator fusion opportunities and fail to fully leverage performance. Compared with single-operator mode, graph mode can accelerate model execution efficiency and reduce model memory usage through techniques such as computation graph optimization, multi-stream parallelism, memory reuse, and model sinking. Therefore, torch.compile can be used to enable graph mode to achieve performance optimization. [TorchAir (Torch Ascend Intermediate Representation)](https://www.hiascend.com/document/detail/en/Pytorch/720/modthirdparty/torchairuseguide/torchair_00003.html), as a graph-mode capability extension library of Ascend Extension for PyTorch (torch_npu), provides an Ascend-device-friendly torch.compile graph mode backend. It can help PyTorch networks enable graph mode inference on Ascend NPUs, thereby achieving computation acceleration and performance-level optimization.
- **Optimization method:** In the model inference scenario, enabling graph compilation for the DiT model module and the VAE decoder module can avoid graph compilation failures caused by scalars entering the graph, thereby improving inference performance. At the same time, enable the tiling graph sinking optimization.
```python
#Added part
if args.full_graph:
    import torchair
    import torch._dynamo
    config = torchair.CompilerConfig()
    config.experimental_config.keep_inference_input_mutations = True
    config.experimental_config.frozen_parameter = True
    config.experimental_config.tiling_schedule_optimize = True
    npu_backend = torchair.get_npu_backend(compiler_config=config)
#Graph compilation part     
if args.full_graph:
    pipeline.model = torch.compile(pipeline.model, dynamic=False, backend=npu_backend, fullgraph=True)
    pipeline.vae.geo_decoder = torch.compile(pipeline.vae.geo_decoder, dynamic=False, backend=npu_backend, fullgraph=True)
#Graph mode command      
python minimal_demo_npu.py --full_graph 
```      
### CPU-to-NPU Transfer Optimization
- **Optimization reason:** Transferring from the CPU to the NPU incurs transfer latency, and frequent transfers cause performance degradation, which is especially obvious in the steps that generate timesteps.
- **Optimization method:** Generate timesteps directly on the NPU
```python
'''Replacement part
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )
'''
#After replacement
def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32,device=t.device) / half)
```

### DiT-Cache Step-level Adaptation
- **Basic principle:** DiT-Cache, as a caching framework for accelerating diffusion model inference, reduces redundant forward computation by reusing/predicting existing results. Its acceleration logic can be clearly divided into the Step-level and Block-level paradigms. Step-level determines whether to skip the complete step computation and directly reuse or predict cached results by judging the specific feature differences between different sampling steps and comparing them against a threshold.
- **Typical Step-level methods:** ** Within the Step-level acceleration scope, FBCache and TeaCache are typical cache methods. [TeaCache](https://arxiv.org/pdf/2411.19108) leverages the strong correlation between model input and output to estimate the output difference through the Timestep Embedding (input): it first uses the input to roughly estimate the output change, then corrects the scaling deviation through polynomial fitting, and finally uses the cumulative difference as the criterion to dynamically decide whether to reuse the output cached in the previous step, avoiding redundant computation. [FBCache](https://arxiv.org/pdf/2411.19108) is based on the First Block L1 error, comparing the difference between the output residual of the first Block and the output residual of the first Block in the previous step. If the difference between the first block output error and the previous round's first block output error is less than a specified threshold, the current step computation is skipped, the residual is reused, and the output of the current step is estimated.
- **Typical Block-level methods:** ** Within the Block-level acceleration scope, TaylorSeer is a typical method that replaces caching with a prediction paradigm. [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) addresses the issue in Step-level where the significantly reduced feature similarity in diffusion models causes a notable increase in the errors introduced by feature caching, damaging generation quality. Based on the theoretical foundation that diffusion model features of future timesteps can be predicted from the values of the previous timestep, and on the phenomenon that features change slowly and continuously across timesteps, the TaylorSeer method approximates the higher-order derivatives of features using differential methods and applies the Taylor series to predict subsequent features. Unlike methods that directly reuse cached features, TaylorSeer leverages the continuity of feature changes to predict future features, enabling diffusion models to achieve training-free, high-ratio acceleration without significantly degrading generation quality. 
- **FBCache optimization effect:** Using a 63-image generation task and the UNI3D metric:

    |Threshold|Skip rate|Sampling time (s)|Uni3d-I|Speedup|
    |:---:|:---:|:---:|:---:|:---:|
    |baseline|0|19.28|0.3748|x1.00|
    |0.04|50%|10.19|0.3704|x1.89|
    |0.05|63%|9.62|0.3687|x2.08|
    |0.06|66%|7.80|0.3690|x2.47|
- **TeaCache optimization effect:** Using a 63-image generation task and the UNI3D/ULIP metrics, the results are as follows. At a skip rate of 75%, the accuracy loss is greater than 1%, so a threshold of 0.1 is recommended:
    |Threshold|Skip rate|Sampling time (s)|Uni3d-I|ULIP-I|Speedup|
    |:---:|:---:|:---:|:---:|:---:|:---:|
    |baseline|0|19.28|0.3179|0.2157|x1.00|
    |0.1|48%|10.32|0.3090|0.2050|x1.86|
    |0.2|75%|5.83|0.23908|0.1879|x3.30|
- **TaylorSeer performance optimization effect:** With 100 inference steps, 3 warmup steps, and 0 cutoff steps, multiple rounds of generation were performed, and the average generation time of each round was taken:
   |Derivative order|Skip interval|0|1|2|3|4|5|6|
     |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    ||attention count|4800|2496|1728|1344|1104|960|864|
    ||MLP count|6400|3328|2304|1792|1472|1280|1152|
    ||Predicted steps|0|48|64|72|77|80|83|
    |0|Total time|19.9s|10.24s|7.66s|6.17s|5.14s|4.58s|3.98s|
    |0|Speedup|1.00|1.85|2.60|3.23|3.87|4.35|5.00|
    |0|Performance improvement|0%|46.0%|61.5%|69.0%|74.2%|77.0%|79.9%|
    |1|Total time|20.44s|11.14s|8.02s|6.44s|5.51s|4.92s|4.35s|
    |1|Speedup|1.00|1.84|2.55|3.17|3.71|4.15|4.70|
    |1|Performance improvement|0%|45.5%|60.8%|68.8%|72.9%|75.9%|78.7%|
    |2|Total time|20.72s|11.47s|8.30s|6.67s|5.82s|5.23s|4.64s|
    |2|Speedup|1.00|1.81|2.50|3.11|3.56|3.96|4.47|
    |2|Performance improvement|0%|44.7%|60.0%|67.9%|71.9%|74.8%|77.6%|


    |Derivative/order|Skip rate|Speedup|DiT time|Performance improvement|Uni3d-I|ULIP-I|
    |:---:|:---:|:---:|:---:|:---:|:---:|:---:|
    |baseline|0|1.00|19.26s|0%|0.374|0.221|
    |O1N1|48|1.84|11.14s|45.5%|0.374|0.222|
    |O1N3|72|3.17|6.4s|68.6%|0.372|0.220|
    |O1N5|78|4.15|4.92s|75.9%|0.371|0.217|
    |O2N1|48|1.81|11.47s|44.7%|0.374|0.222|
    |O2N3|72|3.11|6.67s|67.9%|0.373|0.221|
    |O2N5|78|3.96|5.23s|74.8%|0.368|0.217|
- **How to enable:** This code module determines whether to use Cache, the Cache paradigm, and Cache-related parameters by modifying the cache_config.json file, all of which are directly modified in [cache_config.json](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/3d_vision/Hunyuan3D/hy3dgen/cache/cache_config.json). At the same time, the following command can be used to customize the location of cache_config.json
```python
python minimal_demo_npu.py  --cache_config './hy3dgen/cache/cache_config.json'  #cache_config.json location
```      
The meanings of the parameters are as follows
```python
{
        "cache_forward": "NoCache", #Directly set the Cache scheme. Currently supports FBCache/TeaCache/TaylorSeer. NoCache is enabled by default, that is, no ditcache method. Simply replace NoCache according to the following instructions to enable it
        "comment": "choose from FBCache/TeaCache/TaylorSeer, otherwise use NoCache",
        "enable_separate_cfg": false, #CFG serial switch. Adapted to CFG-serial projects such as wan2.2. When enabled, enables dual-branch TeaCache/FBCache, TaylorSeer is not supported. Keep it disabled for non-CFG-serial projects. Hunyuan3D is non-CFG-serial, so it is disabled
        "FBCache":{
                "cache_name": "FBCache",
                "rel_l1_thresh": 0.05, #FBCache threshold. The larger the threshold, the more steps are skipped and the greater the accuracy loss; the performance and accuracy need to be balanced
                "latent": "latent",
                "judge_input": "cache_latent"
        },
        "TeaCache":{
                "cache_name" : "TeaCache",
                "rel_l1_thresh": 0.1, #TeaCache threshold. The larger the threshold, the more steps are skipped and the greater the accuracy loss; the performance and accuracy need to be balanced
                "coefficients": [733.226126,-401.131952,67.5869174,-3.149879,0.0961237896], #TeaCache polynomial fitting, fitted through input and output
                "warmup": 2, #Warmup steps. Force the first few steps to be computed
                "latent": "latent",
                "judge_input": "modulated_inp"
        },
        "TaylorSeer":{
                "cache_name" : "TaylorSeer",
                "n_derivatives": 3, #TaylorSeer derivative setting. The higher the derivative order, the better the theoretical fitting of the original features, but an excessively large derivative may cause overfitting and occupy more memory
                "skip_interval_steps": 4, #TaylorSeer computation interval setting. Note: computation interval = skipped computation steps + 1. The larger the skip computation interval, the more obvious the performance improvement, but the accuracy will decrease
                "warmup": 1, #Warmup steps. Force the first few steps to be computed to avoid skipping computation when feature changes are significant in the early stage
                "cutoff_steps": 1, #Cutoff steps. Force the last few steps to be computed
                "offload": false #Offload switch. When enabled, memory usage can be reduced
        },
        "NoCache":{
            "cache_name" : "NoCache"
    }
    }

```
- **Framework location:** dit_cache is used as a custom library. The dit_cache scheme insertion logic is as follows:
```  
    cann-recipes-embodied-ai
        +--- module #Common optimization module
        |   +--- dit_cache #Core location of the general Dit_Cache scheme
        |       +--- __init__.py #Initialization
        |       +--- cache_method.py #Dit_Cache scheme selection and scheme design
        +--- 3d_vision
            +--- Hunyuan3D
                +--- set_env.sh #Activate the sample environment and enable importing module/dit_cache
                +--- minimal_demo_npu.py #Top-level inference entry, for cache scheme selection, model layer count passing, and forward inference replacement
                +--- hy3dgen
                    +--- cache #Cache adaptation model interface
                        +--- cache_block.py #Dit_Cache replacement module location
                        +--- cache_config.json #Default cache parameter location
                    +--- models
                        +--- denoisers
                            +--- hunyuan3ddit #Core location of the Cache adaptation model
 ```
## texgen Performance Optimization
### Multithreaded Parallel Rasterization
**Optimization reason:** Executing rasterization on the CPU side incurs huge latency. Hunyuan3D texgen projects normal information, pose information, and texture information from 6 different viewpoints separately, performing 18 rasterizations in total. Since rasterization operations and delighting come from different data paths and have no data dependencies, parallel computing can be adopted to hide the rasterization latency.

**Optimization method:** Apply parallelism to the normal computation, the pose computation, and the computation processes in delighting that have no dependencies on each other, improving computation efficiency. Replace the code in [pipelines.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/3d_vision/Hunyuan3D/hy3dgen/texgen/pipelines.py) as follows:

```python
'''Replacement part
images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]
...
normal_maps = self.render_normal_multiview(
selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
position_maps = self.render_position_multiview(
selected_camera_elevs, selected_camera_azims)
...
texture, mask = self.bake_from_multiview(multiviews,
                                         selected_camera_elevs,selected_camera_azims,selected_view_weights,
                                         method=self.config.merge_method)
'''

#After replacement

def _delighting_render_multiview(self,images_prompt,selected_camera_elevs,selected_camera_azims
    ,use_abs_coor=True):
    total_tasks = len(images_prompt) + 2 * len(selected_camera_elevs)
    max_workers = min(20, total_tasks)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        normal_futures = {}
        position_futures = {}

        for idx,(elev,azim) in enumerate(zip(selected_camera_elevs, selected_camera_azims)):
            normal_future = executor.submit(
                self.render.render_normal, elev, azim,
                use_abs_coor = use_abs_coor,return_type = 'pl'
            )
            normal_futures[normal_future] = idx

            position_future = executor.submit(
                self.render.render_position, elev, azim,
                use_abs_coor=use_abs_coor, return_type='pl'
            )
            position_futures[position_future] = idx

    rast_out_maps = [None] * len(selected_camera_elevs)
    normal_maps = [None] * len(selected_camera_elevs)
    position_maps = [None] * len(selected_camera_elevs)
    images_prompt = [self.models['delight_model'](image_prompt) for image_prompt in images_prompt]

    for future in as_completed(list(normal_futures.keys()) + list(position_futures.keys())):
        try:
            result = future.result()
            if future in normal_futures:
                idx = normal_futures[future]
                normal_maps[idx] = normal
            elif future in position_futures:
                idx = position_futures[future]
                position_maps[idx] = result
        except Exception as e:
            print(f"Task failed with error: {e}")
    
    return images_prompt, normal_maps, position_maps

def _bake_from_multiview(self, view, camera_elevs,
                         camera_azim, view_weights,method='graphcut'):
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for idx,(view,camera_elev, camera_azim, weight) in enumerate(zip(
            views, camera_elevs, camera_azims, view_weights)):
            future = executor.submit(self.render.back_project,
                                     view, camera_elev, camera_azim)
            futures.append(future)
        
        project_textures = []
        project_weighted_cos_maps = []
        project_boundary_maps = []

        for future in as_completed(futures):
            project_texture, project_weighted_cos_map, project_boundary_map = future.result()
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_weighted_cos_map)
            project_boundary_maps.append(project_boundary_map)

def __call__(self,mesh,image):
    ...
    # Corresponding to the rasterization and delighting part
    images_prompt, normal_maps, position_maps = self._delighting_render_multiview(images_prompt, selected_camera_elevs,
                                  selected_camera_azims, use_abs_coor=True)
    ...
    # Corresponding to the multiview part
    texture, mask = self._bake_from_multiview(multiviews, selected_camera_elevs
                                              selected_camera_azims, selected_view_weights,
                                              method = self.config.merge_method)
```

### AI CPU Operator Migration
**Optimization reason:** The FP32 data format causes the torch.mean and torch.std operators to run on the AI CPU, degrading performance.

**Optimization method:** Manually convert FP32-type data so that it runs on the AI Core.

```python
# hy3dgen/texgen/pipelines.py
'''Replacement part
texture = torch.tensor(texture_np / 255).float().to(texture.device)
'''

#Replace with
texture = torch.tensor(texture_np / 255).to(torch.float16).to(texture.device)
```

```python
# hy3dgen/texgen/utils/dehighlight_utils.py
'''Replacement part
src_mean, src_stddev = torch.mean(src_flat[:, i].to(torch.float32)), torch.std(src_flat[:, i].to(torch.float32))
target_mean, target_stddev = torch.mean(target_flat[:, i].to(torch.float32)), torch.std(target_flat[:, i].to(torch.float32))
'''

#Replace with
src_mean, src_stddev = torch.mean(src_flat[:, i].to(torch.float16)), torch.std(src_flat[:, i].to(torch.float16))
target_mean, target_stddev = torch.mean(target_flat[:, i].to(torch.float16)), torch.std(target_flat[:, i].to(torch.float16))

'''Replacement part
image_tensor = torch.tensor(np.array(image)/255.0).to(self.device)
'''

#Replace with
image_tensor = torch.tensor(np.array(image) / 255.0).to(torch.float16).to(self.device)
```

### Inpaint Computation Optimization
**Optimization reason:** Some uncolored points still remain in the generated 3D mesh texture, which need to be filled in by smoothing. The source code implements both Python and C++ code. By default, the Python code is used for colored-point smoothing, but C++ has better performance, so a manual replacement is required.

**Optimization method:** Manually replace it with the C++ code.
```python
# hy3dgen/texgen/differentiable_renderer/mesh_render.py
'''Replacement part
from .mesh_processor import meshVerticeInpaint
'''

#Replace with
import mesh_processor

'''Replacement part
texture_np, mask = meshVerticeInpaint(
                   texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
'''

#Replace with
texture_np, mask = mesh_processor.meshVerticeInpaint(
                   texture_np, mask, vtx_pos, vtx_uv, pos_idx, uv_idx)
```

### Rasterization Result Reuse Optimization
**Optimization reason:** Among the 18 rasterization computations in the source code, there are only 6 different input data sets; the remaining 12 computations are redundant, causing performance loss.

**Optimization method:** Cache the result of the first rasterization computation. Whenever the same camera pose is input, the algorithm directly reads the computation result from the cache.
```python
'''Replacement part
rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)
'''
#Replace with
if (self.save_render and 
            (elev, azim, camera_distance) in self.render_result.keys()):
            rast_out = self.render_result[(elev, azim, camera_distance)]

...

self.render_result[(elev, azim, camera_distance)] = rast_out
```
### Rasterization Process NPU Migration
**Optimization reason:** The rasterization in the Hunyuan3D source code has no version that can be executed on the NPU, so the rasterization process is migrated to the CPU side for execution. Unnecessary memory transfers and serial computation incur huge latency. To adapt to the NPU architecture and improve rasterization performance, cann-recipes-embodied-ai introduces render_npu to migrate the rasterization computation process to the NPU for execution.

**Optimization method:** Modify the rasterization algorithm to replace the original approach of traversing mesh vertices with partitioning the 2D plane into blocks. The triangular meshes in each region are merged into one matrix to compute parameters such as depth and barycentric coordinates, and finally the triangle facet corresponding to each pixel is determined. Small-operator computation executed on the NPU side cannot perform rasterization in parallel at the same time.
```python
'''Replacement part
rast_out, rast_out_db = self.raster_rasterize(
            pos_clip, self.pos_idx, resolution=resolution)
'''
#Replace with
if self.use_render_npu:
    rast_out, _ = render_npu_rasterize(
                    resolution, self.pos_idx, pos_clip
    )
else:
    rast_out, _ = self.raster_rasterize(
                    pos_clip, self.pos_idx, resolution=resolution
    )
```
## Performance Optimization Metrics
### shapegen Performance Optimization Metrics
This solution uses a single Atlas 800I A2 inference product. Using the sample data provided by Hunyuan3D 2.0 (`assets/example_images/004.png`), with the number of diffusion steps set to num_inference_steps=100 (lines 89 and 111 of [minimal_demo_npu.py](https://gitcode.com/cann/cann-recipes-embodied-ai/blob/master/3d_vision/Hunyuan3D/minimal_demo_npu.py)), the performance metrics are as follows
|Enabled method|DiT diffusion time (s)|VAE generation time (s)|
|:---:|:---:|:---:|
|baseline|21.63|17.98|
|Operator optimization + transfer optimization|20.07|18.71|
|Operator optimization + graph mode compilation|19.00|16.68|

### texgen Performance Optimization Metrics
This solution uses a single Atlas 800I A2 inference product. Using the sample data provided by Hunyuan3D 2.0 (`assets/example_images/004.png`), with 20000 input mesh faces and face reduction enabled (--face_reduce), the performance metrics are as follows
|Enabled method|texgen runtime (s)|
|:---:|:---:|
|baseline|59.83|
|Rasterization parallelization|35.43|
|Rasterization parallelization + operator optimization|30.46|
|NPU rasterization + operator optimization + Inpaint optimization + rasterization result reuse|32.86|
|Rasterization parallelization + operator optimization + Inpaint optimization + rasterization result reuse|26.52|

<!-- static-meta
{
  "sidebarTitle": "Inference Optimization",
  "hardware": ["atlas-a2"],
  "cannFeatures": ["npugraph"],
  "cover": {"type": "auto"}
}
-->
