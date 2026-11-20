# DeepSeek-R1 Model RL Training Optimization Practice Based on the verl Frontend & A3 Cluster

## 1. Introduction

### 1.1 Background

After the release of DeepSeek-R1, RL training of large models has received widespread attention. Major vendors and the open-source community have carried out extensive RL training practices, striving to build more efficient RL training systems. This article, based on the open-source verl framework, combined with the MindSpeed and vLLM Ascend frameworks, achieves high-throughput performance on Atlas A3 SuperPoD super-nodes through in-depth optimization and adaptation.

The open-source frameworks involved in this article are shown in the following table.

| **Framework Name**  | **Description**  |
| ----- | ---- |
| [verl](https://github.com/volcengine/verl) | Open-source LLM reinforcement learning framework that combines the advantages of single-controller (control flow) and multi-controller (computation flow), simplifying the implementation of RL algorithms (such as PPO, GRPO). Its training supports FSDP, FSDP2, and Megatron-LM, and its inference supports vLLM, SGLang, and Hugging Face Transformers. |
| [vLLM](https://github.com/vllm-project/vllm) | Open-source engine providing high-throughput and memory-efficient inference services for large language models, using PagedAttention technology to efficiently implement KV cache memory management and supporting numerous inference features. |
| [vLLM Ascend](https://github.com/vllm-project/vllm-ascend) | The Ascend NPU adaptation plugin for vLLM, supporting many optimizations on NPUs. Some optimizations from this practice were contributed to this open-source repository. |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Ultra-large-scale language model training framework supporting multiple distributed parallel strategies. |
| [MindSpeed](https://gitcode.com/Ascend/MindSpeed) | Large-model training acceleration library for NPUs, providing basic functionality adaptation of the Megatron-LM framework to NPUs as well as some NPU-affine optimization features. |
| [MindSpeed RL](https://gitcode.com/Ascend/MindSpeed-RL) | Reinforcement learning acceleration framework based on the Ascend ecosystem; this optimization practice draws some references from it. |


### 1.2 Technology Path Selection

#### 1.2.1 Why Choose Full Co-residency Deployment?

Each training step of DeepSeek R1 can be roughly divided into three compute-intensive phases: inference generating complete Q&A data (rollout/generate) -> the reference model computing reference probability distributions (ref) -> the training model computing loss and performing backward updates (update). For DeepSeek R1, a large model with 671B parameters, all three phases require substantial computational resources. Therefore, deploying the three models — rollout/ref/update — on shared cards and time-sharing computational resources is a simple and efficient deployment strategy.

In addition, for the rollout part we did not adopt the PD separation deployment strategy commonly used in inference (i.e., instantiating and deploying the Prefill phase and Decode phase on different machine resources separately), because the inference Prefill part accounts for only a small proportion of the time in the entire RL training, and the optimization benefit is not high.

#### 1.2.2 Why Choose vLLM Ascend?

vLLM Ascend decouples the framework from hardware dependencies through a hardware plugin mechanism. Users can deploy models on Ascend NPUs without modifying core code. The code is fully open source and continuously adapted for new features. Currently, vLLM Ascend already supports vLLM's V1 mode and has been optimized for DeepSeek R1's TorchAir graph mode, which is essential for optimizing the rollout part.

### 1.3 Overview of verl Adaptation and Optimization Practice
--------------------------

We performed NPU adaptation on the verl open-source code and carried out some key performance optimizations.

The core adaptations mainly include:

- Training engine: enable Megatron-LM through MindSpeed on NPUs to obtain better NPU training performance.

- Inference engine: adapt vLLM Ascend graph mode and large-EP inference; since the sleep feature is not yet supported in multi-machine scenarios, offload and onload, init\_cache and free\_cache need to be manually implemented.

This practice performed performance optimization on verl in the following four aspects:

![](../figures/image2.png)

### 1.4 Detailed Performance Results

Based on the Atlas A3 SuperPoD super-node 128-card cluster, this practice achieved the following performance results in two scenarios: loading real weights and randomly initialized weights.

#### 1.4.1 Loading Real Weights
> Performance update note: the following is the performance with training-side recomputation of 4 layers. To ensure stable long-run training after loading weights, the training recomputation was changed to 8 layers; after about 13 training steps, the throughput can also improve from the initial 112 TPS/card to 120+ TPS/card.

Loading real DeepSeek-V3 weights, limiting the maximum inference length to 3K and enabling partial performance optimizations, we verified the training-inference performance under actual load, achieving a system throughput of 123.16 TPS/card<a id="ref1"></a><a href="#foot1">[1]</a>.

<table border=1px">
<tr>
<th><strong>Model</strong></th>
<th><strong>Deployment Mode</strong></th>
<th><strong>Deployment Environment</strong></th>
<th><strong>RL Framework</strong></th>
<th><strong>Inference Engine</strong></th>
<th><strong>Training Engine</strong></th>
<th><strong>Dataset</strong></th>
<th><strong>step</strong></th>
<th><strong>Average Input Length</strong></th>
<th><strong>Average Output Length</strong></th>
<th><strong>batchsize</strong></th>
<th><strong>Sampling Count</strong></th>
<th><strong>rollout Deployment</strong></th>
<th><strong>actor Deployment</strong></th>
<th><strong>ref Deployment</strong></th>
</tr>
<td>DeepSeek-R1 671B</td>
<td>Full co-residency</td>
<td>128 * A3</td>
<td>verl</td>
<td><p>vLLM+vLLM Ascend</p></td>
<td><p>Megatron+MindSpeed</p></td>
<td>deepscaler</td>
<td>2</td>
<td>77.8</td>
<td>1576.6</td>
<td>512</td>
<td>16</td>
<td><p>256die</p><p>2 DP instances</p>
<p>DP64TP2EP128</p></td>
<td><p>256die</p>
<p>TP4EP8PP8</p></td>
<td><p>256die</p>
<p>TP4EP8PP8</p></td>
</tr>
</table>


<table border="1">
<tr>
<th colspan="5"><strong>Data Generation (s)</strong></th>
<th><strong>Training (s)</strong></th>
<th colspan="2"><strong>Training-Inference Switching (s)</strong></th>
<th><strong>Total Time (s)</strong></th>
<th><strong>Inference Throughput<br />
(tokens/s/card)</strong></th>
<th><p><strong>Training Throughput</strong></p>
<p><strong>(tokens/s/card)</strong></p></th>
<th><p><strong>System Throughput</strong></p>
<p><strong>(tokens/s/card)</strong></p></th>
</tr>
<tr>
<td colspan="5" align="center">523.38</td>
<td align="center">319.3</td>
<td colspan="2" align="center">17</td>
<td rowspan="3" align="center">859.68</td>
<td rowspan="3" align="center">271.05</td>
<td rowspan="3" align="center">331.6</td>
<td rowspan="3" align="center">123.16</td>
</tr>
<tr>
<td>rollout (s)</td>
<td>old_log_prob (s)</td>
<td>ref prefill (s)</td>
<td>reward (s)</td>
<td>adv (s)</td>
<td>update(s)</td>
<td>reshard(s)</td>
<td>offload etc. (s)</td>
</tr>
<tr>
<td align="center">373.65</td>
<td align="center">0</td>
<td align="center">144.5</td>
<td align="center">5.0</td>
<td align="center">0.21</td>
<td align="center">319.3</td>
<td align="center">11.5</td>
<td align="center">5.5</td>
</tr>
</table>

#### 1.4.2 Loading Random Weights

With randomly initialized weights, a fixed inference length of 3K, forced expert load balancing, and all performance optimizations enabled, we tested the peak performance under ideal conditions, achieving a system throughput of 250.2 TPS/card<a id="ref2"></a><a href="#foot2">[2]</a>.

<table border=1px">
<tr>
<th><strong>Model</strong></th>
<th><strong>Deployment Mode</strong></th>
<th><strong>Deployment Environment</strong></th>
<th><strong>RL Framework</strong></th>
<th><strong>Inference Engine</strong></th>
<th><strong>Training Engine</strong></th>
<th><strong>Dataset</strong></th>
<th><strong>step</strong></th>
<th><strong>Average Input Length</strong></th>
<th><strong>Average Output Length</strong></th>
<th><strong>batchsize</strong></th>
<th><strong>Sampling Count</strong></th>
<th><strong>rollout Deployment</strong></th>
<th><strong>actor Deployment</strong></th>
<th><strong>ref Deployment</strong></th>
</tr>
<td>DeepSeek-R1 671B</td>
<td>Full co-residency</td>
<td>128 * A3</td>
<td>verl</td>
<td><p>vLLM+vLLM Ascend</p></td>
<td><p>Megatron+MindSpeed</p></td>
<td>gsm8k</td>
<td>5</td>
<td>75</td>
<td>3039.8</td>
<td>512</td>
<td>16</td>
<td><p>256die</p>
<p>DP128TP2EP256</p></td>
<td><p>256die</p>
<p>TP4EP8PP8</p></td>
<td><p>256die</p>
<p>TP4EP8PP8</p></td>
</tr>
</table>




<table border="1">
<tr>
<th colspan="5"><strong>Data Generation (s)</strong></th>
<th><strong>Training (s)</strong></th>
<th colspan="2"><strong>Training-Inference Switching (s)</strong></th>
<th><strong>Total Time (s)</strong></th>
<th><strong>Inference Throughput<br />
(tokens/s/card)</strong></th>
<th><p><strong>Training Throughput</strong></p>
<p><strong>(tokens/s/card)</strong></p></th>
<th><p><strong>System Throughput</strong></p>
<p><strong>(tokens/s/card)</strong></p></th>
</tr>
<tr>
<td colspan="5" align="center">436.86</td>
<td align="center">328.8</td>
<td colspan="2" align="center">35.63</td>
<td rowspan="3" align="center">796.8</td>
<td rowspan="3" align="center">686.8</td>
<td rowspan="3" align="center">607.2</td>
<td rowspan="3" align="center">250.2</td>
</tr>
<tr>
<td>rollout (s)</td>
<td>old_log_prob (s)</td>
<td>ref prefill (s)</td>
<td>reward (s)</td>
<td>adv (s)</td>
<td>update(s)</td>
<td>reshard(s)</td>
<td>offload etc. (s)</td>
</tr>
<tr>
<td align="center">318.27</td>
<td align="center">0</td>
<td align="center">104.7</td>
<td align="center">13.7</td>
<td align="center">0.188</td>
<td align="center">328.8</td>
<td align="center">11.18</td>
<td align="center">24.45</td>
</tr>
</table>


## 2. Framework Adaptation

### 2.1 verl + MindSpeed Adaptation

The current verl version natively supports the MindSpeed training backend. Through the repatch mechanism embedded in the MegatronWorker initialization phase, it implements the conversion and parsing of verl configuration parameters to MindSpeed configuration parameters, as well as the replacement of MindSpeed's NPU-affine training implementations within Megatron-LM. Based on this mechanism, this practice configured MindSpeed's various functional and performance optimization features in the verl launch script, and completed functional verification of the DeepSeek-R1 model reinforcement learning training phase on the NPU environment.

### 2.2 verl + vLLM Ascend Adaptation

This practice builds on the Ascend-affine adaptation of the DeepSeek R1 model structure performed by the vLLM community's vLLM Ascend project, achieving excellent performance. To make verl compatible with vLLM Ascend, this practice modified some interface parameters, with the modifications mainly concentrated in the [vllm\_rollout\_spmd.py](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_rl/deepseek/verl_patches/workers/vllm_rollout/vllm_rollout_spmd.py) file used to initialize the vLLMRollout object.

| **Main Operation**   | **Detailed Description**  |
| ----- | ---- |
| Manually enable the vLLM Ascend patch | The patch includes adjustments by vLLM Ascend to some classes, functions, and certain variable types of vLLM. The current version requires manually importing the patch before `import vllm`.   |
| Initialize the DP communication domain   | Configure VLLM_DP_MASTER_PORT to ensure each DP group has an independent communication port for intra-group process synchronization, and specify the DP group master node IP via VLLM_DP_MASTER_IP to adapt to dynamic cluster environments.   |
| Disable sleep mode   | The current sleep mode on NPUs has a failure problem in multi-machine scenarios and cannot release device memory; if not explicitly disabled, weight and kv_cache will be managed by vLLM's custom memory pool and cannot be manually offloaded via torch_npu later.  |
| Configure max_num_batched_token       |    After loading real weights, the default value of 8192 for max_num_batched_token can cause OOM on some cards under imbalanced expert load, so it is necessary to set max_num_batched_token=1024 and enable_chunked_prefill=True.      |

Enabling vLLM Ascend optimization features requires passing the following additional\_config parameters during vLLM initialization:

```
additional_config={
    "torchair_graph_config": {
        "enabled": VLLM_ENABLE_GRAPH_MODE,    #  Enable graph mode
        "use_cached_graph": False,
        "graph_batch_sizes_init": False,
        "graph_batch_sizes": [max_num_seqs],  #  Configure the batch size of each DP rank in graph mode
        "enable_multistream_mla": True,       #  In graph mode, enable MLA multi-stream, placing MLA vector operations on another stream.
        "enable_multistream_moe": True,       #  In graph mode, enable MoE multi-stream, placing shared expert computation on another stream
        "enable_view_optimize": False,        #  In graph mode, eliminate some transpose operations
        "enable_kv_nz": True,                 #  In graph mode, enable the NZ layout of kvcache
    },
    "ascend_scheduler_config": {
        "enabled": True,                      #  Use vLLM Ascend's scheduler
    },
    "refresh": True
},

```

- Enable vLLM Ascend's optimization features, including enabling graph mode, enabling MLA multi-stream and MoE multi-stream, and enabling the NZ format of kv\_cache.

- Explicitly set enable\_expert\_parallel=True to enable vLLM's EP parallelism.

The implementation principles of these optimization items can be found in the [Performance Optimization: Inference](#6-Performance-Optimization:-Inference) section.


## 3. Performance Optimization: Inter-Model Scheduling

### 3.1 Slow Training Task Dispatch Problem
----------------------

#### 3.1.1 Abnormal Communication Phenomenon on the Training Side

During debugging, we found that the performance of the verl training side (compute\_ref\_log\_prob, update\_actor) was not as expected. Profiling of these two phases showed that the first communication on some cards took extremely long: the first communication of compute\_ref\_log\_prob took about 50s+, and the first communication of update\_actor took about 90s+.

As shown in the figure below, in 256-Die training, the first communication time of card 0 in ref profiling is abnormal.

![](../figures/image3.png) 

#### 3.1.2 Excessive Start Time Differences of Training-Side Tasks

We added detailed timestamps before and after generate\_sequences, compute\_ref\_log\_prob, and update\_actor on each card, and observed that the end times of generate\_sequences and compute\_ref\_log\_prob are consistent across cards, but the start times of compute\_ref\_log\_prob and update\_actor are inconsistent across cards.

The specific phenomenon is that card 0 always starts earliest, the last card starts latest, each card starts serially, and the maximum start time difference on the training side is very close to the time of the first extremely long communication in profiling. As shown in the figure below, in 256-Die training, the update start time difference between card 0 and card 255 can reach 96 seconds.

![](../figures/image4.png)

#### 3.1.3 Analysis of verl Controller Logic

At this point, we can conclude that the poor performance of the verl training side is due to significant time differences in task dispatch, which may be related to verl's controller logic. According to the func\_generator function in [base.py](https://github.com/volcengine/verl/blob/main/verl/single_controller/ray/base.py), after verl calls a task interface in the CPU-side main process, it actually executes the following 3 steps:

- dispatch\_fn: the main process splits the held task input data into multiple parts along the batch dimension according to the task DP size.

- execute\_fn: the main process dispatches tasks to each worker process.

- collect\_fn: the main process collects the output results from each worker and concatenates them along the batch dimension into a complete result.

dispatch\_fn and collect\_fn respectively perform the split of input tensors and the concat of output tensors, which in practice take very little time. Further analyzing execute\_fn, which corresponds to the execute\_all\_async function in [base.py](https://github.com/volcengine/verl/blob/main/verl/single_controller/ray/base.py), as shown in the figure below. The code logic here matches the previously observed phenomenon of each worker serially starting tasks. Adding timestamps before line 586 can confirm that it is the serial task dispatch that causes time differences in task start times across workers, which in turn leads to the poor performance exhibited on the training side.

![](../figures/image5.png)

 

#### 3.1.4 Analysis Conclusion on the Performance Problem

**Ray dispatching a task to a worker involves at least three steps: input data serialization, data transmission, and output data serialization. Therefore, when the input tensor data volume of a task is large, the task dispatch time across many workers accumulates continuously, creating a training performance bottleneck**.

In addition, compared with training-side tasks, the dispatch time of the generate\_sequences task is within one second, which also proves from the side that task dispatch time is related to data volume. As shown in the figure below, the gen input data volume is far smaller than the training-side input data volume, for several reasons:

- The DP used by gen when partitioning data in dispatch\_fn equals the total number of cards (e.g., 256), far larger than the training-side DP size (e.g., 8).

- In GRPO training, the input batch\_size of gen is not amplified by the n\_sample multiplier.

- The input of gen is only the prompt, while the sequence length of some training-side inputs equals prompt+response.

![](../figures/image6.png)


#### 3.1.5 Summary of verl Controller Logic

Based on the above analysis, for a single worker, the inter-task scheduling logic of the verl controller can be described by the following flow chart:

![](../figures/image7.png)

The following sequence diagram shows the performance bottleneck caused by task dispatch in the verl framework more clearly.

![](../figures/image8.png)


### 3.2 Performance Optimization: D2D Data Transfer

#### 3.2.1 Optimization Approach

Based on the previous analysis, the goal of this practice is to improve the task dispatch performance of the verl training side, so that in large-scale training all workers start executing tasks as simultaneously as possible, reducing the resource waste caused by workers waiting for each other, and improving training-side throughput.

In addition, this practice attempted to change ray task dispatch in verl to multithreaded parallelism, but measurements showed it could not meet the requirements:

- The ref computation task dispatch could only be reduced from 66 seconds to 24 seconds, making it difficult to fully eliminate task dispatch time.

- Multithreaded parallel task dispatch can cause tensor data serialization errors on some workers, which are difficult to resolve.

After reviewing the design directions of optimization schemes, this practice decided to focus mainly on the following two goals.

- **Goal 1: Reduce the time consumed in transferring data between tasks.**

According to the previous analysis, verl's poor performance in large-scale training on NPUs is mainly because serial task dispatch includes serialization, ray transmission, and deserialization of each worker's input data. But in fact, the input data of most training-side tasks is generated on the device side by preceding tasks; sending the data to the host side, adjusting it in the main process, and then sending it back to the device side is unreasonable from a performance perspective.

We can consider using the AlltoAllV collective communication to directly implement D2D data transfer to accelerate the data flow between verl tasks, i.e., the original logic of the main process collecting and dispatching data is directly implemented through AlltoAllV during device-to-device communication. In addition, because the actor model and the ref model in our verl training use the same parallel strategy and the DP
rank layout is completely identical, the input/output data used by ref is completely identical to the input data required by update, so the collective communication operations can even be eliminated.

- **Goal 2: Task control maintains verl's single-controller logic.**

verl uses Ray for scheduling at the upper layer, transferring data between workers to implement the control flow of RL algorithms, which is the so-called single controller. The advantage of verl's single controller is its clear architecture, simplifying the implementation of different RL algorithms within the framework: one only needs to write the worker methods and then call them at appropriate points in the main process training flow.

However, data being directly communicated and transferred between multiple workers belongs to multi-controller logic, which would complicate the implementation of inter-task control logic in RL training and conflict with the verl framework's original design philosophy.

We can consider decoupling verl's control flow from its data flow: accelerate the data flow as much as possible with D2D direct transfer, while keeping the control flow logic unchanged. Using a cache approach, data that subsequent tasks will also use is cached in place on the device side, and later tasks pull it from the cache on demand. Other data that must be returned to the CPU side or must be obtained from the CPU side retains its original logic coupled with the control flow.

#### 3.2.2 Overall Flow

We use TensorCache to manage tensors shared between preceding and subsequent tasks on the device, thereby accelerating transfer speed and reducing task dispatch time. For a single worker, the inter-task flow in the new scheme is roughly as follows:

![](../figures/image9.png)

The following sequence diagram shows the benefit sources of the scheme: mainly that after data flow acceleration, the time for tasks to be dispatched to workers is greatly reduced, and the task start time differences between workers are greatly reduced.

![](../figures/image10.png)

In addition, to minimize the overhead of data transfer between the host-side main process and device-side worker processes, the data flow in RL training needs to be carefully optimized. Examining the three main tasks generate\_sequences, compute\_ref\_log\_prob, and update\_actor, verl's original data flow is as follows, where all input and output tensors pass through the main process:

![](../figures/image11.png)

To maximize the acceleration of the data flow, the main process's training main-flow code needs to be modified accordingly, ensuring that only necessary tensors are transferred between host and device (for example, host-side computation output as device-side computation input, or device-side computation output as host-side computation input), ultimately achieving the following data flow:

![](../figures/image12.png)

#### 3.2.3 TensorCache Design Details

The TensorCache class is designed as follows. Each worker instance holds a TensorCache, i.e., each device has a TensorCache managing tensor data.

![](../figures/image13.png)

| **Attribute or Method**   | **Description**  |
| ----- | ---- |
| mini_bs | The batchsize of a single training step in GRPO, used together with the batch-axis size of tensors in the cache to compute their src DP size.   |
| tensor_cached   | The dictionary of tensors cached in the cache, where the key is the tensor's name.   |
| cached_tensors()  | Given DataProto data, cache the tensors within it. Certain tensors can be specified not to be cached, and certain tensors can be specified to be cached while not being deleted (they will be passed to the CPU side for use later).  |
| get_cached_tensors() | Fetches the tensor with the given name from the cache and returns a DataProto instance. |
| clear() | Clears the cache, generally called after update_actor finishes. |


get\_cached\_tensors() may involve AlltoAllV communication, specifically in two cases:

- The current task and the preceding task have the same DP rank layout

  This situation occurs when the update task uses the cache updated by the ref computation task. At this point, the data with the same name used by each card in the two tasks is completely identical, so the data in the cache can be used directly without communication.

- The current task and the preceding task have different DP rank layouts

  This situation occurs when the ref computation task uses the cache updated by the generate task. At this point, the tensors in the cache need to be resharded along the batch dimension, which is equivalent to using AlltoAllV communication to accomplish the dispatch+collect logic that verl's main process originally performed. Taking 8 cards as an example, the communication effect that AlltoAllV needs to achieve is as follows.

  ![](../figures/image14.png)

 

#### 3.2.4 Performance Gains 

After the optimization was completed, verification was carried out on the Atlas A3 128-card environment, using randomly initialized weights, a prompt length of 0.075K, and a response length of 3K. The comparison of training-side start time before and after optimization is as follows. It can be seen that the task dispatch time on the training side is basically eliminated.
|  |  Max start time difference before optimization  | Max start time difference after optimization    | Benefit    |
| ----- | ---- |---- | ---- |
| compute_ref_log_prob |  ~57s   | <1s   | ~98.2%   |
| update_actor  | ~73s   | ~5s    | ~93.2%  |

In another experiment of the same scale, keeping other variables unchanged, comparing the step-3 training performance data printed to the screen, the end-to-end benefits were verified as follows, and the results met the expected goals.

|  |  Single-phase time before optimization | Single-phase time after optimization    | Benefit    |
| ----- | ---- |---- | ---- |
| timing_s/ref（s） |  156.804  | 105.061 | ~33.0% |
| timing_s/update_actor（s） | 400.891 | 334.152 | ~16.6% |
| perf/throughput（tps） | 208.5 | 238.46 | ~14.4% |

**However, this scheme currently still has some usage limitations**:

- Currently, balance\_batch and pack sequence enabling are not supported. Supporting balance\_batch would require TensorCache to cooperate with the CPU side in adjusting the order and content of samples in the cache, which may be relatively complex.

- Currently only the scenario where Actor, Rollout, and Ref share worker instances is considered; the communication pattern in other scenarios may need adjustment.

- Currently only GRPO on-policy is supported; off-policy may require detailed adjustments to batch\_size.

 

## 4. Performance Optimization: Training-Inference Switching

Large-model RL training involves two states: training and inference. When the model completes a training update, the subsequent inference needs to use the updated model parameters. Since training and inference typically use different parallel strategies, the model parameters need to be repartitioned and adjusted when passed to the inference side, i.e., reshard. On the other hand, because this practice chose the strategy of deploying the three models — Actor, Rollout, and Ref — on fully shared cards, the training-inference switching also requires offloading the previous model to the host side and then loading the subsequently used model to the device side. Therefore, this practice optimized both reshard and offload.

### 4.1 reshard Optimization

#### 4.1.1 Expert Parameter Targeted Routing Scheme Based on AlltoAllV

As a typical MoE architecture model, DeepSeek R1's expert layer (Expert Layer) parameters account for over 70% of the total parameter scale, and its dynamic routing mechanism makes the shard migration of expert parameters (EP) one of the key bottlenecks in the training-to-inference phase. To address the expert parameter migration requirements of the DeepSeek R1 model, the scheme takes communication topology mapping as its core and designs a targeted tensor routing mechanism based on the AlltoAllV operation, achieving zero-redundancy migration of training parameters to the inference architecture.

**Original Scheme**

By identifying parameter names containing the ".mlp.experts.linear\_fc" keyword, the EP routed expert parameter migration logic is triggered. First, an all\_gather of expert parameters is executed within the ep\_group scope to obtain all routed expert parameters (256 experts), and then the inference side's load\_weight is used to load the expert parameters needed by a single card (1 expert). The figure below uses a total of 8 routed experts as an example to illustrate this process. It is assumed that the parameters iterated in the current round have been distributed to each PP rank through Broadcast, and parameters within the same pp group are identical.

![](../figures/image15.png)

Although implementing parameter migration through full aggregation is simple in code, it has significant limitations in large-scale heterogeneous environments:

- Communication Redundancy

  The all\_gather operation of ep\_group retains full-broadcast characteristics. Taking the training EP4 → inference EP8 in the figure above as an example, the original logic still sends the local shard to all members of the ep\_group (e.g., full transmission to all 4 ranks under EP4) rather than only to the target inference ranks. Specifically, when training rank 0 only needs to send shards to inference rank 0 and rank 1, the full broadcast additionally transmits invalid data to rank 2 and rank 3, resulting in 50% bandwidth waste, and the redundancy ratio climbs further as ep\_size increases.

- Memory Peak Amplification

  The all\_gather operation of ep\_group aggregates the current Tensors of all experts onto each rank. In 671B model training, fully aggregating 256 routed expert W1 weights of 7168×4096 causes the per-card incremental memory to reach 0.54G\*256=14G, greatly increasing the risk of OOM.

**Optimized Scheme**

To address the redundant migration of routed experts analyzed above, we designed a targeted routing mechanism based on AlltoAllV communication, achieving zero-redundancy migration of training parameters to the inference architecture. The scheme is shown in the following figure:

![](../figures/image16.png)

Based on the expert ownership of the training-side global rank, a sparse send route is generated (taking the 8 send buffers in the example; 0-7 in the figure above respectively represent 8 expert parameters):
- rank0 (training Group0): holds experts 0 and 1, which need to be targeted to inference rank0 and rank1 respectively, so the send table is [0, 1, ∅, ∅] (the first two positions are valid experts, and the last two positions are empty).

- rank4 (training Group1): has no experts to send (its shards have been transferred by other ranks in the same group), so the send table is [∅, ∅, ∅, ∅].
- rank6 (training Group1): holds experts 4 and 5, which need to be targeted to inference rank4 and rank5 respectively, so the send table is [4, 5, ∅, ∅] (the first two positions are valid experts, and the last two positions are empty).

Construction of the inference-side receive route table:

- rank0: only needs to receive the experts from rank0 of the same training group, so the receive table is [0, ∅, ∅, ∅].
- rank4: needs to receive the experts from rank6 in training Group1, so the receive table is [∅, ∅, 6, ∅].
- rank6: receives the experts from rank7 in training Group1, so the receive table is [∅, ∅, ∅, 7].

Taking the single-card routed expert w1 parameter as an example, as shown in the following table, the incremental memory relative benefit brought by reshard reaches 99.6%. By the same calculation, the communication volume relative benefit also reaches 99.6%.

|      |  Experts per layer    | Inference-side routed expert w1 parameter count per card per layer    | Inference EP   | Incremental memory per card   |
| ---- | ---- | ---- | ---- | ---- |
|  Before optimization    |  256    |  2*(7168*2048)*2=0.0547 GB    | 256     |  AllGather requires preparing space for all experts: 0.0547*256=14G    |
|  After optimization    |  256    |  2*(7168*2048)*2=0.0547 GB    |  256    |  AlltoAllV requires at most space for one expert: 0.0547*(256/256)=0.0547G    |
| Relative benefit   |   -   |   -   |   -   |  99.6%    |


**Constraints**

- Currently only the scenario where Megatron enables "moe\_tp\_extend\_ep" is supported.

- Training EP must be divisible by inference EP.

#### 4.1.2 PP-Domain Broadcast Communication Optimization

**Original Scheme**

Currently, the DeepSeek R1 inference side does not use the PP parallel strategy, so the training model parameters need to be aggregated within the PP domain. The original Reshard adopts a parameter-traversal mode, following the "broadcast first, then aggregate" processing logic:

1. First, parameters are distributed to each training PP rank through the Broadcast operation, so that every card has parameter shards of all layers of the model.

2. Then, subsequent flows such as TP-domain AllGather are executed on the parameter shards within each layer, enabling the inference side to collect all required parameters.

The above process is shown in the following figure. In this mode, parameters with the same name undergo repeated TP-domain AllGather on different training PP ranks, bringing significant communication redundancy overhead.

![](../figures/image17.png)

**Optimized Scheme**

For TP-partitioned parameters, we refactored the reshard flow: first perform AllGather within the TP domain to complete the local aggregation of parameter shards, then execute one AlltoAll within the PP domain to broadcast the parameters across the PP domain.

The flow of the new scheme is shown in the following figure. Its advantages mainly come from the following two points:

1. The order adjustment of TP domain first and PP domain second reduces the total communication volume and the number of communications.

2. Originally, one Broadcast was performed per parameter within the PP domain; after optimization, a single AlltoAll can complete the broadcast of PP parameters, making communication more efficient.

![](../figures/image18.png)

Note that some parameters currently do not use this new scheme: for example, ".mlp.experts.weight" uses the expert AlltoAllV optimization; input/output-related parameters such as "embedding" and "output\_layer", as well as the dense layers of the MLP, will be iteratively optimized later.

**Communication Comparison Before and After Optimization**

The communication comparison before and after optimization is shown in the following table:


|      |  Total model layers    | Complete parameter count per layer    | Number of TP parameters per layer    |  Training PP   |  Training TP    | Step1 communication volume (measured by per-card received amount), number of communications     |  Step2 communication volume (measured by per-card received amount), number of communications    |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  Before optimization    |   n   | m     |   k   |  p   |  t    | PP-domain broadcast collects all parameters, communication volume approximately: n\*m/t <br>Number of communications: n\*k   |   TP-domain full aggregation receives all parameters, communication volume approximately: n\*m <br>Number of communications: n\*k   |
|  After optimization    |   n   |  m    |  k    |   p   |  t    |  TP-domain parameter full aggregation, communication volume approximately: n\*m/p <br>Number of communications: n\*k/p    |  PP-domain AlltoAll collects all parameters, communication volume approximately: n\*m <br>Number of communications: n\*k/p  |
|  Communication volume change    |   -   |   -   |   -   |   -   |   -   |  Usually p is greater than t, e.g., p=8, t=4, so after optimization the first-step communication volume is halved    |   No change   |
|  Number of communications change    |   -   |   -   |   -   |  -   |   -   |  TP-domain AllGather communication count: reduced by (p-1)/p * n*k times    |  PP-domain communication count: reduced by (p-1)/p * n\*k times    |



### 4.2 offload Optimization

Performing offload on NPUs involves a certain amount of adaptation work. Because vLLM's wake\_up and sleep functions used in verl still have problems in NPU multi-machine testing, these functions are currently not enabled. This practice references the MindSpeed-RL implementation, replacing the wake\_up function with onload\_model\_weight and init\_cache\_engine, and replacing the sleep function with offload\_model\_weight and free\_cache\_engine.

This practice further made some detailed adjustments to offload to optimize performance, mainly including the following two points:

- When reshard is performed immediately after training, the training model weights are still needed, so there is no need to offload the Megatron weights after training, nor to reload them before inference; therefore these two redundant offload/onload operations can be omitted.

- The inference init\_cache\_engine operation can be postponed as much as possible, for example, executed after reshard finishes and after offloading the Megatron training weights. This can effectively reduce peak memory, thereby improving vLLM memory utilization and helping to improve inference performance. 

The comparison of the offload/onload flows before and after adjustment is shown in the following figure:

![](../figures/image19.png)

## 5. Performance Optimization: Training

### 5.1 Reusing MindSpeed Training Optimizations

This practice integrated verl with MindSpeed, so the training side mainly reuses many of MindSpeed's optimization techniques in memory, communication, and computation, including the Ascend Swap Optimizer and enabling fused operators, among others. The corresponding configuration script excerpt is shown below. For detailed principles of these optimization techniques, please refer to the [MindSpeed official documentation feature introduction section (Chinese)](https://gitcode.com/Ascend/MindSpeed#特性介绍); they will not be elaborated here.
```shell
+actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
+actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
+actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
+actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True
```

### 5.2 old_log_prob-Free Computation

For on-policy training algorithms such as GRPO, the old\_log\_prob computation result is numerically identical to the log\_prob obtained from the update\_actor forward computation (because the training model parameters are the same), so log\_prob.detach() can be directly used instead of old\_log\_prob, thereby saving the time of one training model forward computation.

![](../figures/image21.png)

The current verl does not support this feature. This practice supplemented verl with the old\_log\_prob-free computation feature. Setting config.actor\_rollout\_ref.actor.recompute\_old\_log\_prob to False enables this optimization.

## 6. Performance Optimization: Inference

The data generation inference phase has always been the top priority of RL training performance optimization. In long-sequence scenarios, the inference phase can account for more than 80% of the total time; in the current DeepSeek-R1 1K-to-3K scenario, the inference phase also accounts for nearly half of the total time. Therefore, this practice focused on performance optimization across framework scheduling, communication, model operator affinity, and other aspects of RL inference, ultimately achieving an **inference throughput of 610+ token/p/s in the DeepSeek-R1 randomly initialized weights 1K-forced-to-3K scenario**. The main journey is shown in the following figure:

![](../figures/image22.png)

All the above optimization code has been merged into the [vLLM Ascend community repository](https://github.com/vllm-project/vllm-ascend). This section will introduce, one by one, the features with significant benefit contributions, including TorchAir full-graph sinking enablement, MoE/MLA multi-stream, zero-redundancy TP-to-EP communication optimization, KV cache supporting NZ, Transpose/Cumsum elimination for large EP, and RL inference scheduling bound optimization.

### 6.1 TorchAir Full-Graph Sinking

#### 6.1.1 Phenomenon Analysis

PyTorch's Eager mode can seamlessly connect to Ascend hardware through the Ascend Extension for PyTorch (torch\_npu), but the total inference time is dragged down by the operator dispatch scheduling overhead, resulting in a so-called scheduling Bound phenomenon that greatly impacts performance. The following figure uses the MLA structure of DeepSeek-V3 as an example to provide an intuitive demonstration of PyTorch's scheduling overhead in Eager mode: the NPU remains idle most of the time, and the CPU-side operator scheduling overhead has become the main factor in inference time.

![](../figures/image23.PNG)

#### 6.1.2 Optimization Scheme

To address this problem, PyTorch 2.0 released the major feature torch.compile, which implements native CPU/GPU-based graph mode capability based on TorchDynamo, supporting the capture of operators in the network into graphs with unified scheduling, eliminating scheduling latency during Decode inference. Correspondingly, TorchAir (Torch Ascend Intermediate Representation) is middleware that adapts TorchDynamo to Ascend IR, providing the foundation for PyTorch's graph mode in NPU environments. This practice enables the TorchAir graph mode in vLLM, eliminating the operator dispatch overhead of the inference Decode part during the rollout phase.

#### 6.1.3 Optimization Effect

After enabling the TorchAir graph mode, the profiling of a single-layer model is shown in the following figure, where it can be seen that the scheduling overhead is completely eliminated:

![](../figures/image24.PNG)

In measurements, the double-layer Dense + MoE network inference time is reduced by 9.5ms in graph mode, and Decode throughput is doubled in the 671B full-parameter network.

### 6.2 MoE/MLA Multi-Stream

#### 6.2.1 Phenomenon Analysis

The TorchAir graph mode supports expressing the semantics of multi-stream operator execution in network scripts. Based on this feature, combined with the structural characteristics of the DeepSeek-V3 network, some operators can be executed concurrently across multiple streams to improve Decode performance.

##### MoE Multi-Stream - Communication-Computation Parallelism

The MoE structure of the DeepSeek-V3 network computes the activations of routed experts and shared experts. The weights of routed experts are partitioned by EP, requiring AlltoAll communication between cards before and after the GroupedMatmul computation (implemented by the MC2 operator); the weights of shared experts are partitioned by TP, requiring AllReduce communication between cards after the Matmul computation. Moreover, the computations of routed experts and shared experts have no data dependency.

In the native vLLM Ascend implementation, shared expert computation and routed expert computation execute serially, corresponding to the profiling shown in the following figure. It can be seen that there are the following two problems in its computation-communication orchestration:

![](../figures/image25.PNG)

- While the MC2 operator (which contains both communication and Vector computation) executes, the Cube compute unit is idle.

- Shared expert computation takes a short time, but the subsequent AllReduce operation runs concurrently with MC2, causing communication bandwidth contention.

##### MLA Multi-Stream - CV Parallelism

Similar to the MoE structure, the MLA structure of the DeepSeek-V3 network also contains some computation steps without data dependencies. The native vLLM Ascend implementation does not consider this characteristic and executes all operators serially, leaving further room for optimization.

![](../figures/image26.PNG)

#### 6.2.2 Optimization Scheme

For the MoE part, this practice uses TorchAir's multi-stream capability to split shared expert computation onto parallel computation streams and rearrange it. The shared expert's "gate\_up\_proj" matrix operation is parallelized with the routed expert's "dispatch" communication operator, and the shared expert's "down\_proj" matrix operation is parallelized with the routed expert's "combine" communication operator, optimizing Decode time through communication-computation parallelism.

![moe_optimization_1](../figures/image27.png)

At the same time, to avoid the AllReduce communication tail after the shared expert's "down\_proj" computation, this practice additionally eliminates the AllReduce communication brought by tensor parallelism by storing redundant shared expert weights. Since the shared expert's matrix operations are inherently small in scale, they can be basically hidden by the MC2 operator even without TP partitioning, yielding further performance gains.

Similarly, for the MLA part, through certain pipeline arrangement, the Cube and Vector computations without data dependencies can be executed in parallel across multiple streams, improving Decode performance.

![moe_optimization_2](../figures/image28.png)

#### 6.2.3 Optimization Effect

The profiling of the network segment after enabling the MoE multi-stream and MLA multi-stream optimizations is shown in the following figure:

![](../figures/image29.PNG)

![](../figures/image30.PNG)

It can be seen that the MoE layer shared expert computation and MLA layer Vector computation have been offloaded to secondary streams, and the related time is hidden by the corresponding communication operators and Cube computation operators. This corresponds to a 6ms TPOT optimization for single-step Decode and a 7% improvement in end-to-end inference throughput.

### 6.3 Zero-Redundancy TP-to-EP Communication Optimization

#### 6.3.1 Phenomenon Analysis

To break through the single-device memory limit, this practice uses TP and EP partitioning for the MLA layer and the MoE layer respectively during inference, splitting the complete DeepSeek-V3 model across multiple devices. In the native vLLM Ascend implementation, the MLA layer completes data aggregation in tensor parallel scenarios through AllReduce, and the MoE layer uses the Split operator to extract the data needed by the current card, completing the conversion from TP partitioning to EP partitioning. Here, the combination of AllReduce + Split introduces double redundancy in both communication volume and computation volume.

![](../figures/image31.png)

#### 6.3.2 Optimization Scheme

This practice adopts a zero-redundancy TP-to-EP communication scheme. During the Decode phase, the AllReduce operator in the MLA layer is replaced with a ReduceScatter operator, so that the MoE layer only receives the data necessary for the current card's computation, and the AllGather communication at the end of the MoE layer is moved to the MLA layer, ensuring the correctness of the original TP partitioning logic. Since the communication volume of the AllReduce operator is equivalent to the sum of the communication volumes of the ReduceScatter and AllGather operators, this scheme eliminates the original AllGather communication and the Split operator in the MoE part while keeping the overall communication volume of the MLA layer unchanged.

![](../figures/image32.png)

#### 6.3.3 Optimization Effect

After enabling the zero-redundancy TP-to-EP communication optimization, the StridedSlice and AllGather operators in the MoE layer are eliminated. On the other hand, the reduction of MoE layer communication further alleviates the inter-card communication imbalance problem, ultimately optimizing inference TPOT by more than 2ms and improving Decode throughput by more than 3%.

### 6.4 KV Cache Supporting NZ

#### 6.4.1 Phenomenon Analysis

NZ is a special private data format on NPUs. As shown in the following figure, it divides complete data into data blocks and adopts a memory layout that is column-major between blocks and row-major within each block.

![](../figures/image35.png)

When matrices undergo block computation, the NZ format allows compute units to access contiguous data memory, thereby maximizing the performance of the NPU Cube units. Operators generally include a step to convert the data format to NZ. By pre-storing KV cache in the NZ format, the internal format conversion stage of the FusedInferAttentionScore operator can be eliminated, optimizing the performance of the FA operator during the inference Decode phase.

#### 6.4.2 Optimization Scheme

By specifying the cache\_mode parameter of the KvRmsNormRopeCache operator as "PA\_NZ" in both the Prefill and Decode phases, KV cache can be stored in the NZ format. At the same time, the KV also needs to undergo relevant transformations to satisfy the FA operator's constraints on the shape and layout of input parameters in the NZ format scenario.

#### 6.4.3 Optimization Result

This practice added a new configuration item enable\_kv\_nz under torchair\_graph\_config in vLLM Ascend to control the KV cache NZ optimization. The benefit of this optimization grows as the sequence length increases. The specific results are shown in the following figure; when the sequence length is 3K, the Decode TPOT benefit exceeds 1ms.

![](../figures/image36.PNG)

### 6.5 Redundant Operator Elimination for Large EP

#### 6.5.1 Phenomenon Analysis

##### Redundant Transpose Operator

DeepSeek-V3 models commonly use large-EP inference deployment, which reduces the number of routed experts participating in GroupedMatmul computation in the MoE part to lower single-Decode time. However, when analyzing profiling under the EP256 configuration, we found that an extra Transpose operation appears before the GroupedMatmul operator executes, causing significant performance degradation instead.

![](../figures/image37.png)

![](../figures/image38.png)

The profiling information in the above figure shows that the total time of the Transpose operator in a single MoE layer is **as high as 175us**. After inspecting the model code and comparing the dump graphs of the EP128 scenario without Transpose, we found the problem is caused by the following reasons:

- Under the EP256 single-card-1-expert configuration, because the expert weight shape contains dimensions with a value of 1, the Transpose operator on the weights triggers TorchAir's "enable\_view\_optimize" optimization (see [PyTorch Graph Mode Usage Guide > View-type Operator Optimization Feature](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00033.html) (Chinese)), causing an extra Reshape operator to be inserted.
  ![](../figures/image39.png)

- The newly introduced Reshape operator prevents the fusion Pass "GroupedMatmulTransFusionPass" from detecting its intended target pattern, ultimately causing the Transpose operator to be retained into the graph execution phase.
  ![](../figures/image40.png)

 

##### Redundant Cumsum Operator

The multi-routed expert computation in the DeepSeek-V3 network is implemented by the GroupedMatmul operator, which is used to batch-process multiple matrix multiplication operations. By combining matrix multiplication operations with the same or similar shapes, it reduces memory access overhead and the waste of compute resources, thereby improving computational efficiency. The corresponding torch\_npu interface "npu\_grouped\_matmul" relies on the input parameter "group\_list" to obtain the actual M-axis size in each sub-matrix operation, and supports providing this parameter in either the "**cumulative sum sequence**" or "**per-dimension specification**" form. The model script of vLLM Ascend uses the "cumulative sum sequence" mode rather than the latter, thus introducing the Cumsum operator used to implement the cumulative sum in the MoE layer.

![](../figures/image41.png)

In fact, this accumulation process is redundant. By configuring the "group\_list\_type=1" parameter of the "npu\_grouped\_matmul" interface, group\_list can be provided in the "per-dimension specification" manner, thereby eliminating the overhead of the Cumsum operator.

#### 6.5.2 Optimization Scheme

The optimization item "enable\_view\_optimize" is enabled by default in TorchAir, aiming to fuse consecutive view-type operators. Since the DeepSeek-V3 network does not involve this target scenario, we choose to explicitly disable it in the "additional\_config" configuration to avoid the resulting Transpose/Gmm operator fusion failure.

![](../figures/image42.png)

At the same time, modify the network script of vLLM Ascend to adjust the "group\_list\_type" configuration of the "npu\_grouped\_matmul" interface, eliminating the original Cumsum operation.

#### 6.5.3 Optimization Effect

After adjusting TorchAir's optimization feature switch "enable\_view\_optimize" and the configuration of the torch\_npu interface "npu\_grouped\_matmul", the Transpose/Cumsum operators on the critical path of the MoE phase are eliminated. Although the GroupedMatmul operator time increases, it still achieves a Decode performance benefit of 158us per layer and 9ms for the entire network, optimizing the total rollout time by **more than 28s** at a 3K inference length.

![](../figures/image43.png)

 

### 6.6 RL Inference Scheduling Bound Optimization

#### 6.6.1 Phenomenon Analysis

After significant performance optimization on the Decode model side, the proportion of pure model time in the overall Rollout is significantly reduced, which means that if only the pure model performance is optimized, subsequent improvements in overall Rollout throughput will face major limitations. Based on this, combined with the then-current Rollout time of 373s (corresponding to a throughput of 547 token/p/s), we re-decomposed and analyzed the overall RL inference process, with the results as follows:

![](../figures/image44.png)

It can be seen in the above figure that **the pure model computation time of Prefill and Decode accounts for less than 60% of the overall RL inference time, and the remaining 40% is concentrated in scheduling operations such as the pre/post-processing of Decode, vLLM V1 Engine scheduling, and weight memory offloading before RL training-inference switching**. To further improve Rollout performance, we need to shift from focusing solely on model performance optimization to optimizing the "scheduling bound" problem in such inference scenarios. After in-depth analysis, the following results were obtained:
1.  vLLM Sche part: the time is concentrated in the EOS detection synchronization across the entire DP domain before scheduling, which uses CPU AllReduce communication to obtain the result of "has\_unfinished\_requests" and decide whether to execute dummy\_batch. This CPU communication takes an average of 10ms per call, accumulating to **as much as 30s** after 3000 inference iterations.

2. Decode prepare input part: the time is concentrated in the P/D detection synchronization across the entire DP domain, similar to the "vLLM Sche part" above, taking 10ms per call and accumulating to a total of **as much as 30s** after 3000 Decode iterations. It is worth mentioning here that the deployment scheme for RL inference is P/D co-residency, so during the Sche phase it is necessary to determine whether PD hybrid deployment exists among the current DP ranks, to decide whether to use the Prefill AlltoAll or the Decode Dispatch/Combine, preventing blocking caused by inconsistent communication operators between DP ranks. Later, if this practice scheme switches to P/D separated deployment, this communication will no longer be needed.

3. RL Sche part: after inference ends and before switching to training, the RL framework first needs to offload the inference-phase memory, then reshard the model weights for the training phase. However, testing found that calling the "torch.cuda.empty\_cache()" interface to offload memory also takes about 4s. Investigating the reason, we found it calls the underlying "aclrtFreePhysical" interface, releasing the torch memory pool to the operating system. The memory released per call is about 26G, and two memory releases occur during the inference-to-training and training-to-inference processes, for a **total of about 8s.** In principle, reducing operations that directly release memory to the OS and reusing memory in the memory management of the [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/710/index/index.html) (Chinese) will help further improve the overall efficiency of training-inference switching.

#### 6.6.2 Optimization Scheme

For the CPU AllReduce communication of EOS detection and P/D detection synchronization across all DP ranks, we plan to change it from CPU-side communication on the Gloo backend to NPU-side communication on the HCCL backend. The related schematic diagram is as follows:

![](../figures/image45.png)

Here, the actual communication volume of the AllReduce communication is very small, only at the byte level, so the full-interconnect HCCS high communication bandwidth of Atlas A3 series products is of no help to us. What we care about is the latency comparison between CPU TCP protocol communication and NPU HCCS communication:

| Comparison Item     |  NPU-HCCS (HCCL)    |  CPU-TCP (Gloo)    |
| ---- | ---- | ---- |
|  Latency    |  Extremely low (microsecond level, typically < 10μs)    |  Relatively high (tens to hundreds of microseconds, affected by OS scheduling/protocol stack overhead)    |
|  Protocol overhead    |  Dedicated hardware protocol, bypasses the OS kernel, at the data link layer (L2)    | Must go through the TCP/IP protocol stack (kernel-user mode switching), at the transport layer (L4)     |
| Large-cluster scalability     |  Nearly linear scaling    |   As the number of nodes increases, latency rises significantly (increased hop count in tree/ring topologies)   |


On the Atlas A3 128-card environment, a single CPU-TCP communication takes far more than the microsecond level, reaching the order of 10ms. Testing found that CPU communication efficiency is strongly correlated with cluster size. Gloo communication uses a tree algorithm, and as the cluster scale grows, latency increases as O(log2 N). In contrast, NPU HCCL communication uses optimizations such as the Double-Ring communication algorithm and hierarchical communication, with latency growing nearly linearly as the cluster scales. Therefore, it is inferred that switching the communication backend to HCCL can significantly reduce communication latency and obtain substantial performance benefits.

#### 6.6.3 Optimization Effect

In the Atlas A3 128-card environment, after changing the backend of the allreduce communication in the "has\_unfinished\_requests" function of vLLM's pre-scheduling processing and the "\_get\_forward\_metadata\_across\_dp" function of Decode's pre-processing from Gloo to HCCL, the single communication time is optimized from the original 14ms to within 3ms, as shown in the following figure:

![](../figures/image46.png)

The benefits achieved on the complete DSR1-671B network on Atlas A3 128 cards are as follows:

1.  Inference throughput is optimized from 519 tps to 618 tps.

2.  Inference time is optimized from 385s to 322s, a benefit of about 60s (2 communications\* 10ms optimization\* 3072 Decode inferences), an improvement ratio of nearly 20%, with significant effect.
![](../figures/image47.png)

<span id="foot1"><a href="#ref1">[1]</a> :
The features enabled here include Sections [5.1](#5.1-Reusing-MindSpeed-Training-Optimizations), [6.1](#6.1-TorchAir-Full-Graph-Sinking), [6.2](#6.2-MoE/MLA-Multi-Stream), and [6.4](#6.4-KV-Cache-Supporting-NZ) of this article. The related code can be found in [verl](https://github.com/volcengine/verl/pull/3427) and [vLLM Ascend](https://github.com/vllm-project/vllm-ascend/tree/v0.9.1rc2).

<span id="foot2"><a href="#ref2">[2]</a> :
Here, all the optimization features described in Chapters 3 to 6 of this article are enabled. Among them, the inference optimizations have been merged into the vLLM Ascend mainline; for other optimizations, please refer to the open-source RL training recipe code in the [GitCode (Chinese)](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_rl/deepseek/README.md) repository.

<!-- static-meta
{
  "sidebarTitle": "DeepSeek RL Optimization",
  "parallelism": ["data-parallel", "tensor-parallel", "pipeline-parallel", "expert-parallel"],
  "hardware": ["atlas-a3"],
  "cannFeatures": ["multi-stream"],
  "frameworks": ["megatron", "mindspeed", "verl", "vllm"],
  "cover": {"type": "auto"}
}
-->
