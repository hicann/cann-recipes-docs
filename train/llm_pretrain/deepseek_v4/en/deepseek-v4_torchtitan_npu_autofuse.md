# DeepSeek-V4 Ascend Training Support: Minimalist Training Optimization Practice of TorchTitan-NPU + AutoFuse on the CANN Platform

**Abstract**: This article introduces the training optimization practice of the DeepSeek\-V4\-Flash model based on the CANN platform. Built on the TorchTitan\-NPU framework, it adopts the pure FSDP + large EP minimalist parallel strategy to achieve optimal memory usage; innovatively enables the training-graph technology and, leveraging the Ascend C AutoFuse capability, obtains an end-to-end 32% compilation benefit; for the sparse attention structure, it customizes high-efficiency fused operators to fully unleash chip compute power. The DeepSeek\-V4\-Flash model achieves a training throughput of 1100 tokens/p/s on a 64-card A3 cluster with 4K-sequence BF16 training. This rapid support capability validates the feasibility and efficiency of the minimalist out-of-the-box solution in NPU large-model training, aiming to help existing Ascend A3 cluster users quickly carry out continued training/SFT and self-developed algorithm validation based on the new DeepSeek model architecture.

All code involved in this article is open source: [torchtitan-npu repository](https://gitcode.com/cann/torchtitan-npu). For a quick hands-on experience, see [DeepSeek-V4-Flash Training Deployment Guide Based on TorchTitan-NPU](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_pretrain/deepseekv4/README.md) (Chinese).
***

## Highlights

* The training framework adopts the TorchTitan + TorchTitan\-NPU plugin-based solution, using a streamlined parallel partitioning strategy of **super-node-affinity large EP + pure FSDP** to achieve **optimal memory usage** with extremely low adaptation cost and communication overhead, striking a good balance between usability and performance

* TorchTitan\-NPU deeply adapts to the torch.compile mechanism and enables the training-graph technology. Relying on **Inductor + AutoFuse** (an Ascend C-based Codegen backend), it implements end-to-end **automatic fusion of Vector operators**, delivering up to **31.8%** out-of-the-box performance gains for the entire network

* For complex structures such as sparse attention, we developed multiple high-efficiency NPU fused operators including **SparseAttnSharedkv** and **LightningIndexer**, collaboratively optimizing from dimensions such as load-balanced split-core computation and memory-computation balance, fully unleashing the chip's sparse compute power

* Based on the above optimizations, CANN already supports DeepSeek\-V4\-Flash model training based on TorchTitan, achieving a training **throughput of 1100 tokens/p/s** on an A3 cluster with BF16 precision, 64 cards, 4K sequence + MTP1

## Introduction

TorchTitan brings the industry a brand-new option for PyTorch Native large-model training solutions. Its core design philosophy is to naturally decouple model algorithms from distributed parallelism and operator optimization. By supporting torch.compile training-graph optimization, it eliminates hand-written fused kernels and greatly improves the usability of large-scale distributed training. In terms of parallel strategies, TorchTitan deeply integrates FSDP2, achieving cross-device sharding of parameters, gradients, and optimizer states, significantly reducing per-card memory usage.

The overall software architecture of this practice, from top to bottom, consists of TorchTitan, the TorchTitan\-NPU plugin, TorchInductor, and the AutoFuse component at the CANN layer. Each layer has clear responsibilities and complementary strengths:

* **TorchTitan**: As a PyTorch-native distributed training framework, it avoids dependence on heavyweight frameworks such as Megatron, providing out-of-the-box support for parallel capabilities like FSDP with a streamlined codebase.

* **TorchTitan\-NPU**: Provided as a pluggable, installable plugin, it implements hardware-accelerated adaptation of communication and operators for Ascend clusters while maintaining lightweight integration with upstream TorchTitan.

* **TorchInductor**: It directly inherits the graph capture and fusion optimization capabilities of the community torch.compile, reusing the relevant infrastructure of the PyTorch compilation ecosystem without reinventing the wheel.

* **AutoFuse**: Targeted at Ascend NPUs, it is positioned as an Ascend C-based Codegen & Schedule backend. On top of Inductor, it injects architecture-aware fusion, optimization, and expansion strategies to transform computation graph optimization into efficient NPU instruction sequences.

![Architecture](../figures/Architecture.png)

Based on the above layered architecture, we completed the adaptation and tuning of the DeepSeek\-V4\-Flash model continued-training workflow on top of the **TorchTitan\-NPU acceleration plugin**. The entire solution revolves around three main threads: "minimalist parallelism, automatic fusion, and custom operators". The following sections elaborate on the minimalist distributed parallelism based on FSDP+EP, the collaboration between torch.compile and AutoFuse automatic fusion, and high-performance fused operator optimization and acceleration.

## Minimalist Distributed Parallelism Optimization

### Advantages of the Minimalist FSDP+EP Partitioning

In MoE large-model training, EP partitioning can effectively share the storage pressure of routed expert weights, but non-expert parameters such as the attention module, shared experts, word embeddings, and vocabulary projection layers still require additional partitioning mechanisms. On the other hand, as model architectures enter another phase of rapid development, traditional TP/PP schemes face numerous challenges in both performance overhead and usability, while FSDP demonstrates relatively significant advantages in the above scenarios. Therefore, based on the A3 super-node 64-card environment, this practice innovatively chooses the pure FSDP + EP minimalist partitioning scheme in DeepSeek\-V4\-Flash model training, with the specific configuration as follows:

|**TP**|**PP**|**VPP**|**EP**|**FSDP**|**MBS**|**GBS**|**Cluster**|
|---|---|---|---|---|---|---|---|
|1|1|1|128|128|1|1024|64 cards|

#### Introduction to FSDP

FSDP (Fully Sharded Data Parallel) is a data parallelism technique provided by PyTorch. It significantly reduces per-card memory usage by sharding model parameters, gradients, and optimizer states across devices. Its principle is to decompose DDP's AllReduce into a forward AllGather and a backward ReduceScatter, collecting parameters on demand during computation and releasing them immediately after use to minimize resident memory. In addition, FSDP's communication operations can overlap with computation: AllGather prefetches the next layer's parameters in advance, while ReduceScatter transfers synchronously during backward computation, effectively hiding communication latency within computation time.

![FSDP-Illustration](../figures/FSDP-Illustration.png)

#### Memory Benefits

Although routed expert weights in MoE models absolutely dominate the global parameter count, after EP-dimensional partitioning, the proportion of non-routed expert weights in per-card memory rises significantly and becomes a new memory bottleneck. The following table uses the DeepSeek\-V4\-Flash model as an example to show the bottleneck shift before and after EP partitioning, where **master weights** and **model gradients** are both calculated based on the common 4-byte size in mixed-precision training scenarios.

<table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; text-align: center; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th><strong>FSDP</strong></th>
      <th><strong>DP</strong></th>
      <th><strong>EP</strong></th>
      <th><strong>Total Parameters</strong></th>
      <th><strong>Non-expert Parameters</strong></th>
      <th><strong>Non-expert Weights + Gradients</strong></th>
      <th><strong>Expert Parameters</strong></th>
      <th><strong>Expert Weights + Gradients</strong></th>
      <th><strong>Non-expert Memory Ratio</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>128</td>
      <td>1</td>
      <td>285B</td>
      <td>8B</td>
      <td style="background-color: #ffcccc;">59.5GB</td>
      <td>277B</td>
      <td>2060GB</td>
      <td>2.8%</td>
    </tr>
    <tr>
      <td>1</td>
      <td>128</td>
      <td>128</td>
      <td>285B</td>
      <td>8B</td>
      <td style="background-color: #ffcccc;">59.5GB</td>
      <td>~2B</td>
      <td>16GB</td>
      <td>80%</td>
    </tr>
    <tr>
      <td>128</td>
      <td>1</td>
      <td>128</td>
      <td>285B</td>
      <td>0.06B</td>
      <td style="background-color: #ccffcc;">0.465GB</td>
      <td>~2B</td>
      <td>16GB</td>
      <td>2.8%</td>
    </tr>
  </tbody>
</table>

Given the 64GB memory size of a single A3 card, it is unavoidable to introduce partitioning of non-expert parameters during model training. Compared with TP/PP, FSDP's communication can be completely hidden by computation (see the performance section for details), so a relatively aggressive memory partitioning configuration can be used without being constrained by performance considerations. In measurements, after enabling FSDP128, non-expert weights are evenly distributed across all 128 DIE, **almost completely eliminating this portion of memory usage** — after TorchTitan completes model initialization, the per-card weight memory is only about 8~9GB, which can be considered to consist solely of routed expert weights after EP partitioning.

![FSDP-vs-TP-PP](../figures/FSDP-vs-TP-PP.png)

#### Performance Overhead

FSDP communication data does not depend on the intermediate computation results of the current layer, so the framework can dispatch collective communication operations to an independent communication stream to execute asynchronously and concurrently with the model's forward/backward computation. **Thanks to the high communication bandwidth of the A3 super-node, measured FSDP communication overhead can be completely hidden by the computation process.** This property breaks the trade-off between performance and memory — when FSDP128 expands the communication domain to 8 machines, training throughput is still not degraded by exposed communication, providing feasibility guarantees for extreme compression on the memory side.

![FSDP-overlapped](../figures/FSDP-overlapped.png)

In contrast, communication in TP/PP is tightly coupled with the critical path of computation and cannot be hidden. Even with the SIO high-bandwidth communication of the A3 chip and extremely small communication domain configurations such as TP2, tensor parallelism still introduces considerable communication overhead. On the other hand, the pipeline bubble rate of PP partitioning increases synchronously with the expansion of the communication domain scale, and the resulting device idle time is also non-negligible. These practical constraints in turn restrict the conversion of data parallelism into TP/PP, and in actual deployments data parallelism often cannot be fully partitioned into TP/PP, leaving a certain degree of memory redundancy.

#### Usability Advantages

In addition to memory and performance advantages, FSDP is also significantly better than TP/PP in engineering complexity:

* **For TP partitioning**, the introduction of MLA/DSA/C4A/C128A structures in the DeepSeek series models makes TP partitioning of the attention mechanism increasingly complex. The attention mechanism of DeepSeek\-V4 inherits, in its overall framework, the DSA structure introduced by DeepSeek\-V3.2. Its Indexer module involves accumulation operations along the model head dimension when computing both forward results and Loss values, as shown in the figure below. The semantics of this operation determine that TP partitioning cannot be performed directly along the head dimension — forcing such partitioning would require simultaneously introducing new AllGather aggregation communication to ensure the correctness of ReduceSum.

![Indexer-TP](../figures/Indexer-TP.png)

* **For PP partitioning**, non-uniform modules in the model (Embedding, LmHead, MTP, etc.) can easily lead to uneven time consumption across PP stages, and the load imbalance inherent to data distribution in MoE models further aggravates the difficulty of PP performance tuning.

In contrast, FSDP's approach of indiscriminately sharding parameters with asynchronous AllGather achieves weight partitioning that is agnostic to model algorithms and structures — during forward/backward computation, the complete single-layer model weights can always be obtained, and the computation executed across DP ranks has no differences. Moreover, adaptation requires only a single-line `fully_shard` function call, which effectively overcomes the shortcomings of TP/PP in the aforementioned scenarios.

### Dual Communication Domain Overlap Optimization

Based on the above advantages of FSDP over TP/PP, we selected FSDP128 + EP128 as the model partitioning strategy, and during actual testing optimized a communication domain initialization problem in the upstream TorchTitan implementation: in the aforementioned partitioning strategy, both the FSDP domain and the EP domain are identical to the global default communication domain of the distributed parallel group, causing PyTorch to **assign the same ProcessGroup to FSDP and EP**, which forces their communication operations to execute serially and prevents concurrency.

![FSDP-EP-bug](../figures/FSDP-EP-bug.png)

Specifically, as shown in the figure above, EP communication lies on the critical path of MoE model execution, and any blocking directly delays the overall computation of the model. When FSDP and EP share a ProcessGroup, FSDP's AllGather/ReduceScatter forms a serial waiting relationship with EP's AllToAll: FSDP's communication operations block the execution of EP communication, and ultimately further block subsequent computation operators. This means that the FSDP communication originally designed to be hidden by computation is exposed to the critical path, causing overall throughput degradation.

![FSDP-EP-bugfix](../figures/FSDP-EP-bugfix.png)

By independently marking the EP communication domain during the initialization phase, FSDP and EP are able to hold their own ProcessGroups. Their communication operations can therefore execute concurrently on different HCCL streams, restoring the design intent that FSDP communication be hidden by the model's main computation flow. After the fix, the critical-path communication of EP is no longer blocked by FSDP. Measured results show that the fix reduces the communication wait time of the model at the MoE layer from a maximum of 10ms to ~3ms, bringing a 1.1% improvement in model training performance. We will later feed this issue back to the PyTorch/TorchTitan upstream community, hoping to facilitate optimization of the framework in subsequent versions.

### Memory Offload Optimization

In the native FSDP2 solution, although parameters, gradients, and optimizer states are all partitioned, AdamW's two FP32 momenta still reside in device memory during forward/backward computation, resulting in idle waste. **As a memory optimization supplement to FSDP**, [SwapOptimizer (Chinese)](https://gitcode.com/Ascend/MindSpeed/blob/master/docs/zh/features/swap-optimizer.md) further offloads the optimizer states that FSDP has partitioned to a single card to host memory, swapping them in on demand only during the weight update phase. We implemented this feature in TorchTitan by referencing the MindSpeed approach, adapting it to the DTensor weight scenario, and designing a slice pipeline at parameter granularity to overlap the serial "load-update-offload" process in the manner shown in the figure below, reducing the memory peak during the optimizer update phase.

![Swap-optimizer](../figures/Swap-optimizer.png)

This mechanism further reduces the resident memory in the AdamW optimizer scenario from FSDP's 16 bytes/parameter (weight 4 + gradient 4 + optimizer 8) to 8 bytes/parameter (weight + gradient only), nearly halving memory usage.

## torch.compile + AutoFuse

With the proliferation of MoE and multimodal architectures, dynamic, fine-grained compositions of small operators have become the mainstream design pattern. Taking the mHC proposed by DeepSeek as an example, it decomposes the HC process into three parts — HcRes, HcPre, and HcPost — which guarantee mathematical properties through doubly stochastic matrices and Sigmoid gating coefficients respectively, but the implementation relies on a large number of fragmented PyTorch operations. Such flexible structures are increasingly common in model design. If the traditional approach of hand-writing fused operators is continued, development efficiency can no longer keep pace with model iteration speed, making automatic operator fusion capability a rigid requirement.

![mHC](../figures/mHC.png)

torch.compile provides the foundational capability for this: by capturing the FX computation graph, it converts model logic into an operator graph representation, which is handed to the Inductor backend for operator fusion and code generation. Through vertical and horizontal fusion strategies, Inductor merges multiple fragmented operations into a single efficient kernel, achieving cross-operator global optimization. **TorchTitan provides native deep support for torch.compile**, fully reusing PyTorch's existing compilation ecosystem infrastructure and optimization results.

However, NPUs and GPUs differ fundamentally in memory architecture and computational paradigm, and directly applying GPU optimization strategies cannot fully unleash NPU potential. To this end, **the AutoFuse component of the CANN framework, based on inheriting the overall Inductor flow, implements a Schedule & Codegen backend targeting Ascend C.** Through architecture-aware fusion strategies and instruction scheduling, it achieves automatic mapping from computation graphs to efficient NPU operator code, significantly improving the execution efficiency of fused operators on Ascend platforms.

![AutoFuse](../figures/AutoFuse.png)

In the training scenario of this practice, since TP partitioning is not enabled, the computation volume of each operator is inherently large, and after operator fusion the Kernel Launch overhead accounts for an extremely low proportion, leaving no room for static-graph benefits. Therefore, we adopt the dynamic-graph Eager mode to execute the compiled operator graph. Moreover, the Eager mode does not need to handle dynamic Shape constraints, making it simpler and more flexible while still capturing the main performance benefits of fusion.

**AutoFuse Highlights**

* Easy to use: enabling it requires only one extra line of code in the PyTorch frontend

* Performance improvement: relying on NPU-affine operator generation technology, this model achieved a fusion benefit of approximately 31.8%

    * NPU-affine, template-reduction Schedule technology

    * Hardware-modeling-based, dynamically solved operator Tiling technology

    * Ascend LoopIR-expression-based Ascend C operator kernel code generation technology

* Generalization and completeness: currently 46 of 152 LoopIR expressions are supported, and the rest will be gradually completed in the future to achieve full parity with LoopIR

### AutoFuse Benefit Analysis

After enabling AutoFuse, the end-to-end throughput gain of the DeepSeek\-V4\-Flash model is 31.8%. The specific performance improvement breakdown is shown in the table below. The gains mainly come from operator fusion on AIV and the reduction of Free Time brought by Host Bound mitigation:

![AutoFuse-gain](../figures/AutoFuse-gain.png)

1. **AIV's gains mainly come from various operator fusions:**

|**Fused operator name**|**Count**|**Total time after fusion (s)**|**Total time before fusion (s)**|**Net benefit time (s)**|**Optimization core**|
|---|---|---|---|---|---|
|autofused\_mul\_sum series|10,232|1.6|7.1|5.5|Bandwidth optimization: eliminates the operation of writing Mul results back to memory and reading them back, completing the reduction directly on-chip.|
|Basic element-wise fusion (Mul/Add/Div)|~200k|7.8|12.8|5|Memory access convergence: merges a large number of independent multiplication, addition, and division operations, significantly reducing the total number of memory accesses.|
|autofused\_add\_div\_expand\_mul\_pow|1,584|1|2.6|1.6|Instruction acceleration: converts the extremely high-latency independent Pow operator into instructions, hiding memory access latency through computation.|
|BroadcastTo implicit (aggregation terms)|~33k|0.3|2.5|2.2|Implicit broadcast: implements broadcast through Stride logic transformation, completely eliminating the physical memory copy time of more than 2s.|
|autofused\_npu\_dtype\_cast series|14,832|1.8|2.5|0.7|Scheduling simplification: absorbs tens of thousands of type conversion operations, greatly reducing the frequency of CPU dispatching tasks to the NPU.|

**The following uses one fragment as an example to illustrate:**

![AutoFuse-example](../figures/AutoFuse-example.png)

Three core benefit sources:

* Eliminating the "memory wall" (Memory Bound)

    * Reconstructs the "read-compute-write" loops of 12 independent operators into 2 highly integrated fused kernels

    * Intermediate tensors reside in on-chip cache (SRAM/L1), avoiding the huge bandwidth overhead of frequent reads/writes to HBM (global memory)

* Hiding high-latency instructions (Operator Hiding)

    * Performs instruction-level reordering of high-cycle operators such as Pow/RealDiv through the compiler

    * Computation pressure is hidden within the vector computation pipeline, achieving "near-zero-cost" merging of nonlinear mathematical formula chains

* Reducing kernel launch overhead (Launch Overhead)

    * Greatly reduces the number of Host-to-Device handshakes.

    * Cleans up the "computation debris" between heavyweight operators (such as MatMul), ensuring NPU cores remain in a sustained high-load productive state

2. **Significant reduction of Host-Bound (scheduling bottleneck)**

Core optimization path: converts small operators originally executed at discrete intervals into continuous execution of fused operators, reducing hardware bubbles, ensuring the continuity of the NPU computation pipeline, and greatly improving hardware utilization

* Scheduling optimization: static analysis + Codegen Wrapper \-> precompiled execution flow (replacing the dynamic Dispatcher), reducing Host latency

* Operator-level optimization: operator fusion \-> fewer Launch invocations \-> relieving the Host-side instruction dispatch pressure

![AutoFuse-hostbound](../figures/AutoFuse-hostbound.png)

## High-Performance Fused Operator Optimization

For the Attention core structure of the newly released model, we designed and implemented 4 high-performance Ascend C operators used in training scenarios, improving the performance of core computation modules and optimizing memory.

### LightningIndexer

The LightningIndexer (LI) operator obtains the Top-k positions corresponding to each token through a series of operations, and outputs the indices of the Top-k positions for SparseAttnSharedkv to use as input to complete the computation. The specific computation formula is as follows:

$$
\operatorname{Top-}k \Bigl\{ [1]_{1 \times g} @ \bigl[ (W @ [1]_{1 \times S_k}) \odot \operatorname{ReLU}( Q_{\mathrm{index}} @ K_{\mathrm{index}}^\top ) \bigr] \Bigr\}
$$

The computation flow of LightningIndexer can be divided into 3 stages:

1. C0: Cube operation, i.e., the matrix multiplication of Q and K as well as the ReLU operation

2. V1: Vector operation, the multiple vector computations after ReLU (excluding Topk).

3. V2: Vector operation, after the preceding complete score computation finishes, the Topk computation for each token is completed through secondary split-core execution.

For pipeline scheduling, this operator designs a Preload C process to improve the pipeline parallelism between C and V computations. An example of the pipeline scheduling diagram is as follows:

![LI-pipeline](../figures/LI-pipeline.png)

### SparseAttnSharedkv

The SparseAttnSharedkv (SAS) operator is designed to complete the Attention computation described by the following formula. Based on different input cmp\_ratio values, it supports 3 types of Attention computation: Sliding Window Attention (SWA), Compressed Attention (CFA), and Sparse Compressed Attention (SCFA). The Attention part is Multi-Query Attention:

$$
S = Q @ \tilde{K}^\top
$$

$$
m = \max\bigl( \mathrm{sinks},\; \max(S) \bigr)
$$

$$
\mathrm{Attention} = \frac{ e^{S - m} @ \tilde{V} }{ \sum e^{S - m} + e^{\mathrm{sinks} - m} }
$$

Here, $\tilde{K}$ and $\tilde{V}$ are the K and V actually involved in computation, controlled by input parameters such as ori\_kv (the original KV), cmp\_kv (the compressed KV), and cmp\_ratio (the compression ratio). The implementation of the SAS operator is mainly divided into 5 stages: V0/C1/V1/C2/V2, where V denotes vector computation and C denotes Cube computation. The corresponding computation formulas are as follows:

$$
V_0 : \operatorname{Gather}\bigl( \mathrm{cmpkv},\; \mathrm{topkIndices}[i] \bigr), \quad 0 \le i < \mathrm{selectBlockCount}
$$

$$
C_1 : \mathrm{qk} = Q @ K^\top
$$

$$
V_1 : P = \operatorname{onlineSoftmax}( \mathrm{qk},\; \mathrm{sinks} )
$$

$$
C_2 : O' = P @ V
$$

$$
V_2 : O = \operatorname{rescale}( O' )
$$

During implementation, the pipeline scheduling preloads one round of V0+C1 to stagger the dependencies between different stages, achieving CV pipeline parallelism except for the head and tail.

![SAS-pipeline](../figures/SAS-pipeline.png)

### SparseAttnSharedkvGrad

SparseAttnSharedkvGrad (SASG) is the backward operator of SAS. The computation flow of the operator is as follows:

![SASG-graph](../figures/SASG-graph.png)

The main flow of the SASG operator can be divided into 5 stages: Gather, Cube12, Process, Cube345, and Scatter, as shown in the figure. The operator pipeline scheduling is implemented as follows, achieving CV pipeline parallelism by preloading one Gather+Cube12.

![SASG-pipeline](../figures/SASG-pipeline.png)

### SparseLightningIndexerGradKLLoss

Because the LI module incurs huge memory overhead during Loss computation (the memory overhead reaches the square level of the sequence length, as it requires computing the Main Attention score), the SparseLightningIndexerGradKLLoss operator fuses the Main Attention score computation, the backward pass of LI, and the Loss computation process, reducing intermediate memory usage and optimizing memory and performance. The operator's computation formulas are as follows:

The computation formula for the values selected by Top-k in LI can be expressed as:

$$
I_{t,:} = W_{t,:} @ \operatorname{ReLU}\bigl( q_{t,:} @ (K_{:t,:})^\top \bigr)
$$

When LI is trained independently, the corresponding loss function is:

$$
\mathcal{L}(I) = \sum_t D_{\mathrm{KL}} \bigl( p_{t,:} \;||\;\operatorname{Softmax}(I_{t,:}) \bigr)
$$

Here, p is the target distribution, obtained by summing the Main Attention score over all head dimensions and then applying L1 regularization to the summation result along the context direction. Here, $D_{\mathrm{KL}}$ is the KL divergence, whose expression is:

$$
D_{\mathrm{KL}}(a \| b) = \sum_i a_i \log \frac{a_i}{b_i}
$$

By differentiation, the gradient expression of the Loss can be obtained:

$$
\mathrm{d}I_{t,:} = \operatorname{Softmax}(I_{t,:}) - p_{t,:}
$$

Using the chain rule, the gradients of the weight, query, and key matrices can be further computed:

$$
\mathrm{d}W_{t,:} = \mathrm{d}I_{t,:} @ \bigl( \operatorname{ReLU}( S_{t,:} ) \bigr)^\top
$$
$$
\mathrm{d}q_{t,:} = \mathrm{d}S_{t,:} @ K_{:t,:}
$$
$$
\mathrm{d}K_{:t,:} = ( \mathrm{d}S_{t,:} )^\top @ q_{:t,:}
$$

Here, S is the softmax result of the QK matrix. The computation process can be split into 5 stages:

1. V0: extracts valid data from the K of Main Attention and the K of LI based on the Top-k indices

2. C1: completes the matrix operations of the original Q and K of Main Attention as well as the Q and K of LI

3. V1: completes the KLLoss and dW computations

4. C2: completes the dQ computation of the LI backward pass

5. V2: completes the dK computation of the LI backward pass through ScatterAdd

During implementation, V0 is preloaded twice, and the C1 and C2 computations are hidden through PingPong, improving the pipeline parallelism.

![SLIG-pipeline](../figures/SLIG-pipeline.png)

## Performance Results and Future Outlook

### Performance Results Based on A3 SuperPods with 64 Cards

Relying on the CANN platform and the TorchTitan\-NPU plugin, we quickly completed the basic training performance tuning of the DeepSeek\-V4\-Flash model on an A3 64-card cluster. The solution adopts the minimalist parallel partitioning strategy of large EP + pure FSDP, integrates the fused operators developed for sparse attention modules, and combines the Ascend C AutoFuse automatic fusion mechanism, achieving a significant increase in model throughput from the initial 397 tokens/p/s to 1100 tokens/p/s. Among these, custom fused operators and AutoFuse contribute 90% and 30% of the throughput improvement respectively.

![Total-gain](../figures/Total-gain.png)

<table border="1" cellspacing="0" cellpadding="8" style="border-collapse: collapse; text-align: center; font-family: Arial, sans-serif;">
  <thead>
    <tr style="background-color: #f2f2f2;">
      <th>CUBE</th>
      <th>FA</th>
      <th>VEC</th>
      <th>EP</th>
      <th>FSDP</th>
      <th>FREE</th>
      <th>MFU</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>16.64s<br><strong>(27.89%)</strong></td>
      <td>11.8s<br><strong>(19.77%)</strong></td>
      <td>22.08s<br><strong>(37.01%)</strong></td>
      <td>5.70s<br><strong>(9.56%)</strong></td>
      <td>1.78s<br><strong>(2.98%)</strong></td>
      <td rowspan="2">1.67s<br><strong>(2.79%)</strong></td>
      <td rowspan="2"><strong>28.78%</strong></td>
    </tr>
    <tr>
      <td colspan="3">50.52s<br><strong>(84.68%)</strong></td>
      <td colspan="2">7.48s<br><strong>(12.53%)</strong></td>
    </tr>
  </tbody>
</table>

### Future Outlook

Judging from the current Profiling data of the A3 cluster, computation time is absolutely dominant, reaching 84.68% of the total time, of which Vector-type operators account for nearly 40% of the entire network time. Therefore, further performance optimization can focus on the following directions:

* Adopt a finer-grained on-demand recomputation strategy to avoid redundant recomputation of Vector operators whose activation values do not need to be retained

* Further expand the coverage of AutoFuse automatic fusion to compress computation latency

* The current TorchTitan\-NPU version has not yet integrated MC2 or the EP-domain communication-computation parallelism mechanism based on operator pipeline orchestration; this capability will be supplemented in subsequent phases

In terms of feature expansion, we plan to follow up on the evolution directions in the DeepSeek\-V4 technical report:

* For the next-generation A5 platform, support FP8 and A8W4 low-precision quantization training features, leveraging the **MxFP8/MxFP4 low-precision** computation and communication capabilities of the A5 generation

* Integrate **Muon optimizer** support in TorchTitan\-NPU to pursue faster model convergence, along with AutoFuse and fused operator acceleration capabilities

<!-- static-meta
{
  "sidebarTitle": "Training Optimization",
  "frameworks": ["torchtitan"],
  "cover": {"type": "auto"}
}
-->
