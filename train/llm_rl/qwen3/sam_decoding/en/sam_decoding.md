# SAM Speculative Inference: A Powerful Accelerator for Long-Sequence Reinforcement Learning Training

To address the time bottleneck of massive interactive sampling in large language model reinforcement learning training (RL-training), traditional speculative decoding techniques rely on auxiliary models and carry the risk of distribution drift caused by policy updates. This document proposes introducing a model-free (Model-Free) speculative decoding scheme based on a **suffix automaton** (SAM) into RL training. This method requires no auxiliary models and leverages the inherent structured repetition characteristics in RL data (such as mathematical reasoning and code generation) to generate candidate sequence retrieval. Combined with engineering optimizations such as adaptive batch scheduling and vectorized rejection sampling, this document completes end-to-end verification of SAM speculative decoding in RL post-training scenarios on the Qwen3 series models (32B/235B). Practice shows that SAM significantly reduces Rollout inference latency while ensuring strictly lossless accuracy, especially achieving more than 35% acceleration gains in the long-tail stage. The related code has been fully open-sourced in [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train).

## 1. Background

### 1.1 SAM Principles

SAM (suffix automaton) is a data structure that can efficiently solve many string problems. Intuitively, the SAM of a string can be understood as a compressed form of **all substrings** of the given string. SAM mainly maintains two important sets:

1. End position `endpos`: considering any non-empty substring T of string S, we denote `endpos(T)` as the set of all end positions of T in string S. For example, for the string ABCBC we have `endpos(BC)`={2,4}. In SAM, all substrings with the same `endpos` set are grouped into the same **state**, also called an **`endpos` equivalence class**. The partitioning of `endpos` equivalence classes allows SAM to store all substring information with O(|S|) space complexity.

2. Suffix link `link`: for any non-initial state u in SAM, the longest substring in u is w; then the state pointed to by `link(u)` corresponds to the longest suffix of w whose `endpos` set is different from that of w, i.e., the state where the **largest proper suffix** of all substrings represented by u resides. Suffix links constitute the core tree structure of SAM, called the **suffix link tree**, which allows SAM to be constructed in O(|S|) linear time, because each time a new character is added, only the suffix link needs to be used to quickly locate and update the relevant states.

The figure below shows the SAM constructed based on the string ABCBC:  

<div align="center">
<img src="../figures/image35.png" style="width:60%; height:auto" />
</div>

  
In the figure, the left `next` is the state transition function of SAM, indicating how different states in SAM transition in order, and is the **automaton** part of SAM; the right `link` is the suffix link described above, and is the **structure** part of SAM. SAM uses the [**incremental construction method**](https://oi-wiki.org/string/sam/) (Chinese) with linear complexity. Each time a new character is added to the current SAM, both `next` and `link` are updated, and the time complexity of a single update is O(1).

[**SAM Decoding**](https://arxiv.org/abs/2411.10666) **applies SAM to speculative decoding**: speculative decoding is transformed into suffix matching in the suffix automaton, and SAM can efficiently implement this process in O(1) constant time.

The original work uses two types of suffix automata — static SAM (S-SAM) and dynamic SAM (D-SAM).

* Static SAM builds the suffix automaton based on an existing corpus, concatenating all texts into a continuous string and adding special symbols (such as EOS) at the end of each text segment, then traversing the preprocessed corpus token by token and expanding nodes by adding next edges and link edges. For each state of the automaton, we record the position of the first occurrence of all endpos-equivalent strings corresponding to that state (min\_endpos).

* Dynamic SAM builds a dynamic suffix automaton entirely based on the currently processed prompt. The automaton dynamically expands with the generation process. Each time a new token is generated and verified by the LLM, D-SAM is immediately updated: if the current token can transition through the next edge of an existing node, the node state is directly updated; if it cannot transition, a new node is created and next and link edges are added, ensuring that D-SAM always contains the most recently generated text sequence. For D-SAM, min\_endpos can be maintained while building the automaton.

In the practice of this document, we use pure dynamic SAM to achieve zero migration cost and avoid introducing additional system maintenance complexity.

**Draft generation based on SAM-Decoding**: starting from the last token of the current sequence, traverse back through the SAM's link edges until a node that can transition to the next token through a next edge is found. According to the node's min\_endpos, locate the start position of the matching suffix in the current text sequence, and extract the subsequent consecutive tokens as the candidate sequence. The complexity of a single query is O(|S|) in the worst case and O(1) on average. The following is a specific example.

Again using the string ABCBC as an example (the figure below corresponds to the link state diagram above), for simplicity we use the longest substring in each equivalence class to represent the node. The current last token is C. We traverse the last node of the SAM back along the link edges to reach node BC.

<div align="center">
<img src="../figures/image36.png" style="width:50%; height:auto" />
</div>

  
This node can find a next edge (the blue dashed arrow in the figure below) to transition to the next token, so the traversal stops. The longest substring BC in this node is the **longest suffix** we are looking for (which appeared in the prefix substrings). Therefore, we use the subsequent sequence BC of this node as the draft token generated this time.

<div align="center">
<img src="../figures/image37.png" style="width:50%; height:auto" />
</div>

After generating the draft, we use the large model to verify it token by token. If a candidate token matches the large model's predicted distribution, it is retained; if it does not match, "rejection" is triggered — the large model regenerates starting from the rejected position. This process is **rejection sampling**. Rejection sampling ensures that the final output is exactly the same as the result generated directly by the large model, achieving "lossless acceleration".

> Suppose there is a function $\alpha\in[0,1]$ representing the acceptance probability. **Rejection sampling** samples a sample x from q\(x\), samples a random number $\epsilon$ from U\[0,1\], and accepts the sample if $\epsilon\leq\alpha(x)$; otherwise, it rejects and samples again according to this flow.

### 1.2 Comparison with Other Speculative Methods

As the demand for large language model inference acceleration grows, speculative decoding has evolved from the initial "large model + small model" paradigm into a variety of more refined and efficient technical routes. To understand the current frontier solutions in the industry more intuitively, we will focus on comparing three representative methods that include SAM speculation:

* **SAM-Decoding**: a retrieval-based non-parametric method.

* **EAGLE-3**: a lightweight auxiliary head method based on feature extrapolation.

* **MTP**: a self-speculative method based on the model itself.

These three methods represent three different evolution directions of speculative decoding: SAM attempts to mine repeated patterns in the context entirely through algorithmic data structures (suffix automaton), achieving acceleration at extremely low compute cost; EAGLE-3 further squeezes the model's internal features, performing tree-shaped speculation at the feature level through an extremely lightweight "parasitic" head, representing the SOTA level of the auxiliary model route; while MTP (as adopted by DeepSeek-V3) internalizes the speculative capability as part of the model training objective, achieving native speculation through an auxiliary head that shares the backbone parameters.

The following table summarizes the core differences of these three methods in draft generation mechanism, architecture requirements, and applicable scenarios:

<table>
    <thread>
        <tr>
            <th> Feature/Algorithm </th>
            <th style="width: 30%;"> SAM </th>
            <th style="width: 30%;"> EAGLE-3 </th>
            <th style="width: 30%;"> MTP </th>
        </tr>
    </thread>
    <tbody>
        <tr>
            <td> Draft generation method </td>
            <td> Retrieval with the help of a suffix automaton: from the generated historical text, retrieve "longest suffix + subsequent sequence" as the draft.</td>
            <td> Above the feature layers of the main model, use a lightweight EAGLE head (usually 1 layer) combined with tree attention (Tree Attention) to perform feature extrapolation and generate the draft. </td>
            <td> Add multi-token prediction heads (multi-token heads) to the model structure; these prediction heads predict multiple future tokens at once as the draft. </td>
        </tr>
        <tr>
            <td> Additional components </td>
            <td> No additional model parameters; if using cold start, an additional corpus is needed.</td>
            <td> A very small network attached to the main model needs to be trained (parameter count usually &lt;1%). Training-time Test is adopted — the "guess-verify" process at inference time is directly simulated during training to eliminate the deviation between training and inference, making the draft model output more consistent with the main model.</td>
            <td> Add multi-token prediction heads (multi-token heads) to the model structure; these prediction heads predict multiple future tokens at once as the draft.</td>
        <tr>
        <tr>
            <td> Deployment cost/difficulty </td>
            <td> Low: plug-and-play, no fine-tuning or additional training of the model is required. The main challenge lies in efficiently implementing the automaton data structure in engineering; if using cold start, a corpus with good coverage is needed.</td>
            <td> Medium: the corresponding EAGLE Head needs to be trained for a specific main model. Although the training cost is low, it increases the complexity of model management (Base Model + Adapter).</td>
            <td> Relatively high: the MTP head needs to be added in the pretraining stage; an external MTP head + fine-tuning can give an existing model a certain degree of MTP capability, but the effect is usually far inferior to the native MTP structure.</td>
        </tr>
        <tr>
            <td> Applicable scenarios </td>
            <td> Performs excellently when the prompt reuse rate is high or the generated content is highly structured. Suitable for scenarios with high overlap between input and output, or contexts with strong repeated patterns (such as math, code, and some Agent scenarios). </td>
            <td> Depends on the data used to train the EAGLE head, and performs well in scenarios with distributions similar to the training data. </td>
            <td> Since natively trained MTP heads are used, it has good generalization and can handle more diverse usage scenarios. </td>
        </tr>
    </tbody>
</table>


### 1.3 SAM Acceptance Rate

We used the Qwen3-8B model to test the acceptance rate of SAM-Decoding on the Math500 dataset, with the specific configuration as follows:

* Number of speculative tokens: 3

* Sampling parameters: temperature=0, repetition penalty=1, top-k and top-p sampling not set

* Parallel setting: tensor parallel = 4

* Inference framework: vLLM Ascend

Note: the acceptance rate here refers to the mean acceptance length, including 1 bonus token. In addition, the SAM used here (and in the following text) is the dynamic SAM, which does not involve an additional corpus.

<div align="center">
<img src="../figures/image38.png" style="width:60%; height:auto" />
</div>


It can be seen that without cold start, the acceptance rate is positively correlated with the sequence length. Therefore, using SAM-Decoding in long-sequence inference scenarios brings good performance gains. We set the default number of speculative tokens to 3, because adding the 1 bonus token that would be decoded anyway makes exactly 4 tokens. In addition, the acceptance rate improvement brought by increasing the number of speculative tokens also has boundary effects, so it is not recommended to set this value too large.

### 1.4 Significance of SAM in RL Post-Training Scenarios

In the reinforcement learning post-training scenario of LLMs, the model needs to perform a large amount of interactive sampling to collect or evaluate data. This process usually involves millions of generation iterations, becoming the main bottleneck of inference cost. Due to its model-free (Model-Free) and retrieval-based (Retrieval-Based) characteristics, SAM-Decoding provides a lightweight and easy-to-deploy practical path for accelerating this critical stage.

First, as a non-parametric speculative sampling method, SAM-Decoding does not require additional model parameters, which saves the cost of specially training a draft model or an auxiliary head, and there is no need to worry about the distribution drift problem caused by model weight updates. In addition, no additional memory management is required during training, which is especially important for the colocate framework where training and inference share cards, achieving plug-and-play acceleration for existing systems.

Second, SAM-Decoding has a natural affinity for reinforcement learning datasets. In many RL tasks, the generated Context often contains a large number of repeated or structured elements. For example, in the RLVR (Reinforcement Learning with Verifiable Rewards) scenario, the model is trained to generate detailed, structured reasoning processes, which manifest as repeated formula structures and computation steps in mathematical problem solving; as highly repeated syntax patterns and library calls in code generation; and as structured action instructions or observation reports in Agent planning tasks. The suffix automaton design of SAM-Decoding can efficiently capture these local repeated and structured elements. It can quickly retrieve the longest matching suffix from historical generation as a high-quality draft, thereby significantly improving the acceptance rate of draft tokens. This capability is crucial for RL post-training flows that need to generate a large number of structured, long-sequence reasoning trajectories, effectively reducing the high sampling cost and accelerating the iteration efficiency of the model in professional fields (such as mathematics, programming, and autonomous agents).

The [Seer](https://arxiv.org/abs/2511.14617) reinforcement learning framework recently proposed by the Moonshot AI team also uses a Model-Free speculative decoding method based on compressed suffix trees, leveraging the grouping feature of "generating multiple responses for the same prompt" in RL algorithms (such as GRPO), and mining the pattern similarity of responses within a group through online context learning to achieve efficient, high-accuracy speculative decoding. This is similar in spirit to the SAM in this document.

Next, this document will present in detail the accuracy and performance verification conclusions of SAM speculative decoding in RL post-training scenarios through the long-sequence RL training practice of the Qwen3 series models.

## 2. RL Training Accuracy Verification

Although rejection sampling theoretically guarantees the losslessness of SAM speculation, in practice accuracy is also affected by various other complex factors. To ensure the accuracy correctness of SAM in RL scenarios, we performed verification work including operator validation, downstream task evaluation, and end-to-end training.

### 2.1 Operator Accuracy

The first is operator accuracy verification. Ordinary decoding and speculative decoding use different operators. In vLLM Ascend, we use `torch_npu._npu_paged_attention` for token-by-token decoding and `torch_npu._npu_paged_attention_splitfuse` for speculative decoding to generate the logprobs of draft tokens. To verify the equivalence of these two operators, we counted the difference in the logprobs predicted by the two operators for each token given the same input. The results are as follows:

<div align="center">
<img src="../figures/image39.png" style="width:70%; height:auto" />
</div>

It can be seen that the difference between the two is strictly 0, indicating that there is no accuracy difference between the operators used by ordinary decoding and speculative decoding.

### 2.2 Downstream Evaluation

We conducted pure inference evaluation (pass@1) based on the Qwen3-32B model on the math datasets GSM8k and AIME2024, with sampling parameters temperature=0.6, repetition penalty=1, and top-k and top-p not set. The test results are as follows:

| Dataset|SAM disabled|SAM enabled|
|---|---|---|
|GSM8k|88.9|88.7|
|AIME2024|66.5|66.8|

It can be found that enabling SAM speculation does not affect the accuracy of these inference tasks.

### 2.3 End-to-End Post-Training

Finally, we directly verify the accuracy of SAM through end-to-end RL training. We use the DAPO algorithm (based on [verl](https://github.com/volcengine/verl)) to train the Qwen3-32B model on the DAPO-MATH-17k dataset. We fixed the random seed of vLLM. The average response length curve and reward curve per step with and without SAM are shown below:
<table>
    <tr>
        <td style="width: 50%; text-align: center;">
            <img src="../figures/image40.png" style="width:100%; height:auto" />
        </td>
        <td style="width: 50%; text-align: center;">
            <img src="../figures/image41.png" style="width:100%; height:auto" />
        </td>
    </tr>
</table>

It can be seen that the training curves with SAM enabled basically match the baseline without SAM in both values and trends. Combined with the previous operator accuracy and downstream evaluation verification, it can be concluded that the accuracy of SAM speculative decoding in RL post-training scenarios is lossless.

## 3. RL Training Performance Verification

### 3.1 Performance Optimization

The performance of speculative decoding depends on two factors: the first is the acceptance rate, and the second is the additional time introduced by the speculative algorithm itself (generation and verification of draft tokens, rejection sampling). To maximize the gains of SAM lossless speculation in RL training, we need to increase the acceptance rate as much as possible and reduce the additional time. To this end, we made the following two major optimizations — an adaptive switch based on batch size and rejection sampling acceleration. The related code has been fully open-sourced in [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_rl/qwen3/verl-mindspeed).

#### 3.1.1 Adaptive Switch

We adapted the lossless speculation adaptive switch based on batch size in [model_runner_v1.py](https://github.com/vllm-project/vllm-ascend/blob/v0.11.0rc0/vllm_ascend/worker/model_runner_v1.py). When the switch is enabled, speculative decoding takes effect only when the number of requests currently processed by the vLLM engine is less than or equal to the threshold. Setting the speculative adaptive switch is both to reduce the additional time and to improve the acceptance rate.

First, from the perspective of reducing additional time. The Page Attention operator we currently use to verify draft tokens degrades severely when the input shape (batch size, sequence length) increases. Under large batch size and long sequence inputs, the additional time brought by draft token verification exceeds the gain of speculation itself. Therefore, we want to enable SAM speculation only at small batch sizes to balance the acceleration gains and the compute overhead.

Second, from the perspective of improving the acceptance rate. Since we currently use the mode of dynamically building SAM, i.e., treating the current sequence as the entire corpus of the suffix automaton, in the early stage of Rollout, the suffix tree is small in size, so the acceptance rate is also low. This point is also reflected in Section 1.3 above. In the later stage of Rollout, due to the uneven Rollout generation lengths in the synchronized RL framework, the request queuing and preemption mechanisms cause scheduling delays for long-output requests, resulting in a few requests dominating the final execution stage. We treat the total number of requests processed in a single step as an indicator of the Rollout process to hint whether to enable SAM speculation. In this way, significant acceleration can be achieved on long-tail requests.

#### 3.1.2 Rejection Sampling Acceleration

Rejection sampling is an additional operation introduced by speculative inference. To maximize the gains of speculative inference, the performance of this module also needs to be optimized as much as possible. The native speculative inference implementation of vLLM Ascend has poor performance, with a single inference step bringing more than 50ms of additional time, completely masking the performance gains brought by speculative inference. We used pure PyTorch-level code optimization to reduce the rejection sampling time to about one tenth of the original.

This time we mainly optimized three functions in [rejection_sampler.py](https://github.com/vllm-project/vllm-ascend/blob/v0.11.0rc0/vllm_ascend/sample/rejection_sampler.py):

* `expand_batch_to_tokens`: expands the sampling parameters (temperature, top-k, top-p) from a batch vector with "one value per request" into a token vector with "one value per draft token", so that the subsequent rejection sampling can correctly apply these parameters token by token.

* `sample_recovered_tokens_pytorch`: in rejection sampling, generates a recovered token in real time for each rejected draft position, ensuring that the sequence can continue to advance.

* `rejection_random_sample_pytorch`: compares the draft with the target probability one by one according to the accept-reject rule, decides whether to use the draft token, the recovered token, or terminate, and appends a bonus token when all are accepted, finally assembling the output sequence in the random sampling mode.

Taking `sample_recovered_tokens_pytorch` as an example, the [native implementation](https://github.com/vllm-project/vllm-ascend/blob/v0.11.0rc0/vllm_ascend/sample/rejection_sampler.py#L461) contains nested for-loop operations. This element-by-element operation method cannot leverage the parallel compute capability of the NPU, and repeatedly creates new tensors inside the loop (such as `.clone()`, `torch.full()`), resulting in extremely low efficiency.

The core idea of our optimization is **vectorization (Vectorization)**: remove all Python loops and use PyTorch tensor operations to process all data at once. We used tensor operations to eliminate the repeated tensor creation inside the loop, and implemented the computation of key variables through efficient indexing and broadcasting. For the specific implementation, refer to our [open-source code](https://gitcode.com/cann/cann-recipes-train/blob/master/llm_rl/qwen3/verl-mindspeed/patches/vllm_ascend). Later, Triton can be used to further optimize this module.

### 3.2 Measured Gains in RL Training

We tested the acceleration gains of SAM speculative decoding in the DAPO long-sequence RL scenario. As for model selection, we chose two representative models: the Qwen3-32B dense model and the Qwen3-235B-A22B MoE model. The specific configurations and performance data are introduced below.

#### 3.2.1 Qwen3-32B

Relevant configuration:

* Dataset: DAPO-MATH-17k

* Maximum response length (`max_response_length`): 34816

* Training batch size (`train_batch_size`): 32

* Generation batch size (`gen_batch_size`): 96

* Rollout maximum number of requests (`max_num_seqs`): 128

* Rollout model tensor parallelism (`tensor_model_parallel_size`): 8

* SAM-related configuration: the adaptive batch size switch threshold is 8, and the number of speculative tokens is 3.

* Number of 910C cards: 2 machines, 16 cards, 32 dies

* Runtime mode: eager

The performance data is shown in the table below, where "average single-round inference time" refers to the average inference time of a single round (the DAPO algorithm may have multiple rounds of generation due to its dynamic filtering mechanism) in each DP (Data Parallel) domain, and "single-step total inference time" refers to the total Rollout time of a single step:

| |Average single-round inference time / s|Single-step total inference time / s|Single-step total time / s |
|---|---|---|---|
|SAM disabled|3904.22|4159.06|4730.86|
|SAM enabled|3548.62|3793.29|4374.99|
|Gain|10.02%|9.64%|8.13%|

#### 3.2.2 Qwen3-235B-A22B

Relevant configuration:

* Dataset: DAPO-MATH-17k

* Maximum response length (`max_response_length`): 34816

* Training batch size (`train_batch_size`): 128

* Generation batch size (`gen_batch_size`): 128

* Rollout maximum number of requests (`max_num_seqs`): 64

* Rollout model tensor parallelism (`tensor_model_parallel_size`): 4

* SAM-related configuration: the adaptive batch size switch threshold is 8, and the number of speculative tokens is 3.

* Number of 910C cards: 8 machines, 64 cards, 128 dies

* Runtime mode: eager

Performance data:

| |Average single-round inference time / s|Single-step total inference time / s|Single-step total time / s|
|---|---|---|---|
|SAM disabled|7102.92|14287.54|15441.45|
|SAM enabled|6467.52|12811.98|13960.98|
|Gain|9.82%|11.52%|10.60%|

#### 3.2.3 Customer Scenario Validation

We deployed SAM speculative decoding in a customer's training scenario, helping achieve the goal of reinforcement learning performance improvement for Qwen3-32B+verl+DAPO on the 128-die A3 cluster.

Relevant configuration:


* Maximum response length (`max_response_length`): 34816

* Training batch size (`train_batch_size`): 512

* Generation batch size (`gen_batch_size`): 1536

* Rollout maximum number of requests (`max_num_seqs`): 256

* Rollout model tensor parallelism (`tensor_model_parallel_size`): 8

* SAM-related configuration: the adaptive batch size switch threshold is 4, and the number of speculative tokens is 4.

* Number of 910C cards: 8 machines, 64 cards, 128 dies

* Runtime mode: eager

The specific performance data is as follows:

| |Single-step total inference time / s|Single-step total time / s |
|---|---|---|
|SAM disabled|9094|11870|
|SAM enabled|8223|11000|
|Gain|10.59%|7.91%|

#### 3.2.4 Rollout Long-Tail Analysis

In addition, we specifically counted the performance of SAM speculation in the Rollout long-tail stage (when the number of currently processed requests is below the speculation threshold, i.e., the purple part in the figure below), reaching **1.35X** of non-speculative performance. If the operators are optimized later to support a larger speculation threshold, this gain will be even more obvious.

<div align="center">
<img src="../figures/image42.png" width = "50%" />
</div>

## 4. Summary and Outlook

The successful deployment of SAM-Decoding proves that in the field of LLM training/inference acceleration, innovations at the traditional algorithm and data structure level (such as the suffix automaton) can serve as a powerful complement to model parameter methods. It provides a plug-and-play and easy-to-integrate acceleration solution for LLMs in complex RL scenarios with long sequences, high concurrency, and coexisting training and inference. This has significant practical value for RL post-training flows that urgently need to reduce the cost of massive interactive sampling. In the future, we will explore more efficient speculative strategies (to improve the acceptance rate) and try combining them with graph mode (to improve operator execution efficiency). Through continuous technical deepening and engineering optimization, we believe that Model-Free speculative technologies represented by SAM speculative decoding will play an even greater role in LLM reinforcement learning training.

<!-- static-meta
{
  "sidebarTitle": "Speculative Decoding",
  "parallelism": ["tensor-parallel"],
  "hardware": ["atlas-a3"],
  "frameworks": ["verl", "vllm"],
  "cover": {"type": "auto"}
}
-->
