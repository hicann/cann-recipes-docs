# HiF8 Precision and Performance Dual Breakthrough — Low-Precision Optimization Practice for Large-Model Training on the Ascend 950DT
## Chapter 1 Introduction to the HiF8 Data Format

This chapter first introduces HiF8 (HiFloat8), a new 8-bit floating-point data format independently developed by Huawei, with emphasis on its encoding modes and its support characteristics for special values. Compared with the traditional FP8 (E4M3/E5M2) formats, HiF8 significantly improves the dynamic range and representation precision through an optimized allocation of the exponent and mantissa. On this basis, HiF8 proposes an innovative universal floating-point encoding/decoding method, allowing its field widths, dynamic range, and precision to be scaled according to the requirements of specific scenarios. Notably, HiF8 additionally introduces the **Dot** field. As shown in Table 1, the format consists of four core fields: **Sign**, **Dot**, **Exponent**, and **Mantissa**.

**Table 1: Field structure of HiF8**

| Field | Width (bit) | Description |
| - | - | - |
| Sign (S) | 1 | Sign of the value, where 1 indicates negative and 0 indicates positive |
| Dot (D)                                  | 2\~4 | Indicates the width of the exponent field and the **Denormal encoding mode (DML)** |
| Exponent (E)                                  | D bits | Uses sign-magnitude encoding |
| Mantissa (M)                                  | 1\~3 | Mantissa bits with the leading 1 omitted

The following describes each field in detail:

* Sign Field: 1 bit, used to determine the positive or negative sign of a HiF8 value. By default, 1 indicates a negative sign and 0 indicates a positive sign.
* Dot Field: Uses prefix encoding, where different encoding lengths correspond to different Dot values, as shown in Table 2.

**Table 2: Dot encoding**

| Width (bits) | Encoding | Encoded value |
| - | - | - |
| 2 | 11 | 4 |
| 2 | 10 | 3 |
| 2 | 01 | 2 |
| 3 | 001 | 1 |
| 4 | 0001 | 0 |
| 4 | 0000 | **DML** |

* Exponent Field: D bits (including one implicit leading numeric bit with value 1 that is not stored), where D equals the encoded value of the Dot field, and D∈{0, 1, 2, 3, 4}.
* Mantissa Field: 1 to 3 bits, encoded as an unsigned integer. For HiF8 normalized numbers, the mantissa represents the fractional part of the significand, i.e., the part to the right of the binary point.

### 1.1 Normal Encoding Mode

![](../figures/image1.png)
**Figure 1: Schematic of the HiF8 Normal encoding mode structure and its exponent magnitude allocation**

Figure 1 shows the Normal (normalized) encoding mode of HiF8. To ensure that the exponent ranges corresponding to exponent fields of different widths do not overlap and to achieve **redundancy-free encoding**, HiF8 adopts the **Sign-Magnitude** encoding scheme and treats the most significant magnitude bit as an implicit bit that does not occupy actual storage space.

Specifically:

* When Dot = 0, the exponent occupies no bit width, and E is directly equal to 0;
* When Dot = 1\~4, the most significant bit of the exponent magnitude is fixed to 1 (the red digits in Figure 1).

Under this rule:

* With Dot = 1, the exponent E can only equal ±1, excluding the already-encoded 0;
* With Dot = 2, the exponent E can only represent ±[2, 3], excluding the already-encoded [-1, 1]. By analogy, the five exponent widths 0\~4 indicated by the Dot field jointly encode the exponents in [-15, 15] without overlapping one another, thereby achieving redundancy-free encoding.

In addition, HiF8 adopts a design in which a wider Dot field matches a narrower (smaller-magnitude) exponent field, thereby achieving a smooth gradient in mantissa precision rather than an abrupt step change. It can be observed that when E = [-3, 3], there is a 3-bit mantissa. Then, as the exponent magnitude grows, the mantissa width in Normal mode gradually shrinks to 1 bit. This gradually varying precision is exactly what makes HiF8 superior to the Posit data format from which it draws inspiration.

### 1.2 Denormal Encoding Mode

![](../figures/image2.png)
**Figure 2:**Schematic of the HiF8 Denormal encoding mode structure and its exponent magnitude allocation****

To support a larger dynamic range, HiF8 also adopts a Subnormal / Denormal scheme that differs from the IEEE 754 style. As shown in Figure 2, in **Denormal** (non-normalized) mode:

* **Encoding logic switch**: When the Dot field indicates that the current value is in Denormal mode, HiF8 no longer retains an independent exponent field. At this point, the original 3-bit mantissa space (which can encode 8 discrete values from 0 to 7) is **directly reused to map and extend the exponent representation range of extremely small values**.
* **Exponent range extension**: On top of the [-15, 15] range (31 exponents in total) supported by **Normal** mode, Denormal mode additionally opens up [-22, -16], a total of 7 exponents. Among them, encoding 0 corresponds to -23 and is used to represent the special values **Zero** and **NaN**.
* **Performance benchmarking**: As a result, HiF8 achieves a large dynamic range covering [-22, 15], a total of 38 exponents, which is already very close to the [-24, 15] exponent coverage of FP16. At the same time, since the [-15, -8] interval itself has a 1-bit mantissa and the newly extended [-22, -16] interval corresponds to a 0-bit mantissa, the two maintain a continuous transition in representation precision rather than an abrupt change.

### 1.3 Special Values

**Table 3: Definition of HiF8 special values**

| Condition | Output X | HiF8 (binary) | HiF8 (hexadecimal) | Meaning |
| - | - | - | - | - |
| S=0, D=DML, M=3'b0 | **Zero** | `00000000` | `0x00` | Zero |
| S=1, D=DML, M=3'b0                                                   | **NaN** | `10000000` | `0x80` | Not a Number |
| D=4, Es=4'b0111,  M=1                                                | **±Inf** | `X1101111` | `0x6F`/`0xEF` | Positive/negative infinity |

As shown in Table 3, in addition to conventional numeric encoding, HiF8 also fully supports the core special values such as Zero, NaN, and ±Inf. HiF8 does not distinguish between positive 0 and negative 0; instead, it uses a single encoding mode for a unified representation. This design effectively saves encoding space and makes HiF8 a complete and efficient 8-bit single format for numeric representation.

### 1.4 Core Features of HiF8

HiF8 possesses a series of excellent features tailored specifically for large model (LLM) training, with the core advantages mainly reflected in the following three aspects:

* **Tapered Precision matching the data distribution**: Traditional FP8 (E4M3/E5M2), due to its fixed exponent bit width, often finds it difficult to simultaneously satisfy the dynamic range and effective precision required by deep learning. Given that the parameters and activations of neural networks usually exhibit a Gaussian-like distribution, HiF8 innovatively adopts a **tapered precision distribution** (as shown in Figure 3), concentrating higher precision in the frequently occurring value ranges, thereby better matching the characteristics of model data.

![](../figures/image3.png)

**Figure 3: Significant-bits vs. exponent diagram of 8-bit floating-point formats**

* **Redundancy-free encoding**: The exponent of HiF8 uses sign-magnitude encoding, and the fixed 1-bit high-order position is set as an implicit bit that does not occupy actual storage space, thereby ensuring that the representation ranges of exponents with different bit widths do not overlap, ultimately achieving redundancy-free encoding.
* **Large dynamic range**: Through the special **Denormal** design, the combined exponent range is extended from [-15, 15] to [-22, 15], covering 38 exponent values in total, which is already close to the range of 40 combined exponent values of FP16. Table 4 summarizes and compares the typical values and key characteristics of HiF8, FP8, and FP16. It can be seen that, under the 8-bit width constraint, HiF8 significantly expands the dynamic range of the format through the tapered precision characteristic that matches the data distribution, while ensuring the precision required for neural network training and inference, providing a more comprehensive 8-bit single format representation for neural networks.

**Table 4: Typical values and key characteristics of HiF8, FP8, and FP16**

![](../figures/image4.png)

## Chapter 2 Quantization Scheme

When performing quantization, the core of converting high-precision data (such as FP32/BF16) to low-bit formats (such as FP8) lies in **scaling**. For a given data block X, it is necessary to multiply by a **scaling factor (Scale)** so that the original values in X can be safely mapped into the dynamic range of the low-bit format without overflow. Formulas (1) and (2) describe the basic scaling principle in the quantization process. Here, F8max represents the maximum value representable by the target 8-bit floating-point format (for example, the maximum value of FP8-E4M3 is 448), and Amax represents the maximum absolute value of the elements in data block X. Formula (1) is used to compute the **scaling factor (Scale)**, and formula (2) is used to map the high-precision data into the low-bit representation range.

$$
\mathrm{Scale} = \frac{A_{\max}}{F8_{\max}} \tag{1}
$$

$$
X_{\mathrm{scaled}} = X \cdot \mathrm{Scale} = X \cdot \frac{A_{\max}}{F8_{\max}} \tag{2}
$$

Although 8-bit floating-point (FP8) has a natural advantage in dynamic range compared with fixed-point formats, its precision representation capability and numerical coverage are still limited under the constraint of only 8 bits. To minimize the quantization loss when converting from high precision (such as FP16/BF16) to FP8, the industry usually introduces a "fine-grained quantization" strategy. As shown in Figure 4, this strategy greatly increases the density of scaling factors to adjust the data mapping range in real time, mainly reflected in the following two dimensions:

* Per-Token fine granularity for activations (for dynamic distributions): Because activations in an LLM usually exhibit large dynamic range differences across different Token dimensions, accompanied by frequent outliers. Per-Token scaling applies an independent scaling factor to each input Token (or Tile) to ensure that the core data range of each Token can be effectively covered by the FP8 format, thereby maintaining quantization precision.
* Per-channel / Per-Block fine granularity for weights (for static distributions): Model weights usually have different distribution characteristics across different output channels or weight blocks. Per-Channel / Per-Block scaling applies an independent scaling factor to each channel or each weight block to further reduce quantization error and improve model inference precision. A typical Block size can be set to `128 × 128`.

However, this extreme pursuit of precision introduces a large number of Per-Token / Channel / Block scaling factors, which significantly increases the complexity of hardware implementation and control logic. During inference, the system needs to frequently load, store, and apply these additional scaling factors, resulting in extra memory bandwidth overhead and computation latency that partially offset the acceleration benefits brought by 8-bit floating-point computation itself.

![](../figures/image5.png)

**Figure 4: Schematic of common quantization granularities**

Because HiF8 has a larger **dynamic range**, coarse-grained Per-Tensor quantization can often be adopted in training, while formats such as FP8 and MXFP8 often require Per-Block quantization.

## Chapter 3 Training Scheme

### 3.1 Common Training Schemes

Before introducing quantization training schemes based on low-bit formats (such as HiF8), it is necessary to first review the most widely used baseline scheme in the industry today — mixed-precision training based on BF16.

* **BFloat16 (BF16) format**: BF16 is a 16-bit floating-point format specifically designed for Deep Learning, proposed in 2017. In terms of memory layout, it consists of a 1-bit Sign, an 8-bit Exponent, and a 7-bit Mantissa. Compared with the traditional FP16 (5-bit Exponent, 10-bit Mantissa), BF16 sacrifices Mantissa precision but retains exactly the same Exponent width as the high-precision data type FP32. This means that BF16 has the same dynamic range as FP32 (approximately $10^{38}$ to $10^{-38}$). In neural network training, the distributions of model parameters and gradients usually span an extremely wide range of values; using BF16 can effectively avoid overflow or underflow caused by an insufficient representation range, while its smaller bit width can also significantly reduce memory usage and memory bandwidth overhead.
* **BF16-based mixed-precision training**: Mainstream deep learning frameworks typically use the Automatic Mixed Precision (AMP) mechanism when supporting BF16 training. This paradigm selects appropriate data types at the graph or operator level: in the forward pass and backward pass, the framework automatically down-converts the inputs of compute-intensive operators (such as MatMul and Convolution) to BF16 to fully exploit the acceleration advantages of the underlying hardware; for operators that are highly sensitive to numerical precision (such as Softmax, LayerNorm, and Loss computation), it keeps FP32 to ensure numerical stability. In addition, during the parameter update step of the Optimizer, the system usually maintains a set of FP32 high-precision Master Weights in memory, thereby avoiding weight update stalls caused by insufficient BF16 Mantissa precision.

On the basis of the above mixed-precision schemes, 8-bit mixed-precision training based on scaling techniques can further improve training efficiency.

### 3.2 Scaling Schemes

Current mainstream 8-bit mixed-precision training schemes usually involve two types of scaling strategies: Current Scaling and Delayed Scaling. As shown in Figure 5, Current Scaling traverses the high-precision data in the input ***X*** at every training step, obtains the maximum absolute value Amax, and completes scaling based on this Amax value. This means that the process of computing Amax and calculating Scale must be executed in every iteration and is serialized with downstream computations (such as MatMul), partially offsetting the performance benefits brought by 8-bit. In contrast, Delayed Scaling shown in Figure 6 uses historical statistics to estimate the Amax of the current training step, decoupling the operation of traversing the input ***X*** to obtain Amax from subsequent computations, thereby more fully exploiting the performance advantages of 8-bit training.

![](../figures/image6.png)

**Figure 5: Current Scaling computes Amax per iteration and scales the input X into the Float8 representation range**

![](../figures/image7.png)

**Figure 6: Delayed Scaling computes Amax over an interval = K iterations and scales the input X into the Float8 representation range, where N_guard is a protection parameter that prevents overflow**

FP8-E4M3, commonly used in 8-bit mixed-precision training, often requires fine-grained **Current Scaling** because of its small dynamic range, whereas HiF8, benefiting from the balance it achieves between precision and dynamic range, can adopt **Delayed Scaling** and estimate the **Amax** value over a longer interval while still maintaining training stability.

## Chapter 4 Low-Precision Training Frameworks

### 4.1 The MindSpeed Low-Precision Training Framework

The HiF8 and MXFP8 mixed-precision validation described later in this document is carried out based on the MindSpeed and MindSpeed LLM training frameworks. MindSpeed Core is a large-model acceleration library for Huawei Ascend devices. It helps large-model training and inference tasks migrate efficiently to the Ascend platform and supports Ascend-specific optimization capabilities. On the basis of MindSpeed Core, the system further provides the MindSpeed LLM component for large language model scenarios, and its overall framework is shown in Figure 7.

![](../figures/image8.png)

**Figure 7: Schematic of the MindSpeed and MindSpeed LLM frameworks**

Transformer Engine (TE) is an acceleration library for Transformer model training and inference. It supports 8-bit floating-point (FP8) precision on Hopper GPUs to reduce memory usage and improve training and inference performance. The TE module of MindSpeed remains compatible with the official TE in terms of interfaces and capabilities, supports seamless migration of third-party frameworks, and can achieve efficient large language model training and inference on Ascend NPUs.
The core of the MindSpeed low-precision training flow is to quantize the general matrix multiplication operations in the forward propagation (Fprop), activation backward propagation (Dgrad), and weight backward propagation (Wgrad) to FP8 / HiF8 precision for execution. The overall training flow is still based on the BF16 / FP16 AMP mixed-precision mode, but in specific compute operators, especially the matrix multiplication operations in the Linear layer (covering Fprop, Dgrad, and Wgrad), the FP8 / HiF8 precision path is used.
![](../figures/image9.png)

**Figure 8: Implementation of the Transformer Engine quantization code modules**

As shown in Figure 8, TERowParallelLinear and TEColumnParallelLinear correspond to the Linear layer implementations, and their core is to replace the MatMul operations in Fprop, Dgrad, and Wgrad with the FP8 / HiF8 low-precision path. FP8GlobalStateManager is a singleton object used to maintain the global FP8 state and the runtime information of each TE Module. Each Recipe class is responsible for executing a different quantization strategy and manages the Scale and data tensors through the corresponding Tensor classes.

In the process of quantizing high-precision tensors to low-precision tensors, the FP8 data format usually supports the following types of scaling strategies:

* Delayed Scaling: Computes the scaling factor based on historical Amax values, and then uses this scaling factor to quantize the Tensor;
* Tensorwise Scaling: Computes Amax online and applies the scaling factor in real time to quantize the Tensor;
* Blockwise Scaling: Divides the Tensor into blocks, then computes Amax for each block and applies the scaling factor to quantize the Tensor;
* MX Scaling: Converts floating-point vectors into MX blocks through a combination of block-level shared Scale and low-bit-width elements, achieving dynamic quantization.

The currently supported low-precision data formats include MXFP8 (E4M3), MXFP8 (E5M2), and HiF8, and the corresponding parameter combinations are shown in Table 5.

**Table 5: Parameter combinations supported by MindSpeed low-precision training**

| Data format | Quantization granularity | MatMul | Grouped MatMul | TP-MC2 | MatMul Add | Grouped MatMul Add |
| - | - | - | - | - | - | - |
| FP8 | Delayed | ✅ | Tensorwise path | ❌ | ❌ | ❌ |
| FP8                                                                                  | Tensorwise | ✅ | ✅ | ❌ | ❌ | ❌ |
| FP8                                                                                  | Blockwise | ✅ | BF16 fallback | ❌ | ❌ | ❌ |
| FP8                                                                                  | MXFP8 | ✅ | ✅ | ✅ | ✅ | ✅ |
| HiF8                                                                                 | Delayed | ✅ | Tensorwise path | ❌ | ❌ | ❌ |
| HiF8                                                                                 | Tensorwise | ✅ | ✅ | ❌ | ❌ | ❌ |

Notes:

* "✅" indicates that the corresponding operator supports a low-precision implementation under the current combination;
* "❌" indicates that the corresponding operator does not support a low-precision implementation under the current combination;
* "Tensorwise path" indicates that the corresponding operator is currently executed using the Tensorwise quantization approach;
* "BF16 fallback" indicates that the corresponding operator falls back to the BF16 precision path under the current combination;
* "TP-MC2" indicates a communication-and-computation fused operator.

### 4.2 The TorchTitan Low-Precision Training Framework

TorchTitan is a native distributed training framework developed by the PyTorch community for GPU environments, with the following characteristics:

* Supports applying 3D parallelization in a modular, composable way, as shown in Figure 9. It also provides elastic scaling to adapt to changing compute demands;
* Provides comprehensive logging, checkpoint storage, and debugging capabilities to meet production-grade training requirements;
* Integrates software-hardware co-design capabilities, and can improve hardware utilization in combination with mechanisms such as Float8 training and Symmetric Memory;
* Provides engineering references for the selection and combination of distributed training techniques based on practical experience with TorchTitan.

![image](../figures/image10.png)

**Figure 9: Schematic of the modular and composable initialization flow of TorchTitan**

TorchTitan supports float8 and MXFP8 quantized training for dense and MoE models based on Blackwell GPUs. The general flow of its low-precision quantized training is shown in Figure 10: before calling `model_parts` to build the model, the system executes model conversion through `model_converters.convert`. When the MXFP8 configuration is enabled, ordinary linear layers are replaced with low-precision linear layers.

For MXFP8, there are currently two main types of conversion paths: MXLinearConverter and MXGroupedMMConverter, targeting the linear layer and the MoE layer, respectively. The two conversion flows are generally similar, mainly including two stages: initialization (init) and model conversion (convert).

* **Initialization stage (init)**: Mainly verifies the current environment, such as whether TorchAO is installed and whether the current hardware environment supports MXFP8, and completes the relevant Python package imports and configuration initialization. After initialization is complete, the enable flag is set to 1.
* **Conversion stage (convert)**: After confirming that the enable flag is valid, the system imports the relevant TorchAO Python packages, reads the config configuration, and calls `quantize_` to complete the model conversion. The core parameters of `quantize_` include the model to be converted `model`, the quantization configuration `config`, and `filter_fn`, which is used to filter the layers to be converted.

![](../figures/image11.png)

**Figure 10: Schematic of the TorchTitan low-precision quantization implementation**

torchtitan-npu is a repository planned to be open-sourced on GitCode. It is built on the TorchTitan training flow and, while retaining the ease of use of TorchTitan, provides a stable, reproducible, and easy-to-analyze training framework for the Ascend platform. Since the HiF8 and MXFP8 mixed-precision validation described later in this document is not carried out based on this training framework, this section only briefly introduces its low-precision implementation path, and more details will be supplemented in subsequent versions.

## Chapter 5 HiF8 and MXFP8 Mixed-Precision Validation

### 5.1 HiF8 Mixed-Precision Training Experiment

The HiF8 experiment is conducted on the 950DT platform, carrying out mixed-precision pretraining validation based on a trimmed DeepSeek-V3 model. The current experiment only quantizes the **MatMul (MM)** and **Grouped MatMul (GMM)** operators in the entire network; the other numerically sensitive operators remain on higher-precision paths to ensure training stability.

The experimental configuration is as follows:

* **Model**: Trimmed DeepSeek-V3
* **Training framework**: MindSpeed / MindSpeed LLM
* **Hardware platform**: 950DT-8P, single-card compute power 486 TFLOPS
* **Quantization coverage**: MM / GMM operators
* **Low-precision format**: HiF8
* **Quantization granularity**: Per-tensor
* **Scaling strategy**: Delayed Scaling
* **Update interval**: `interval = 10 steps`
* **Control group**: BF16 mixed-precision training

The observed experimental metrics include:

* Training Loss
* Gradient Norm
* Per-step end-to-end training time

Figure 11 shows the training metrics obtained on the 950DT platform with the trimmed DeepSeek-V3 model and the Delayed Scaling (interval = 10 steps) setting, including the loss function and Gradient Norm. The experimental results show that the loss curves and Gradient Norm of HiF8 and BF16 are essentially aligned, indicating that HiF8 achieves training precision similar to BF16 under this setting.

![](../figures/image12.png)

**Figure 11: BF16 vs. HiF8, training loss (upper left), relative difference in training loss (lower left), Gradient Norm (upper right), and relative difference in Gradient Norm (lower right)**

Under this experimental setting, the end-to-end time per iteration of BF16 training is 3.63629095 s, while HiF8 training takes only 2.80796152 s. Compared with BF16, HiF8 achieves a 29.5% end-to-end training speedup.

### 5.2 MXFP8 Mixed-Precision Training Experiment

To compare the differences between HiF8 and existing fine-grained 8-bit floating-point training schemes, this document further conducts MXFP8 (E4M3) mixed-precision training experiments on the same platform and model configuration.

The experimental configuration is as follows:

* **Model**: Trimmed DeepSeek-V3
* **Training framework**: MindSpeed / MindSpeed LLM
* **Hardware platform**: 950DT-8P, single-card compute power 486 TFLOPS
* **Quantization coverage**: MM / GMM operators
* **Low-precision format**: MXFP8 (E4M3)
* **Quantization granularity**: Per-block
* **Scaling method**: Fine-grained block-level scaling
* **Control group**: BF16 mixed-precision training

By keeping the model structure, training task, and quantization coverage consistent, the differences between HiF8 and MXFP8 in training stability and performance benefits can be compared fairly.

Figure 12 shows the long-stability precision comparison between MXFP8 (E4M3) quantization and BF16 on the 950DT platform with the trimmed DeepSeek-V3 model. The experimental results show that the loss curves and Gradient Norm of MXFP8 and BF16 are essentially aligned, indicating that MXFP8 achieves training precision similar to BF16 under this setting.

![](../figures/image13.png)

**Figure 12: BF16 vs. MXFP8, training loss (upper left), relative difference in training loss (lower left), Gradient Norm (upper right), and relative difference in Gradient Norm (lower right)**

Under this experimental setting, the end-to-end time per iteration of BF16 training is 3.63629095 s, while MXFP8 training takes only 3.12526599 s. Compared with BF16, MXFP8 achieves a 14.05% end-to-end training speedup.

## Chapter 6 Benefit Analysis

This document performs a statistical analysis of the end-to-end time of **HiF8** and **MXFP8** training tasks. By comparing the overall execution time under different data formats, the differences of **HiF8** in training throughput and implementation overhead can be observed. The specific results are shown in Table 6.

**Table 6: Comparison of training performance among HiF8, MXFP8, and BF16**

| Metric | HiF8 time (s) | MXFP8 time (s) | BF16 time (s) | HiF8 speedup over MXFP8 | HiF8 speedup over BF16 |
| - | - | - | - | - | - |
| Time per iteration | 2.80796152 | 3.12526599 | 3.63629095 | **1.113x** | **1.295x** |

As can be seen from Table 6, under the current experimental setting:

* HiF8 achieves approximately **11.3%** training speedup over MXFP8;
* MXFP8 achieves approximately **14.05%** training speedup over BF16;
* HiF8 achieves approximately **29.5%** training speedup over BF16.

It should be noted that the current experiment only quantizes the **MM** and **GMM** operators in the entire network, so the currently measured benefits are end-to-end benefits under partial operator quantization. If HiF8 is later extended to more operators suitable for low-bit quantization, there is still room for further improvement in the overall benefits.

### 6.1 Source of HiF8's Benefits over BF16

Compared with BF16, the performance gain of HiF8 mainly comes from the memory-access benefits brought by the reduced data bit width, as well as the throughput advantages of the low-bit computation path.

#### 6.1.1 Memory-Access Benefits from Halving the Data Bit Width

When the data bit width is reduced from 16-bit BF16 to 8-bit HiF8, the data movement volume of the inputs, outputs, and intermediate activations of matrix-multiplication-related operators decreases accordingly. In large-model training, some operators are often limited by memory bandwidth, so halving the bit width can usually bring the following benefits:

* **Reduced on-chip memory bandwidth pressure**: The amount of data transferred per unit time decreases, thereby easing the read/write pressure on the high-bandwidth on-chip memory;
* **Improved on-chip cache utilization**: With the same on-chip SRAM / Cache capacity, more data can be accommodated, which helps improve the data reuse rate;
* **Reduced data movement latency**: The data transfer overhead between different storage levels, such as from on-chip memory to registers, decreases, thereby shortening the waiting time in non-compute stages.

#### 6.1.2 The Throughput-Improving Effect of 8-bit Computation

In addition to the memory-access benefits, HiF8 can also exploit the higher throughput that the underlying hardware provides for low-bit matrix operations. Current AI acceleration hardware is usually specifically optimized for the 8-bit computation path, so when backbone operators such as MM and GMM are mapped to the HiF8 path, compute utilization can be further improved and translated into training speed gains. As shown in Figure 13, the average speedup of HiF8 over BF16 for the MM and GMM operators exceeds 1.8x.
![](../figures/image14.png)

**Figure 13: Comparison schematic of the MM operator (left) and GMM operator (right) for HiF8 vs. BF16**

### 6.2 Source of HiF8's Benefits over MXFP8

Compared with MXFP8, the additional benefits of HiF8 are mainly related to the quantization granularity and the scaling metadata overhead. In the trimmed DeepSeek-V3 network, the profiling comparison of the GMM operator under the HiF8 and MXFP8 quantization schemes is shown in Figure 14. Combined with the experimental results, the advantages of HiF8 over MXFP8 can be summarized into the following three aspects.

![](../figures/image15.png)

**Figure 14: Profiling comparison schematic of the GMM quantization operator for HiF8 vs. MXFP8**

#### 6.2.1 Lower Scaling Metadata Overhead

MXFP8 usually adopts **Per-Block** fine-grained quantization, which requires maintaining additional scaling factors for each data block and frequently accessing this metadata during computation. In contrast, because HiF8 has a larger dynamic range, it can usually adopt **Per-Tensor** quantization, thereby reducing the number of scaling factors and their memory-access overhead. The corresponding benefits mainly include:

* Fewer load and store operations for scaling factors;
* Reduced extra memory overhead for storing fine-grained Scale;
* More contiguous data access patterns, reducing alignment and access issues caused by fragmented metadata.

#### 6.2.2 Relatively Simple Control Logic

**Per-Block** quantization requires the operator to handle logic such as block boundary switching, local dequantization, and cross-block Scale alignment during execution, which increases control complexity and affects operator fusion and pipelined execution. In contrast, when HiF8 adopts the **Per-Tensor** path, the internal data flow of the operator is more contiguous and the related control logic is simpler, making it easier to achieve stable execution efficiency.

#### 6.2.3 More Suitable for Delayed Scaling Training

HiF8 has a larger dynamic range, so during training it can adopt Delayed Scaling with a lower update frequency without significantly affecting training stability. This gives it certain advantages in the following aspects:

* **Lower frequency of Amax statistics**;
* **Reduced Scale updates and related synchronization operations**;
* **Less interference with the main computation flow**.

In contrast, FP8 / MXFP8 with smaller dynamic ranges often need to update scaling parameters more frequently, or rely on finer-grained scaling strategies, which adds extra control overhead.

### 6.3 Current Conclusions and Future Optimization Space

Judging from the current experimental results, HiF8 shows good application potential in large-model training scenarios.

On the one hand, under the existing experimental configuration, the loss curve and gradient norm of HiF8 are essentially aligned with those of BF16, indicating that it can maintain good training stability on the current task. On the other hand, in terms of end-to-end performance, HiF8 is superior to both BF16 and MXFP8, indicating that it achieves a good balance among numerical range, quantization granularity, and system overhead.

From an implementation perspective, HiF8 has a low dependence on fine-grained scaling and large amounts of metadata, which helps reduce the complexity of operator implementation and facilitates subsequent scaling to larger-scale training tasks.

Future work can continue in the following directions:

* **Expanding operator coverage**: Extending HiF8 quantization to more operators suitable for low-bit conversion, such as **FlashAttention (FA)** and related collective communication operators;
* **Optimizing the scaling update mechanism**: Further researching Amax statistics and Scale update strategies better suited to HiF8, so as to reduce extra overhead and improve numerical stability;
* **Carrying out software-hardware co-optimization**: Combining compiler and underlying hardware instruction capabilities to further optimize the low-bit data path and operator fusion implementations.

## Chapter 7 Recommended Scheme and Implementation Guide

Based on the aforementioned format design, quantization approaches, and analysis of experimental results, this document summarizes a set of recommended configurations suitable for large-scale neural network training.

### 7.1 Recommended Configuration for HiF8 Training

While balancing training precision and system throughput, the recommended configuration is shown in Table 7.

**Table 7: Recommended configuration for HiF8 training**

| **Configuration dimension** | **Recommended scheme** | **Core motivation** |
| - | - | - |
| **Quantization granularity** | **Per-tensor** | Maximizes the advantages of the HiF8 format and eliminates the memory-access and control burden brought by Per-block. |
| **Scaling strategy**                                                                                      | **Delayed Scaling** | Reduces the metadata overhead and control complexity caused by Per-Block quantization |
| **Update interval**                                                                                      | **Interval = 10 steps** | Leverages the dynamic range advantage of HiF8 to reduce the update frequency of scaling parameters |
| **Operators covered**                                                                                      | **MM / GMM** | Prioritizes covering compute-intensive backbone operators |
| **Master Weights**                                                                                | **FP32 / BF16** | Ensures numerical precision during the parameter update stage |

When the model and task allow, it is recommended to prioritize Per-Tensor quantization to reduce Scale metadata overhead and implementation complexity. For the scaling strategy, it is recommended to prioritize Delayed Scaling and set the update interval to 10 steps or higher to reduce statistics and synchronization overhead. For the quantization coverage, one can start with compute-intensive operators such as MM and GMM, and then gradually expand to other operators suitable for low-bit conversion.

### 7.2 Core Scheme Optimization Recommendations

When implementing the above scheme, it is recommended to follow the following principles to balance training precision, system efficiency, and implementation complexity.

* **Prioritize Per-Tensor quantization.**
  Because HiF8 has a large dynamic range, Per-Tensor quantization can usually already meet the precision requirements in most LLM training scenarios. Compared with fine-grained quantization, this approach can significantly reduce the storage and access overhead of Scale metadata, and reduce memory-access fragmentation and control complexity.
* **Prioritize Delayed Scaling.**
  It is recommended to combine the numerical range advantage of HiF8, adopt the Delayed Scaling strategy with a lower update frequency, and set the update interval to 10 steps or higher. This can reduce the extra overhead caused by Amax statistics, Scale updates, and related synchronization operations, which is beneficial for improving the continuity of the main computation flow.
* **Evaluate the scheme comprehensively from the three aspects of numerical capability, training stability, and system overhead.**
  Through the Dot field, redundancy-free exponent encoding, and Denormal extension, HiF8 provides a large dynamic range at 8-bit width; under the experimental setting in this document, its loss curve and gradient statistics are essentially aligned with those of BF16, demonstrating good training stability; at the same time, HiF8 supports coarser-grained quantization and longer-interval scaling updates, which helps reduce metadata access and control logic overhead and achieve good end-to-end performance benefits.

### 7.3 Summary

This document analyzes the data format design, quantization scheme, training framework support, and experimental results of HiF8. The experiments show that, on the 950DT platform and the trimmed DeepSeek-V3 model, HiF8 achieves a 29.5% end-to-end training speedup over BF16 while maintaining training precision similar to BF16, and obtains an 11.3% performance improvement over MXFP8.

Judging from the current results, HiF8 achieves a good balance among numerical range, training stability, and system overhead. **For large-model training scenarios that seek to reduce memory-access pressure, improve training throughput, and control implementation complexity, HiF8 is an 8-bit mixed-precision training scheme with engineering application value**.

In the future, a HiF8 pretraining example based on the trimmed DeepSeek-V3 model will be added to the [cann-recipes-train ](https://gitcode.com/cann/cann-recipes-train) open-source repository for subsequent validation and reproduction.

<!-- static-meta
{
  "sidebarTitle": "HiF8/MXFP8 Pretrain",
  "quantization": ["hif8", "mxfp8"],
  "hardware": ["ascend-950"],
  "frameworks": ["mindspeed"],
  "cover": {"type": "auto"}
}
-->
