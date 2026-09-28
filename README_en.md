<h1 align="center">CANN-RECIPES-DOCS</h1>

<p align="center">
  CANN Recipes Technical Documentation Center<br>
  A collection of inference, training, embodied AI, and CANN atomic feature optimization documents that captures reusable model optimization experience and engineering practices
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Docs-Online%20Preview-orange.svg" alt="Online Preview"></a>
  <img src="https://img.shields.io/badge/Content-CANN%20Recipes-blue.svg" alt="CANN Recipes">
</p>

<p align="center">
  <a href="#-overview">📖 Overview</a> ·
  <a href="#-content-categories">🗂️ Content Categories</a> ·
  <a href="#-documentation-website">🧭 Usage</a> ·
  <a href="#-project-structure">📁 Project Structure</a> ·
  <a href="#-contributing">🤝 Contributing</a>
</p>

---

## 📖 Overview

This repository is the unified technical documentation repository for CANN Recipes. It primarily contains model optimization documents, scenario-specific practice documents, and descriptions of CANN atomic features derived from recipe code repositories.

The documentation serves the following objectives:

| Objective | Description |
|---|---|
| Capture optimization experience | Records practices for model adaptation, performance optimization, parallel partitioning, quantization, and operator optimization. |
| Provide a unified content portal | Centrally maintains technical documentation for **inference, training, and embodied AI**, and provides documentation for CANN atomic features. |
| Support structured search | Uses the `static-meta` fields at the end of each document to support category filtering, sidebar titles, and cover selection. |

> The online site only provides a portal for reading and searching documents. Technical documentation remains the core content of this repository.

---

## 🧭 Source Code and Runtime Environment

### Obtain the Source Code

Run the following commands to obtain the repository source code:

```shell
git clone https://gitcode.com/cann/cann-recipes-docs.git
cd cann-recipes-docs
```

### Runtime Environment

The Python scripts in this repository require **Python 3.11**. Run the following command to verify the Python version:

```shell
python --version
```

### Documentation Website

The repository includes a static documentation website under [`utils/site/`](utils/site/). The website reads documents and image files from the current repository. Run the following commands from the repository root to generate the indexes and start the local website:

```shell
python utils/site/build_content.py
python utils/site/serve.py --port 8001
```

Then, go to `http://127.0.0.1:8001`. For the website development specifications and validation checklist, see [`blog_site_skill.md`](skill/blog_site_skill.md) (Chinese). For descriptions of website files, see [`site/README.md`](utils/site/README.md) (Chinese).

---

## 🗂️ Content Categories

| Directory | Summary | Related Code Repository |
|---|---|---|
| [infer/](infer/) | **Inference**: Contains inference optimization documents for large language models, multimodal generation, and recommendation systems. | [cann-recipes-infer](https://gitcode.com/cann/cann-recipes-infer) |
| [train/](train/) | **Training**: Contains training optimization documents for pre-training, reinforcement learning, and long-sequence training. | [cann-recipes-train](https://gitcode.com/cann/cann-recipes-train) |
| [embodied_ai/](embodied_ai/) | **Embodied AI**: Contains optimization documents for 3D vision, robotic manipulation, navigation, and world models. | [cann-recipes-embodied-ai](https://gitcode.com/cann/cann-recipes-embodied-ai) |
| [cann_features/](cann_features/) | **CANN Atomic Features**: Contains documentation for cross-scenario common capabilities such as MultiStream, SuperKernel, Prefetch, and NPUGraph. | - |

### Inference

| Application Scenario | Directory | Sample Content |
|---|---|---|
| Large language models | `infer/llm/` | Inference optimization for models such as DeepSeek, Qwen, Kimi, LongCat, and GPT-OSS. |
| Multimodal generation | `infer/multimodal/` | Inference optimization for image generation, video generation, and multimodal models. |
| Recommendation systems | `infer/recommendation/` | Inference deployment and performance optimization for recommendation models. |

#### Sample List

**Large Language Models**

- [DeepSeek-R1](infer/llm/deepseek_r1/en/deepseek_r1_decode_optimization.md)
- [DeepSeek-V3.2-Exp](infer/llm/deepseek_v3_2_exp/en/deepseek_v3.2_exp_inference_guide.md)
- [DeepSeek-V4](infer/llm/deepseek_v4/en/deepseek_v4_inference_guide.md)
- [GPT-OSS](infer/llm/gpt_oss/en/gpt_oss_optimization.md)
- [Hy3](infer/llm/hy3/en/hy3_optimization.md)
- [Kimi-K2-Thinking](infer/llm/kimi_k2_thinking/en/kimi_k2_thinking_inference_guide.md)
- [Kimi K3](infer/llm/kimi_k3/en/kimi_k3_inference_guide.md)
- [LongCat-2.0](infer/llm/longcat_2_0/en/longcat_2.0_optimization.md)
- [LongCat-Flash](infer/llm/longcat_flash/en/longcat_flash_optimization.md)
- [Qwen Dense](infer/llm/qwen/en/qwen_dense_optimization.md)
- [Qwen3-MoE](infer/llm/qwen3_moe/en/qwen3_moe_optimization.md)
- [Qwen3.5-MoE](infer/llm/qwen3_5/en/qwen3_5_moe_optimization.md)

**Multimodal Generation**

- [daVinci-MagiHuman](infer/multimodal/davinci_magihuman/en/davinci-magihuman_optimization.md)
- [HunyuanImage-3.0](infer/multimodal/hunyuan_image_3/en/hunyuan_image_3_optimization.md)
- [HunyuanVideo](infer/multimodal/hunyuan_video/en/hunyuan_video_optimization.md)
- [SANA-Video](infer/multimodal/sana_video/en/SANA-Video_optimization.md)
- [Wan2.2-I2V](infer/multimodal/wan2_2_i2v/en/wan2.2-i2v_optimization.md)


**Recommendation Systems**

- [HSTU](infer/recommendation/hstu/en/hstu_optimization.md)


### Training

| Application Scenario | Directory | Sample Content |
|---|---|---|
| Pre-training | `train/pretrain/` | Optimization solutions for large model pre-training. |
| Reinforcement learning | `train/rl/` | RL training, long-sequence training, decoding strategy optimization, and related topics. |

#### Sample List

**Pre-training**

- [DeepSeek-V3](train/pretrain/deepseek_v3/en/deepseek-v3_pre_train_hif8_mxfp8.md)
- [DeepSeek-V3.2](train/pretrain/deepseek_v32/en/deepseek-v32_pre_train_optimization.md)
- [DeepSeek-V4](train/pretrain/deepseek_v4/en/deepseek-v4_torchtitan_npu_autofuse.md)

**Reinforcement Learning**

- [DeepSeek-R1](train/rl/deepseek/en/deepseek_rl_train_optimization.md)
- [Qwen3](train/rl/qwen3/qwen3_235B/en/qwen3_235B_32k_longseq_rl_train_optimization.md)


### Embodied AI

| Application Scenario | Directory | Sample Content |
|---|---|---|
| 3D vision | `embodied_ai/3d_vision/` | 3D generation, geometric reconstruction, rendering optimization, and related topics. |
| Robotic manipulation | `embodied_ai/manipulation/` | Optimization for policy models and robotic control models. |
| Navigation | `embodied_ai/navigation/` | Optimization for robot navigation models. |
| World models | `embodied_ai/world_model/` | Inference and training optimization for world models. |

#### Sample List

**3D Vision**

- [3DGS](embodied_ai/3d_vision/gaussian_splatting/en/gaussian_splatting_alpha_blending_optimization.md)
- [Hunyuan3D](embodied_ai/3d_vision/hunyuan3d/en/Hunyuan3D_optimization.md)
- [VGGT](embodied_ai/3d_vision/vggt/en/vggt_optimization.md)

**Robotic Manipulation**

- [GR00T N1.6](embodied_ai/manipulation/groot_n1_6/en/groot_n1_6_optimization.md)
- [Pi0](embodied_ai/manipulation/pi0/en/pi0_infer_with_torch.md)

**Navigation**

- [Alpamayo-R1](embodied_ai/navigation/alpamayo_r1/en/alpamayo_r1_optimization.md)

**World Models**

- [Cosmos](embodied_ai/world_model/cosmos/en/cosmos_optimization.md)


### CANN Atomic Features

| Application Scenario | Directory | Sample Content |
|---|---|---|
| Atomic features | `cann_features/` | Documentation for cross-scenario common capabilities such as MultiStream, SuperKernel, Prefetch, and NPUGraph. |

#### Sample List

- [NPU Multi-Stream Principles](cann_features/en/multi_stream_principles.md)
- [NPU Graph Mode Optimization Principles](cann_features/en/npu_graph_optimization.md)
- [NPU Prefetch Principles](cann_features/en/prefetch_principles.md)
- [SuperKernel Principles](cann_features/en/super_kernel.md)

---

## 📁 Project Structure

<details>
<summary>Expand the directory description</summary>

```text
.
├─ infer/                        # Inference optimization practice documents
│  ├─ llm/                       # Large language model inference documents
│  ├─ multimodal/                # Multimodal inference documents
│  └─ recommendation/            # Recommendation system documents
├─ train/                        # Training optimization practice documents
│  ├─ pretrain/                  # Pre-training documents
│  └─ rl/                        # Reinforcement learning training documents
├─ embodied_ai/                  # Embodied AI optimization practice documents
├─ cann_features/                # CANN atomic feature documents
├─ skill/
│  ├─ new_document_skill.md      # Skill for adding documents
│  ├─ translate_english_document_skill.md # Skill for translating documents into English
│  └─ blog_site_skill.md         # Documentation website development specifications
├─ utils/
│  ├─ references/
│  │  └─ static_meta_infos.md    # Static field specifications
│  ├─ scripts/
│  │  ├─ metadata_enums.py       # Static field enumerations
│  │  ├─ metadata_labels.py      # Chinese and English label mappings
│  │  └─ validate_metadata.py    # Static field validator
│  └─ site/                       # Static documentation website
│     ├─ index.html               # Website entry point
│     ├─ build_content.py         # Generates website indexes
│     ├─ serve.py                 # Local development server
│     └─ content/
│        └─ homepage_samples.json # Model sample configuration for the homepage
├─ .gitignore                    # Ignore rules for local and generated files
├─ CONTRIBUTION.md               # Contribution guide
├─ LICENSE                       # Apache 2.0 License
├─ DISCLAIMER.md                 # Disclaimer
├─ OAT.xml                       # Open-source compliance configuration
├─ README.md                     # Chinese project description
├─ README_en.md                  # English project description
├─ SECURITY.md                   # Chinese security policy
└─ SECURITY_en.md                # English security policy
```

</details>

---

## 🤝 Contributing

We welcome all forms of documentation contributions, including new optimization documents, improvements to existing content, English translations, and issue fixes.

For the document directory structure, static fields, and validation process, see the [Contribution Guide](CONTRIBUTION.md) (Chinese).

If you have questions or are interested in contributing, join the [Recipes SIG regular meeting](https://gitcode.com/cann/community/blob/master/CANN/sigs/recipes/README.md) (Chinese) to talk with us.

---

## 📝 License and Disclaimer

### License Applicability

- This repository as a whole is licensed under Apache License 2.0. For details, see [LICENSE](./LICENSE).
- If a directory contains a separate license file, that license applies. Otherwise, Apache License 2.0 applies.

### Other Notices

- For the full disclaimer, see [DISCLAIMER](DISCLAIMER.md) (Chinese).

---

<p align="center">
  <sub>CANN Recipes Technical Documentation Center</sub>
</p>
