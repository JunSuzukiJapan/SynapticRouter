[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) is a novel dynamic, sparse, and modular neural network architecture inspired by the biological brain (synapses).
Instead of a massive, static Transformer, SRA dynamically routes inputs to appropriate "synapses" (tiny modules) to achieve more efficient learning and structural intelligence.

## 🎯 Motivation

In recent years, while AI models have become increasingly massive, monolithic networks face significant challenges, such as "escalating computational resources" and "catastrophic forgetting during multi-task learning".
SRA attempts to solve these issues using a **sparse approach: "dynamically calling and combining only the necessary tiny modules (synapses) depending on the input."** This enables learning multiple tasks with different characteristics within the same network without interference, aiming to achieve both scalability and high learning efficiency.

## 🧠 Architecture Overview

SRA primarily consists of the following components:

1. **Synapse (Synaptic Module)**
   - Independent, tiny computational units (e.g., miniature Transformers or MLPs).
   - They specialize in specific functions or pattern processing through learning.
2. **Router**
   - Dynamically selects only the `Top-k` optimal synapses from all available synapses based on the input tokens, enabling sparse computation.
3. **Synapse Space**
   - Each synapse is placed in an embedding space, self-organizing so that the distance between synapses represents "functional similarity."
4. **Local Learning Rule**
   - In addition to standard backpropagation, it combines local learning rules using a 3-factor rule (like `trace × routing × reward` based on Hebbian, STDP, and reward), promoting load balancing and specialization.

## 📁 Directory Structure

The main structure of this repository is as follows:

- `src/` : Core implementations of the SRA model and scripts for training/evaluation.
  - `sra_gpu_models.py` / `sra_language_models.py` : SRA model implementations applying MoE (Mixture of Experts) techniques.
  - `train_mtl_algo.py` / `train_mtl_lang.py` : Execution scripts for multi-task learning (algorithmic reasoning and language modeling).
- `docs/` : Architectural Decision Records (ADR) and reports on various experiments and routing analysis.
- `data/` : Toy datasets used for training and validation (Code, Math, Text, etc.).
- `tests/` : Test code for various components.

## 🚀 Usage

### Requirements
```bash
pip install torch
```

### Basic Execution
You can test the model's training and inference using structural algorithmic tasks such as `copy`, `reverse`, `paren`, and `addmod`.

To run a single task:
```bash
python src/sra_experiment.py --task reverse --steps 2000
```

To sequentially run all tasks (task suite):
```bash
python src/sra_experiment.py --task-suite
```

### Comparison with Other Architectures
Scripts are provided to compare SRA with standard `Transformer` and `MLP` architectures.

```bash
# Compare architectures using the copy task
python src/compare_architectures.py --task copy --steps 500
```

## 📊 Comparison

> **📝 To be updated:**
> We are currently conducting quantitative comparative verification against standard Transformer models.
> Initial verifications (copy task with few steps) show promising signs that SRA reduces Validation Loss faster and reaches high accuracy quicker than Transformers (high learning/sample efficiency).
>
> Detailed benchmark results, including step-by-step throughput and VRAM efficiency improvements, will be added here in the future.

## 🧪 Experiments & Analysis

- [Multi-Task Learning and Routing Analysis in Algorithmic Reasoning](./routing_analysis_algorithmic.md)
  - A report verifying that SRA can simultaneously learn multiple algorithmic tasks without interference, and autonomously separate and modularize experts (synapses) according to the nature of the tasks.
- [Routing Analysis in Cross-Domain Language Modeling (Code / Math / Text)](./routing_analysis_language.md)
  - A report verifying the mechanism where SRA simultaneously learns domains with different grammars and vocabularies (code, math formulas, natural language), and synapses functionally differentiate (specialize) to perform inference for each domain.


---

### 🔌 6. Dynamic Synaptic Hot-Swap Experiment and Router Learning Limits
**File:** [`06_hotswap_experiment_demo_en.ipynb`](./06_hotswap_experiment_demo_en.ipynb)

Demonstrates the true power of SRA: "dynamic addition and removal of synapses as plugins (Hot-Swap)".
We perform an experiment where a Spanish-specific synapse is merged into a running French/German translation model.
In this notebook, you will learn the **crucial importance of sharing and freezing the base model's knowledge space (Embedding/Attention layers, etc.)** to establish a hot-swap. At the same time, you will confront the **greatest barrier of SRA (the vanishing gradient problem)**: standard hard routing (Top-k) cannot retroactively learn (differentiate) the routing of added synapses. This limitation serves as a critical foreshadowing for the subsequent "Gumbel-Softmax (Super Router)" section.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_en.ipynb)

---

### 👑 7. Model Integration via Super Router and Gumbel-Softmax
**File:** [`07_super_router_gumbel_demo_en.ipynb`](./07_super_router_gumbel_demo_en.ipynb)

We build a "Super Router" that bundles multiple specialized models (a FR/DE model and an ES model) and dynamically routes processing based on the input.
This demonstrates the "Lazy Routing" problem of simple Soft Routing and shows how using Gumbel-Softmax achieves **perfect Hard Routing**, cutting unnecessary model computation by 100%.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_en.ipynb)

## 🤝 Contributing & License

This project is currently an experimental architecture in its early stages. Bug reports, discussions on features, and pull requests for performance improvements are very welcome via Issues and PRs!

- **License**: This repository is released under the [MIT License](../../LICENSE). Please refer to the [`LICENSE`](../../LICENSE) file for details.
