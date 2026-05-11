[English](README.md) | [日本語](README_JP.md) | [中文](./docs/zh/README.md) | [한국어](./docs/ko/README.md) | [Español](./docs/es/README.md) | [Français](./docs/fr/README.md) | [Deutsch](./docs/de/README.md) | [Português](./docs/pt/README.md) | [Русский](./docs/ru/README.md) | [Italiano](./docs/it/README.md) | [العربية](./docs/ar/README.md) | [हिन्दी](./docs/hi/README.md)

# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) is a novel dynamic, sparse, and modular neural network architecture inspired by the biological brain (synapses).
Instead of a massive, static Transformer, SRA dynamically routes inputs to appropriate "synapses" (tiny modules) to achieve more efficient learning and structural intelligence.

## 🎮 Interactive Demos (Jupyter Notebooks)

We have prepared Jupyter Notebooks where you can interactively experience SRA's "task-specific brain usage" and "robustness" right in your browser. You can run them in seconds on Google Colab, so please give them a try!

| # | Demo | Description | Colab |
|---|------|-------------|-------|
| 🟢 1 | [SRA Quickstart](./notebooks/01_sra_quickstart_en.ipynb) | Basic SRA structure and routing visualization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_en.ipynb) |
| 🔵 2 | [Learning & Routing](./notebooks/02_learning_and_routing_demo_en.ipynb) | Single-task learning and routing specialization | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_en.ipynb) |
| 🔴 3 | [Multitask Routing](./notebooks/03_multitask_routing_demo_en.ipynb) ✨ | Multitask learning and synapse switching per task | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_en.ipynb) |
| 🕹️ 4 | [Decision Transformer](./notebooks/04_decision_transformer_routing_demo_en.ipynb) | Separation of perception and action in RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_en.ipynb) |
| 🧠 5 | [Lesion Experiment](./notebooks/05_lesion_experiment_demo_en.ipynb) ✨ | Proving functional modularity by destroying synapses | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_en.ipynb) |
| 🔌 6 | [Hot-Swap Experiment](./notebooks/06_hotswap_experiment_demo_en.ipynb) | Dynamic synaptic hot-swap and router learning limits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_en.ipynb) |
| 👑 7 | [Super Router (Gumbel)](./notebooks/07_super_router_gumbel_demo_en.ipynb) | Model integration via Gumbel-Softmax hard routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_en.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](./notebooks/08_sra_llm_demo_shakespeare_en.ipynb) | Build and train a Tiny LLM with SRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_en.ipynb) |
| 📚 9 | [Multidomain LLM](./notebooks/09_sra_llm_demo_multidomain_en.ipynb) | Multi-domain (Code/Math/Text) simultaneous learning | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_en.ipynb) |
| 💻 10 | [Plugin Hot-Swap](./notebooks/10_hotswap_plugins_demo_en.ipynb) | Zero-shot hot-swap with zero catastrophic forgetting | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_en.ipynb) |
| 🗑️ 11 | [Synapse Deletion](./notebooks/11_synapse_deletion_demo_en.ipynb) | Dynamic synapse deletion (pop & clear) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_en.ipynb) |

> 📓 For detailed descriptions of each notebook, see [`notebooks/README.md`](./notebooks/README.md).

## 🎯 Motivation

In recent years, while AI models have become increasingly massive, monolithic networks face significant challenges, such as "escalating computational resources" and "catastrophic forgetting during multi-task learning".
SRA attempts to solve these issues using a **sparse approach: "dynamically calling and combining only the necessary tiny modules (synapses) depending on the input."** This enables learning multiple tasks with different characteristics within the same network without interference, aiming to achieve both scalability and high learning efficiency.

## 💡 Basic Idea

Typical AI models (like Transformers) try to process everything using a single, giant "brain." However, with this approach, the computational burden becomes far too heavy every time the model is made smarter or larger. Therefore, SRA adopts a system where **many "small expert brains (which SRA calls 'synapses')" are prepared, and only the necessary experts are called upon depending on the problem at hand**.

The key here is the mechanism that decides "which expert to call." SRA has a "router (guide)," which instantly selects the most capable-looking expert by looking at the input data. As each expert becomes smarter (learns), this router simultaneously learns "who is the right one to choose," growing to be able to make optimal assignments automatically.

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

- [Multi-Task Learning and Routing Analysis in Algorithmic Reasoning](./docs/routing_analysis_algorithmic.md)
  - A report verifying that SRA can simultaneously learn multiple algorithmic tasks without interference, and autonomously separate and modularize experts (synapses) according to the nature of the tasks.
- [Routing Analysis in Cross-Domain Language Modeling (Code / Math / Text)](./docs/routing_analysis_language.md)
  - A report verifying the mechanism where SRA simultaneously learns domains with different grammars and vocabularies (code, math formulas, natural language), and synapses functionally differentiate (specialize) to perform inference for each domain.
- [Routing Analysis in Multilingual Translation (Eng / Fra / Jpn) & Zero-Shot Generality](./docs/dev/multilingual_translation_routing_analysis.md)
  - A fascinating report showing that SRA automatically assigns different translation modules based on the grammar structure (SVO vs SOV). Even more surprisingly, when asked to translate an unlearned language pair, it unconsciously uses English as a "pivot language" to solve the problem!
- [Complete Separation of Perception and Policy in Decision Transformer (Reinforcement Learning)](./docs/dev/decision_transformer_routing_analysis.md)
  - We gave SRA the ability to play a game. It discovered an incredible strategy on its own: it uses the *exact same "vision" module* to perceive the environment across all tasks, but switches to *completely different "brain" modules* depending on whether it needs to find treasure or run away from an enemy.
- [Verification of Practical Multilingual Translation using SRA Encoder-Decoder](./docs/dev/sra_seq2seq_translation_analysis.md)
  - A report demonstrating that by extending SRA to an Encoder-Decoder architecture and training for 30,000 steps on a real corpus (opus100), it can translate practical expressions like "Merci beaucoup." and "Good morning." with BLEU=1.0. The introduction of Cross-Attention caused a leap from Decoder-only (BLEU=0) to an overall average BLEU of 0.27, and achieved a near-practical accuracy of BLEU=0.56 in the FR→EN direction.



## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways](docs/paper_draft_en.md)
- [Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion](docs/paper_hotswap_en.md)

## 🤝 Contributing & License

This project is currently an experimental architecture in its early stages. Bug reports, discussions on features, and pull requests for performance improvements are very welcome via Issues and PRs!

- **License**: This repository is released under the [MIT License](./LICENSE). Please refer to the [`LICENSE`](./LICENSE) file for details.
