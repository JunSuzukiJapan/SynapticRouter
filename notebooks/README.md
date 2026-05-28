# Synaptic Routing Architecture (SRA) Interactive Demos

This directory contains Jupyter Notebooks designed to help you intuitively understand and easily experiment with the Synaptic Routing Architecture (SRA) right in your browser.

These notebooks are designed to be run in a few minutes using **Google Colab**, even if you don't have a local GPU environment.

## 📓 Notebook List

### 🟢 1. Basic Structure and Routing Validation
**File:** [`01_sra_quickstart_en.ipynb`](./01_sra_quickstart_en.ipynb)

Initializes the basic SRA model structure and visualizes the "routing" (which expert is selected) as a heatmap when random data is input. This is the simplest introductory demo for those who just want to see SRA in action.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_en.ipynb)

---

### 🔵 2. Single-Task Learning and Routing Specialization
**File:** [`02_learning_and_routing_demo_en.ipynb`](./02_learning_and_routing_demo_en.ipynb)

Actually trains SRA on a "Copy task" (outputting the input exactly as is). It draws routing heatmaps before and after a few seconds of training, allowing you to visually experience how the router **increasingly favors specific experts (synapses) intentionally as training progresses (specialization)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_en.ipynb)

---

### 🔴 3. Multitask Learning and Task-Specific Routing (✨ Recommended)
**File:** [`03_multitask_routing_demo_en.ipynb`](./03_multitask_routing_demo_en.ipynb)

Demonstrates SRA's greatest strength: **"Switching experts (synapses) in multitask learning."**
It trains a single model simultaneously on two conflicting tasks: `copy` and `reverse`. After training, it compares side-by-side how the router **selects completely different synaptic pathways** depending on the input task. Ideal for experiencing SRA's dynamic behavior.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_en.ipynb)

---

### 🕹️ 4. Decision Transformer: Separation of Perception and Action
**File:** [`04_decision_transformer_routing_demo_en.ipynb`](./04_decision_transformer_routing_demo_en.ipynb)

Uses SRA as a reinforcement learning agent (Decision Transformer) to solve "Treasure" and "Escape" tasks in a GridWorld.
In addition to the **"switching brains based on task (objective),"** you can verify via heatmaps how **"perception synapses" and "action synapses" are automatically separated and formed** depending on whether the input token is a "State," "Reward," or "Action."

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_en.ipynb)

---

### 🧠 5. [Must-See] Synapse Lesion Experiment
**File:** [`05_lesion_experiment_demo_en.ipynb`](./05_lesion_experiment_demo_en.ipynb)

A hacker-style experimental demo proving that SRA's network is "completely modularized by function."
From a multitask-trained model, we **intentionally destroy (set to zero) the weights of an expert (synapse) used in a specific task (Reverse)**. You can interactively experience its astonishing robustness: while the Reverse task becomes unsolvable, **the Copy task, which uses different synapses, maintains a 100% accuracy rate (remains completely unscathed)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_en.ipynb)


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

---

### 📖 8. SRA LLM Construction and Training (TinyShakespeare)
**File:** [`08_sra_llm_demo_shakespeare.ipynb`](./08_sra_llm_demo_shakespeare.ipynb)

Experience building and training a small-scale language model (Tiny LLM) from scratch using SRA. We use Shakespeare texts and visualize how SRA routes (specializes synapses) according to text patterns.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare.ipynb)

---

### 📚 9. SRA Multidomain LLM Training and Specialization
**File:** [`09_sra_llm_demo_multidomain.ipynb`](./09_sra_llm_demo_multidomain.ipynb)

Experience simultaneous learning of multiple domains (Code, Math, Text) on a small LLM. SRA automatically delegates tasks to different synapses per domain, demonstrating its powerful multitask efficiency.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain.ipynb)

---

### 💻 10. Practical Plugin Hot-Swap (Zero-Shot Hot-Swap)
**File:** [`10_hotswap_plugins_demo_en.ipynb`](./10_hotswap_plugins_demo_en.ipynb)

A practical demo where multiple teams independently train "Code" and "Math" plugins parallel to a "Text" base model. We then physically merge these tensor weights (Hot-Swap) into the base model, demonstrating mathematically ZERO catastrophic forgetting!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_en.ipynb)

---

### 🗑️ 11. Dynamic Synapse Deletion (pop & clear)
**File:** [`11_synapse_deletion_demo_en.ipynb`](./11_synapse_deletion_demo_en.ipynb)

Demonstrates two ways to delete synapses from a trained SRA model: physical removal via `pop_synapses()` (restores model size) and zero-clear purge via `clear_synapses()` (converts a slot into a free reusable slot). Also explores the "cosine similarity trap" that occurs after a zero-clear and how to mitigate it.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_en.ipynb)

---

### 🧬 12. Virtual Neuron Emergence Validation Experiment
**File:** [`12_virtual_neuron_experiment_en.ipynb`](./12_virtual_neuron_experiment_en.ipynb)

Trains an SRA model on **25 tasks across 5 domains × 5 tasks** (NL / Code / Math / DNA / CSV) at the character level. Without any task ID, the router autonomously identifies each task and forms distinct synapse groups (= virtual neurons / cell assemblies) per domain. Includes hierarchical extraction of Universal / Assembly / Peripheral synapses.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/12_virtual_neuron_experiment_en.ipynb)

---

### 🧠 13. Safe Hot-Swap (Unlearning) at the Virtual Neuron Granularity
**File:** [`13_virtual_neuron_hotswap_en.ipynb`](./13_virtual_neuron_hotswap_en.ipynb)

Building on Notebook 12's virtual neurons, demonstrates that operating at the Cell Assembly granularity allows surgical unlearning (deletion) of an entire functional group without affecting the others.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/13_virtual_neuron_hotswap_en.ipynb)

---

### 🔬 14. Knowledge Entanglement and Safe Deletion (Synapse vs. Neuron Unit)
**File:** [`14_compare_deletion_units_en.ipynb`](./14_compare_deletion_units_en.ipynb)

Compares deletion at two granularities — individual Synapse vs. Virtual Neuron — and shows why deleting only dedicated synapses leaves universal-core entanglement behind, motivating neuron-level operations.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/14_compare_deletion_units_en.ipynb)

---

### 📐 15. Capacity Hypothesis: Number of Synapses vs. Safe Unlearning Threshold
**File:** [`15_capacity_hypothesis_experiment_en.ipynb`](./15_capacity_hypothesis_experiment_en.ipynb)

Tests how the number of synapses (model capacity) affects whether dedicated synapses emerge for each task — establishing a capacity threshold above which clean virtual neurons reliably form.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment_en.ipynb)

---

### 💤 16. Lazy Routing Prevention Experiment
**File:** [`16_lazy_routing_prevention_experiment_en.ipynb`](./16_lazy_routing_prevention_experiment_en.ipynb)

Diagnoses and mitigates "router laziness," where a few synapses absorb most traffic and starve others. Explores load-balancing techniques that produce healthy, diverse routing distributions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/16_lazy_routing_prevention_experiment_en.ipynb)

---

### 🔁 17. Routing Fallback (Reassignment) Experiment
**File:** [`17_routing_fallback_experiment_en.ipynb`](./17_routing_fallback_experiment_en.ipynb)

Implements and verifies a fallback mechanism that reassigns traffic when a target synapse becomes unavailable (e.g., deleted or masked). Ensures graceful degradation rather than catastrophic failure.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/17_routing_fallback_experiment_en.ipynb)

---

### 🧩 18. Custom Synapses (Vector DB & Calculator)
**File:** [`18_custom_synapses_en.ipynb`](./18_custom_synapses_en.ipynb)

Demonstrates building **non-trainable, custom Synapses**: a Vector-DB-backed retrieval Synapse and a Python-`eval()` calculator Synapse. They plug into SRA alongside neural synapses through the same routing interface.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/18_custom_synapses_en.ipynb)

---

### 🎯 19. Zero-Shot Hard Routing (Forced Assignment via Metadata)
**File:** [`19_zero_shot_hard_routing_en.ipynb`](./19_zero_shot_hard_routing_en.ipynb)

Uses the router's `allowed_mask` to force 100% of traffic for a task to a specific Synapse — no training required. Useful for pinning a deterministic Synapse (Vector DB / calculator) into the routing path.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/19_zero_shot_hard_routing_en.ipynb)

---

### 🛠️ 20. Routing Fine-tuning (Autonomous Routing Learning)
**File:** [`20_routing_finetuning_en.ipynb`](./20_routing_finetuning_en.ipynb)

After adding a `VectorDBSynapse`, fine-tunes the router and encoder on a small dataset so the model autonomously learns to route appropriate queries to the new synapse — without forcing assignments.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/20_routing_finetuning_en.ipynb)

---

### 🧯 21. Catastrophic Forgetting Check after Routing Fine-tuning
**File:** [`21_finetuning_forgetting_check_en.ipynb`](./21_finetuning_forgetting_check_en.ipynb)

After fine-tuning routing for a new Synapse, verifies whether the accuracy on existing base tasks degrades. Quantifies catastrophic forgetting in the routing-fine-tuning regime.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/21_finetuning_forgetting_check_en.ipynb)

---

### 🧪 22. Coexistence of Neuro-Symbolic Heterogeneous Modules
**File:** [`22_multi_synapse_hotswap_eval_en.ipynb`](./22_multi_synapse_hotswap_eval_en.ipynb)

Validates SRA's true highlight: an LLM (learning-based), Vector DB (retrieval-based), and rule-based calculator coexist as Synapses on the same architecture. Adds `RealCalculatorSynapse` and proves arithmetic ability appears **without retraining**, while existing tasks remain intact.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/22_multi_synapse_hotswap_eval_en.ipynb)

---

### 🦙 23. SRA Native LLM Integration PoC (TinyLlama)
**File:** [`nb23_sra_llm_integration_en.ipynb`](./nb23_sra_llm_integration_en.ipynb)

Proof of concept that natively integrates the SRA router with an existing LLM (TinyLlama, a lightweight Llama variant) by matching dimensions. Validates that SRA can route over a pre-trained LLM's hidden space.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/nb23_sra_llm_integration_en.ipynb)

---

### 🏎️ 24. Router Architecture Benchmark
**File:** [`24_router_architecture_benchmark_en.ipynb`](./24_router_architecture_benchmark_en.ipynb)

Benchmarks router architectures (single-stage vs. multi-stage, last-token vs. all-token aggregation) on the LLM-integrated setup from Notebook 23, surfacing which design generalizes best.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/24_router_architecture_benchmark_en.ipynb)

---

### 🧰 25. Integrated Heterogeneous Routing (Motherboard Architecture)
**File:** [`25_integrated_heterogeneous_routing_en.ipynb`](./25_integrated_heterogeneous_routing_en.ipynb)

End-to-end integration of SRA's near-final form: the **"Motherboard architecture"** that routes Last-Token signals over heterogeneous Synapses (LLM / Vector DB / Calculator) with semantic fallback.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/25_integrated_heterogeneous_routing_en.ipynb)

---

### 💬 26. SRA Chatbot Demo
**File:** [`26_chatbot_demo_en.ipynb`](./26_chatbot_demo_en.ipynb)

A working chat UI built on the Motherboard architecture from Notebook 25 — combining LLM, Vector DB, and Calculator Synapses behind a single chat interface (ipywidgets).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/26_chatbot_demo_en.ipynb)

## 🚀 How to Run

1. **Running on Google Colab (Recommended)**:
   Click the `Open In Colab` badge below each item. An execution environment will launch in your browser, and you can experience the demo just by running the cells in order from top to bottom.

2. **Running on a Local Environment**:
   ```bash
   git clone https://github.com/JunSuzukiJapan/SynapticRouter.git
   cd SynapticRouter
   pip install -r requirements.txt
   jupyter lab
   ```
   Then, open and run the files under `notebooks/` from your browser.
