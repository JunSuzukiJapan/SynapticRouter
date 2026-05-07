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
