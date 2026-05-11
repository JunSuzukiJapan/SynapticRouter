# Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways

**Jun Suzuki**, Independent Researcher

## Abstract
The human brain can learn and execute fundamentally different tasks—such as walking, speaking, and calculating—without mutual interference. This is because the brain's neural circuits (synapses) are dynamically routed according to the task, maintaining spatially isolated "functional localization." In contrast, when Artificial Neural Networks (ANNs) learn multiple tasks within a single monolithic network, they suffer from "Catastrophic Forgetting," where past memories are destroyed.

In this paper, we propose the "Synaptic Routing Architecture (SRA)," a continual learning model inspired by the biological mechanisms of dynamic synapse formation and spatial isolation. SRA consists of an extremely simple, single-layer "Router" and multiple independent, tiny modules (Synapses). Through our experiments, we demonstrate that SRA can autonomously identify the nature of a task from the input—without being provided an explicit task ID during inference—and **learn both routing (pathway selection) and task representations entirely end-to-end simultaneously.** We show that, without artificial weight freezing or complex evolutionary algorithms, an autonomous functional localization emerges within the model, completely avoiding catastrophic forgetting.

## 1. Introduction
In the field of deep learning, "Continual Learning"—where a model continuously acquires new knowledge—is one of the greatest barriers to realizing Artificial General Intelligence (AGI). Monolithic networks, such as current massive Transformer models, inevitably forget previously learned knowledge when fine-tuned on data from new domains.

To address this problem, our research focuses on the brain's "Functional Localization." Just as the brain's language and motor areas use physically distinct circuits to prevent interference, SRA is designed as an architecture that can dynamically turn on/off (plug-in/unplug) tiny independent networks (synapses) through a dynamic routing mechanism.

## 2. Related Work & Novelty of SRA
Existing approaches to prevent catastrophic forgetting include regularization methods like EWC (Elastic Weight Consolidation), which penalize updates to weights that were important for previous tasks. However, these methods are limited by the model's capacity and eventually reach their limits as the number of tasks increases.

A more "structural and modular" approach akin to SRA is **PathNet (2017)** by Google DeepMind. PathNet provides numerous modules and uses genetic algorithms to discover "paths" for each task, freezing the weights after learning to prevent forgetting.

### The Overwhelming Advantage of SRA (Simultaneous Learning)
Compared to conventional approaches like PathNet, the fundamental novelty of SRA lies in its ability to **"simultaneously learn pathway discovery (routing) and module representations in a differentiable, end-to-end manner."**

1. **Autonomous Routing (Task-Agnostic):** PathNet requires the model to be explicitly told "which task it is performing (Task ID)" during inference. In contrast, SRA's single linear router autonomously determines the domain (e.g., "this is a math task," "this is a language task") based on the cosine similarity of input features and routes it to the appropriate synapse.
2. **Elimination of Weight Freezing and Evolutionary Algorithms:** Instead of requiring massive computational costs like genetic algorithms, SRA allows the router and synapses to learn cooperatively using only standard Backpropagation.
3. **Emergence of Dynamic Functional Localization:** Because the router spontaneously learns pathways such that "similar tasks use the same synapse" and "different tasks use different synapses," spatial isolation (functional localization) naturally emerges through sparse activation without the need for artificial weight freezing.

## 3. Architecture (Neuro-inspired Design)
SRA is a dynamic and sparse architecture that mimics the synapse formation of the biological brain.

### 3.1 The Router (Dynamic Synaptic Formation)
The router, the heart of SRA, determines "which neural circuits to fire" based on the input information. The router is a simple linear layer that calculates the cosine similarity between the input features and the "embedding vector" of each synapse, determining the Top-k synapses that best match (fire).

### 3.2 Tiny Synapses (Functional Modules)
Each synapse consists of an independent, extremely small Multi-Head Attention and MLP. Only the synapses instructed to "fire" by the router execute computations; the parameters of other synapses remain un-interfered. This provides forgetting resistance similar to the brain's spatial isolation.

### 3.3 Architecture Diagram
The following diagram illustrates the flow where the input is evaluated by the router, and the optimal synapses (neural circuits) fire.

```mermaid
graph TD
    X[Input Token] --> Base[Residual Base]
    X --> Norm[LayerNorm]
    
    Norm --> Router["Router (Synaptic Routing)"]
    Norm --> SynapseSpace
    
    subgraph Synapse Space (Functional Modules)
        SynapseSpace((Select Top-k))
        S1["Synapse 0<br/>(Mini-Transformer)"]
        S2["Synapse 1<br/>(Mini-Transformer)"]
        S3["Synapse ..."]
        Sn["Synapse 15<br/>(Mini-Transformer)"]
    end
    
    Router -- "Routing Weights" --> SynapseSpace
    SynapseSpace --> S1
    SynapseSpace --> S2
    SynapseSpace -.-> Sn
    
    S1 --> Combine((Weighted Sum))
    S2 --> Combine
    Sn -.-> Combine
    
    Base --> Combine
    Combine --> Out[Output Representation]
```

## 4. Experiment 1: Algorithmic Reasoning
To verify whether simultaneously learning the router and modules can discern task properties, we trained the model concurrently on four entirely different algorithmic reasoning tasks (`copy`, `reverse`, `paren`, `addmod`) without providing any task IDs.

### Results and Emergence of Functional Localization
After 10,000 steps of simultaneous learning, the model achieved 100% Accuracy on all tasks. Furthermore, analyzing the router's pathway distribution revealed an astonishing result.

**Task Clustering by the Router:**
- **Sequence Manipulation Area**: `COPY` and `REVERSE` (Shared the same synapse with a similarity of 0.969)
- **Calculation/Logic Area**: `PAREN` and `ADDMOD` (Shared the same synapse with a similarity of 0.858)
- The similarity between the above two groups was extremely low (0.029 - 0.336), showing clear functional separation.

Without humans providing any instructions (task IDs), the simultaneous learning of the router autonomously clustered "sequence reordering tasks" and "logic-requiring tasks," **resulting in the emergence of modular segregation akin to the brain's functional localization.**

## 5. Experiment 2: Cross-Domain Language Modeling
Next, we conducted a more challenging "cross-domain language modeling." We simultaneously trained the model on three domains with entirely different grammars and vocabularies: `Code` (Python), `Math` (LaTeX), and `Text` (Natural Language).

### Results
Despite only 1,000 steps of training, the router completed the formation of "specialized circuits per domain":
- `Code` Area: Dominated by **Synapse 8**
- `Math` Area: Handled by **Synapses 10 and 13**
- `Text` Area: Handled by **Synapses 0 and 15**

In a situation where a monolithic model would experience catastrophic forgetting, SRA successfully minimized mutual interference by assigning specialized synapses (independent parameter spaces) to each domain through the router.

## 6. Experiment 3: Multilingual Machine Translation
We performed multilingual machine translation using three languages with different syntactic structures (English: SVO, French: SVO, Japanese: SOV).

### Results and Discussion
Analyzing the synapse utilization rate revealed the autonomous formation of an "SVO-shared synapse" that activated frequently during translation between English and French, and an "SOV-specialized synapse" whose usage spiked only during translation into Japanese (SOV). This indicates that the router did not just use language labels, but acquired the essential underlying "word order and syntactic rules," dynamically switching neural circuits based on them.

## 7. Experiment 4: Decision Transformer (Offline RL)
Finally, to demonstrate its adaptability to non-natural language domains, we validated SRA as a Decision Transformer learning offline reinforcement learning (RL) trajectory data. We simultaneously trained the model on play logs from two environments with completely different rules (the "Treasure" task and the "Escape" task).

### Results and Discussion
Visualizing the routing for each token confirmed an astonishing phenomenon: **complete functional separation of "Perception" and "Policy".**
- **Perception Tokens (State Comprehension)**: For state tokens representing coordinates, the router routed them to a **common synapse (Expert 1) without exception**, regardless of the task type. This means a shared "spatial perception area" was formed.
- **Action Tokens (Policy Decision)**: On the other hand, for tokens generating the next action, the pathways clearly branched into different synapses for the Treasure task and the Escape task.

The ideal modular structure of "perceiving the environment with the same eyes, but making decisions with different brains" was acquired without human design through simultaneous routing learning.

## 8. Conclusion
In this paper, through the Synaptic Routing Architecture (SRA), we demonstrated the potential for a paradigm shift from "traditional static neural networks" that share all parameters across all tasks, to a "modular network equipped with biological spatial isolation and dynamic routing."

The greatest breakthrough is that **by performing end-to-end simultaneous learning of the router's pathway selection and the modules' representation learning, task-agnostic continual learning has become possible.** Without needing complex evolutionary algorithms like PathNet, simply optimizing a straightforward router resulted in the autonomous emergence of "functional localization" within the model.
SRA overcomes catastrophic forgetting and allows an infinite number of new tasks (synapses) to be plugged in, representing a crucial step toward scalable Artificial General Intelligence (AGI).

## References

- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.
