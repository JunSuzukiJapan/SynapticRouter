# 受神经启发的突触路由：通过动态模块化路径克服灾难性遗忘

**Jun Suzuki**, Independent Researcher

## 摘要 (Abstract)
人类的大脑能够后天学习并执行截然不同的任务（例如走路、说话和计算）而不会相互干扰。这是因为大脑的神经回路（突触）会根据任务进行动态路由，并保持空间上隔离的“功能定位（Functional Localization）”。相比之下，当人工神经网络（ANNs）在单一的巨型网络中学习多个任务时，它们会遭受“灾难性遗忘（Catastrophic Forgetting）”，即破坏过去的记忆。

在本文中，我们受到这种突触动态形成和空间隔离的生物学机制的启发，提出了一种持续学习模型——“突触路由架构（Synaptic Routing Architecture, SRA）”。SRA由一个极其简单的单层“路由器（Router）”和多个独立的微型模块（突触）组成。通过我们的实验证明，即使在推理时没有从外部获得任务ID，SRA也能从输入中自主识别任务的性质，并**以端到端（End-to-End）的方式完全同时学习路径选择（路由）和任务的表征**。我们展示了在无需人为冻结权重或使用复杂进化算法的情况下，模型内部自发涌现出了自主的功能定位，从而完全避免了灾难性遗忘。

## 1. 引言 (Introduction)
在深度学习领域，模型不断获取新知识的“持续学习（Continual Learning）”是实现通用人工智能（AGI）的最大障碍之一。当前的巨型Transformer等“单体（Monolithic）”网络在利用新领域数据进行微调时，不可避免地会遗忘以前学到的知识。

为了解决这个问题，本研究着眼于大脑的“功能定位”。正如大脑的语言区和运动区使用物理上不同的回路来防止干扰一样，SRA被设计为一种架构，它可以通过动态路由机制，动态地开启/关闭（插入/拔出）独立的微型网络（突触）。

## 2. 相关工作与SRA的创新性
防止灾难性遗忘的现有方法包括诸如 EWC (Elastic Weight Consolidation) 等正则化方法，它们对以前任务中重要的权重更新进行惩罚。然而，这些方法受限于模型的容量（Capacity），随着任务数量的增加，最终会达到极限。

与SRA更接近的“结构化和模块化”方法是Google DeepMind提出的 **PathNet (2017)**。PathNet提供了大量模块，并使用遗传算法来寻找每个任务的“路径（Paths）”，在学习后冻结权重以防止遗忘。

### SRA的压倒性优势（同时学习的力量）
与PathNet等传统方法相比，SRA的根本创新在于其能够**“以可微的方式同时（端到端地）进行路径探索（路由）和模块学习”**。

1. **自主路由（Task-Agnostic）：** PathNet在推理时需要从外部被告知“正在执行哪个任务（任务ID）”。相比之下，SRA的单层线性路由器能够根据输入特征的余弦相似度，在推理时**自主判断**任务领域（例如，“这是一个数学任务”或“这是一个语言任务”），并将其路由到适当的突触。
2. **无需权重冻结和进化算法：** SRA不需要像遗传算法那样庞大的计算成本，路由器和突触仅通过标准的反向传播（Backpropagation）就能协同学习。
3. **动态功能定位的涌现：** 因为路由器自发地学习了“相似的任务使用相同的突触”和“不同的任务使用不同的突触”的路径，即使不人为冻结权重，通过稀疏激活，空间上的隔离（功能定位）也会自然涌现。

## 3. 架构 (Neuro-inspired Design)
SRA是一种动态且稀疏的架构，模仿了生物大脑的突触形成。

### 3.1 路由器 (Dynamic Synaptic Formation)
作为SRA的心脏，路由器根据输入的信息决定“应该激发哪些神经回路”。路由器是一个简单的线性层（Linear Layer），它计算输入特征与每个突触的“嵌入向量（Embedding Vector）”之间的余弦相似度，从而决定最匹配（激发）的Top-k个突触。

### 3.2 微型突触 (Functional Modules)
每个突触由独立的、极小的多头注意力机制（Multi-Head Attention）和多层感知机（MLP）组成。只有被路由器指示“激发”的突触才会执行计算，其他突触的参数不会受到干扰。这赋予了模型类似于大脑空间隔离的抗遗忘能力。

### 3.3 架构图
下图展示了输入由路由器评估，并激发最佳突触（神经回路）的流程。

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

## 4. 实验1：算法推理 (Algorithmic Reasoning)
为了验证同时学习路由器和模块是否能识别任务性质，我们在没有提供任何任务ID的情况下，让模型同时学习四个完全不同的算法推理任务（`copy`, `reverse`, `paren`, `addmod`）。

### 结果与功能定位的涌现
经过10,000步的联合学习，所有任务都达到了100%的准确率。此外，对路由器路径分布的分析揭示了一个惊人的结果。

**路由器对任务的聚类：**
- **序列操作区**: `COPY` 和 `REVERSE` （以0.969的相似度共享同一个突触）
- **计算/逻辑区**: `PAREN` 和 `ADDMOD` （以0.858的相似度共享同一个突触）
- 上述两组之间的相似度极低（0.029 - 0.336），显示出明确的功能分离。

在人类没有提供任何指令（任务ID）的情况下，路由器的同时学习自动将“改变顺序的任务”和“需要逻辑的任务”聚集在一起，**促使了类似于大脑功能定位的模块化隔离的涌现。**

## 5. 实验2：跨领域语言建模 (Cross-Domain Language Modeling)
接下来，我们进行了难度更高的“跨领域语言建模”。我们同时让模型学习了三种语法和词汇完全不同的领域：`Code`（Python）、`Math`（LaTeX）和 `Text`（自然语言）。

### 结果
尽管只进行了1,000步的学习，路由器已经完成了“每个领域的专门回路”的形成：
- `Code` 区: **突触 8** 占主导地位
- `Math` 区: 由 **突触 10 和 13** 负责
- `Text` 区: 由 **突触 0 和 15** 负责

在单体模型会发生灾难性遗忘的情况下，SRA通过路由器为每个领域分配了专门的突触（独立的参数空间），成功地将相互干扰降至最低。

## 6. 实验3：多语言机器翻译 (Multilingual Machine Translation)
我们使用三种句法结构不同的语言（英语：SVO，法语：SVO，日语：SOV）进行了多语言机器翻译实验。

### 结果与讨论
对突触使用率的分析表明，模型自发形成了一个“SVO共享突触”（在英法之间的翻译中频繁激活）和一个“SOV专用突触”（仅在翻译成日语时使用率飙升）。这表明路由器不仅使用了语言标签，而且在底层掌握了本质的“语序和句法规则”，并据此动态切换神经回路。

## 7. 实验4：决策Transformer (Decision Transformer / Offline RL)
最后，为了证明其在自然语言以外领域的适应性，我们将SRA作为决策Transformer，对离线强化学习（RL）轨迹数据进行了验证。我们让模型同时学习了两个规则完全不同的环境（“寻宝 (Treasure)”任务和“逃脱 (Escape)”任务）的游玩日志。

### 结果与讨论
对每个Token的路由进行可视化后，证实了一个惊人的现象：**“感知（Perception）”和“策略（Policy）”功能的完全分离**。
- **感知Token（状态理解）**：对于表示坐标等状态的Token，无论属于哪种任务，路由器**无一例外地将它们路由到了一个公共突触（Expert 1）**。这意味着形成了一个共享的“空间感知区”。
- **动作Token（策略决定）**：另一方面，对于生成下一个动作的Token，寻宝任务和逃脱任务的路径清晰地分叉到了不同的突触。

“用同样的眼睛感知环境，但用不同的大脑做出决定”这一理想的模块化结构，是通过路由的同时学习而在没有人类设计的情况下获得的。

## 8. 结论 (Conclusion)
在本文中，通过突触路由架构（SRA），我们展示了从在所有任务中共享所有参数的“传统静态神经网络”，向“具备生物学空间隔离和动态路由的模块化网络”进行范式转变的可能性。

最大的突破在于，**通过端到端地同时学习路由器的路径选择和模块的表征学习，实现了与任务ID无关（Task-Agnostic）的持续学习。** 无需像PathNet那样使用复杂的进化算法，仅仅优化一个简单的路由器，模型内部就自发涌现了“功能定位”。
SRA克服了灾难性遗忘，并允许插入无限数量的新任务（突触），是迈向可扩展通用人工智能（AGI）的关键一步。

## 参考文献 (References)

- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.
