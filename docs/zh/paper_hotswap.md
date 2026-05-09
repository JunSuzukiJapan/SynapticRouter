# Hotswappable LLM：通过突触路由架构实现零样本模块组合与外科式知识删除

**Jun Suzuki**，独立研究者

## Abstract
大型语言模型（LLM）将所有知识密集存储在单一参数空间的单体结构中，导致特定知识的添加和删除极为困难，对运维灵活性造成严重制约。本文利用Synaptic Routing Architecture（SRA）的模块化特性，提出并验证了模块的动态插拔——即 **Hot-Swap** 技术。Hot-Swap是一种双向模块交换操作：对预训练的基础模型，可以将独立训练的专用突触 **无需任何重新训练即可物理插入（Plug-In）**，当不再需要时可以 **外科式地拔除（Unplug）**。实验结果表明，借鉴向量数据库预过滤技术的硬掩码机制，在插入后基础模型的输出精确到小数点以下完全一致，实现了 **Zero Forgetting**。此外，在拔除操作中发现并解决了余弦相似度中零向量的"黑洞问题"，实现了"训练→插入→拔除→复用"这一模块化AI的完整生命周期。

## 1. Introduction

### 1.1 单体模型的运维局限
自"Attention Is All You Need"以来，Transformer架构在包括自然语言处理在内的众多领域确立了主导地位。然而，拥有数百亿至数千亿参数的单体LLM面临以下严峻的运维问题。

1. **灾难性遗忘（Catastrophic Forgetting）**：对通用模型进行特定领域（例如：公司规章、专用代码）的追加训练时，原有的通用能力会被破坏或退化。
2. **训练成本膨胀**：每次添加新知识都需要对整个模型（或LoRA等适配器）进行重新训练和合并，使得多团队完全并行开发变得困难。
3. **知识删除的不可能性**：让模型选择性遗忘特定知识的"Machine Unlearning"，在参数复杂交织的单体模型中极其困难，且尝试重新训练会同时破坏无关能力。

### 1.2 本文贡献
笔者提出的Synaptic Routing Architecture（SRA）[Suzuki, 2026]是一种由极小独立模块（突触）和轻量路由器构成的稀疏架构。先前研究已实证了路由器的自主任务分离能力，本文聚焦于SRA模块化带来的 **运维革新**——自由执行模块插入和拔除的 **Hot-Swap**，报告以下3项贡献。

1.**Hot-Swap: Plug-In（插入）**：将独立训练的专用模型的权重张量物理复制到基础模型的空闲槽位即可完成部署的方法的实现与验证。
2.**Hot-Swap: Unplug（拔除）**：物理分离（`pop_synapses`）和零清除清洗（`clear_synapses`）两种删除API的设计，以及余弦相似度中"零向量黑洞问题"的发现与解决。
3.**Zero Forgetting的实验证明**：借鉴向量数据库预过滤（Pre-filtering）的硬掩码机制，证实插入和拔除前后基础模型的输出Loss精确到小数点以下完全一致。

## 2. Background: SRA Architecture

SRA（Synaptic Routing Architecture）是一种模仿生物大脑的动态稀疏架构。本节概述理解Hot-Swap所需的组成要素（详细参见[Suzuki, 2026]）。

### 2.1 Router
SRA的核心——路由器，是一个不含Attention机制的 **单一线性层（Linear Layer）**。它计算输入隐藏状态 $h$ 与各突触特征向量（嵌入）$e_i$ 的余弦相似度，选择Top-k个突触。

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

其中 $\alpha$ 为缩放系数。

### 2.2 Tiny Synapses
每个突触是由小型Multi-Head Attention和MLP组成的独立微型模块。仅路由器选中的突触执行计算，实现高计算效率。

### 2.3 共享主干（Shared Trunk）
Hot-Swap成立的绝对条件中极为重要的是 **共享主干方式**。所有专用突触均派生自同一预训练基础模型（Embedding层、Attention层），仅独立训练突触部分。这可以防止模型间内部向量表示的分歧（Representation Divergence），使物理复制进行突触移植成为可能。

## 3. Hot-Swap: Plug-In（模块插入）

Hot-Swap的第一个操作是将独立训练的专用模块 **插入（Plug-In）** 基础模型。本章展示该插入操作完全由PyTorch张量操作完成，是一个极为简单的过程。

### 3.1 方法

```python
# hotswap_model: 生产环境的基础模型（已添加空闲槽位）
# plugin_math: 数学团队独立训练的专用LLM

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # 复制路由器的嵌入向量
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # 复制Expert (TinySynapse)的权重 (w1, w2)
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

只需将训练好的专用模型的张量直接赋值到基础模型张量的特定索引（空闲槽位）。由于SRA通过共享主干方式在完全冻结基础模型共有知识（Attention层等）的状态下仅训练突触，因此这种物理复制是有效的。

### 3.2 实现独立并行开发
通过此方式，"代码团队"和"数学团队"可以基于同一基础模型，互不干扰地各自独立训练专用突触。训练完成后，只需将权重张量内存复制到生产环境基础模型的空闲槽位即可完成部署。

## 4. Zero Forgetting：借鉴向量数据库Pre-filtering的硬掩码机制

### 4.1 挑战：路由器混淆
简单地进行张量物理复制，存在路由器混淆新旧突触、导致基础模型输出发生变化的风险。

### 4.2 向量数据库中的元数据过滤
Pinecone、Weaviate等现代向量数据库中，除了基于余弦相似度的语义搜索外，元数据过滤也发挥着重要作用。

-**Post-filtering（后过滤）**：Top-K搜索后排除不符合条件的结果。容易产生K件不足的"K-NN枯竭问题"。
-**Pre-filtering（预过滤）**：在搜索 **之前** 通过元数据掩码限制搜索空间，仅在符合条件的候选中执行Top-K搜索。噪声被完全排除。

### 4.3 SRA的预执行硬掩码
SRA路由器的本质是计算输入向量与各突触嵌入向量内积的 **内存向量搜索引擎（Maximum Inner Product Search: MIPS）**。

笔者将向量数据库的Pre-filtering直接内嵌于路由器的前向传播中。在推理时向模型传递表示"当前任务允许的突触集合"的元数据掩码。

```python
# Router层的前向传播
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtering: 将未授权突触的logits设为-∞
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-K路由
vals, idx = torch.topk(logits, k, dim=-1)
```

通过此`masked_fill`实现的Pre-filtering，路由器仅从被允许的突触中选择专家。无论多少其他模型的权重共存，在计算图上都会被完全忽略，因此 **基础模型的Loss在合并前后精确到小数点以下完全一致（数学上零干扰）**。

### 4.4 实验结果
对基础模型（Code/Math/Text 3领域语言模型）在Hot-Swap独立训练的专用突触前后，比较了基础模型的Validation Loss。结果显示，合并前后Loss精确到小数点以下完全一致，Zero Forgetting得到实证。

## 5. Hot-Swap: Unplug（模块拔除）

Hot-Swap的第二个操作是将不再需要的模块从基础模型中 **拔除（Unplug）**。既然能"插入"知识，那么"拔除"不需要的知识也是必需的。单体模型中的Machine Unlearning由于参数复杂交织而极其困难，但SRA的模块化结构通过物理操作解决了这一问题。

### 5.2 方法①：物理分离（pop_synapses）
当通过Hot-Swap后添加的突触不再需要时，从张量末尾进行切片，物理性地截除。

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**优点**：VRAM使用量物理性缩减，可将模型完全恢复到添加前的状态。如同卸载操作系统驱动程序一样，可以物理性地拆除AI大脑的部件。

### 5.3 方法②：零清除清洗（clear_synapses）
需要删除非末尾而是中间索引的突触时，物理删除会导致索引偏移，使元数据掩码的控制系统崩溃。因此，将突触内容清零以"空闲槽位化"。

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

不改变张量大小而仅将槽位内容无效化，因此索引的一致性得到完美保持。空闲槽位后续可通过覆写新突触（Hot-Swap）进行复用。

## 6. The Cosine Similarity Trap：零向量的黑洞问题

### 6.1 问题的发现
在实现零清除清洗时，遭遇了 **输出完全崩溃** 的严重bug。

### 6.2 原因分析
SRA的路由器使用余弦相似度进行路由。被零清除的突触嵌入向量变为 $\mathbf{0}$，即使归一化后仍为 $\mathbf{0}$。任意输入向量 $h$ 与零向量的余弦相似度为 $0.0$。

问题在此产生。余弦相似度的值域为 $[-1.0, 1.0]$。若正常突触的余弦相似度为负值（例如：$-0.5$），**空突触（$0.0$）的分数在数学上更高，路由器会优先选择空突触**。

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

"本应消除存在的空闲槽位却将数据吸入并使其消失"——一种如同黑洞般的行为。

### 6.3 解决方案：通过-∞掩码完全封锁
在路由器的前向传播中添加了检测和排除零清除突触的掩码处理。

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# 检测零清除的突触
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

通过 $-\infty$ 掩码，无论其他突触的分数多低，空突触被选中在数学上都不可能。

## 7. The Complete Lifecycle of Modular AI

通过上述机制，SRA实现了以下模块化AI的完整生命周期。

```
训练（Train） → 合并（Hot-Swap） → 运行（Serve）
       ↓                                    ↓
  独立并行开发                         删除（Purge）
                                            ↓
                                    槽位复用（Reuse）
                                            ↓
                                    新合并（Hot-Swap）
```

1. **训练**：多个团队共享基础模型，各自独立并行开发专用突触。
2. **合并**：将训练好的张量物理复制到生产环境的基础模型中进行部署。
3. **运行**：通过硬掩码Pre-filtering保证Zero Forgetting的同时进行推理。
4. **删除**：将不需要的突触物理分离或零清除进行清洗。
5. **复用**：在空闲槽位通过Hot-Swap插入新的专用突触进行复用。

## 8. Discussion

### 8.1 表示分歧（Representation Divergence）
Hot-Swap成立的 **绝对条件** 是所有专用突触都派生自同一预训练基础模型（共享主干）。在完全独立训练的模型之间移植突触，会因内部向量表示的分歧导致路由崩溃。

### 8.2 上层路由器（Super Router）替代方案
为放宽共享主干的约束，已验证了将独立的完整模型封装起来、通过Gumbel-Softmax上层路由器（Super Router）统合的方案。该方式实现了完美的 $1.0$ vs $0.0$ Hard Routing，即使架构不同的模型之间也可以完全动态切换计算资源。

### 8.3 安全风险
Hot-Swap功能由于具有从运行系统外部动态加载权重文件的特性，会产生新的安全威胁。主要风险包括：(1) 通过Pickle Exploit的任意代码执行，(2) 恶意权重注入（Backdoor Injection），(3) 通过路由器密钥伪造的路由劫持，(4) 通过交换抖动的DoS攻击。建议通过强制使用`safetensors`格式、加密签名的突触验证、速率限制的引入等进行防御。

### 8.4 当前限制与未来展望
本研究处于`d_model=128`、`n_layers=4`左右的小规模模型的实验阶段，在10B级别的大型LLM上的验证是未来的重要课题。此外，关于路由器同步问题——在添加具有全新能力的突触时可能需要路由器密钥的适应性学习——也需要进一步研究。

## 9. Conclusion

本文利用SRA（Synaptic Routing Architecture）的模块化特性，提出并验证了使LLM成为"Hotswappable（可动态插拔）"的方法。Hot-Swap的插入（Plug-In）操作仅通过张量的物理复制即可完成部署，拔除（Unplug）操作则确立了物理分离和零清除清洗两种方法。通过借鉴向量数据库Pre-filtering技术的硬掩码机制，数学上保证了Zero Forgetting，通过发现和解决拔除时余弦相似度中零向量的"黑洞问题"，实现了槽位的安全复用。

在模型持续巨型化和黑箱化的当代，能够物理性地插拔智能组件进行控制的"Hotswappable LLM"方案，从模型的可维护性、安全性和运维效率的角度来看，是一个极具前景的方向。

## References

- Suzuki, J. (2026).[All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

本文介绍的Hot-Swap和突触删除的完整流程可在以下Google Colab笔记本中实际运行体验。

- **Hot-Swap突触合并演示（基础训练～独立训练～合并～Zero Forgetting证明）**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **突触删除演示（物理分离～零清除～黑洞问题的解决）**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[SRA的未来：突触的动态热交换与扩展性](./sra_future_hotswap_ja.md)** — 关于卡带式突触运维、个性化、分布式学习的考察。
- **[SRA热交换功能的安全风险与攻击方法](./sra_security_risks_hotswap_ja.md)** — Pickle Exploit、Backdoor Injection、DoS攻击等威胁向量与防御策略。
- **[表示分歧与分层路由](./sra_representation_divergence_ja.md)** — Shared Trunk方式与Super Router解决方案。
- **[SRA分层路由器中Hard Routing方法的比较验证](./sra_hierarchical_hard_routing_ja.md)** — Soft / STE / Gumbel-Softmax的比较实验。
