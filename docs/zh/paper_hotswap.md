# 热插拔大语言模型：基于突触路由架构的零样本模块组合与外科手术式知识遗忘

**Jun Suzuki**, Independent Researcher

## 摘要 (Abstract)
人类能够后天学习新技能（例如骑自行车或一门新语言），并使其作为大脑中独立的回路发挥作用，而不会破坏现有的知识。然而，当前的巨型语言模型（LLMs）拥有单体（Monolithic）结构，将所有知识密集地存储在单一的参数空间中，这使得这种后天知识的独立添加或特定记忆的删除变得极其困难，对操作的灵活性施加了严重的限制。

在本文中，我们利用突触路由架构（SRA）的功能定位机制，提出并验证了一种实现神经回路（模块）动态插入和拔出——即 **热插拔 (Hot-Swap)**——的方法。热插拔是一种双向的模块交换操作：在无需任何重新训练的情况下，将在基础模型之外独立训练的专用突触，**像外科手术般移植（Plug-In）到生产模型中**；并且当不再需要时，**可以安全地切断特定的记忆（Unplug）**。实验结果表明，通过受到向量数据库预过滤（Pre-filtering）技术启发的神经回路硬掩码机制，即使在移植之后，基础模型的输出也能达到精确到小数点的完全一致，实现了**零遗忘（Zero Forgetting）**。此外，在拔出操作中，我们发现并解决了余弦相似度特有的零向量“黑洞问题”，实现了超越生物神经可塑性的模块化AI的完整生命周期：“学习 → 移植 → 移除 → 再利用”。

## 1. 引言 (Introduction)

### 1.1 单体模型与后天记忆控制的局限性
自“Attention Is All You Need”提出以来，Transformer架构在自然语言处理中确立了统治地位。然而，拥有数百亿至数千亿参数的单体LLM，在后天知识的控制方面面临着以下严重问题：

1. **灾难性遗忘 (Catastrophic Forgetting)**: 当通用模型在特定领域（如公司规章、专业代码）上进行微调时，其原有的通用能力会被破坏或退化。
2. **学习成本膨胀**: 每次添加新知识都需要重新训练或合并整个模型，这使得跨多个团队的完全并行开发变得困难。
3. **知识删除的不可能性**: 在参数错综复杂的单体模型中，仅使模型遗忘特定知识的“机器遗忘（Machine Unlearning）”是一项艰巨的任务；如果尝试重新训练，往往会破坏无关的能力。

### 1.2 本文的贡献
我提出的突触路由架构（SRA）[Suzuki, 2026] 是一种通过端到端同时学习路径选择和模块表征，从而根据任务自发涌现“功能定位”的架构。本文聚焦于SRA的生物学模块化所带来的**操作上的革新**——即可以自由移植和移除神经回路的 **热插拔 (Hot-Swap)**——并报告了以下三点贡献：

1. **热插拔：Plug-In（外科手术式移植）**: 实现并验证了一种方法，只需将独立训练的专用模块的权重张量物理复制到基础模型的空槽中，即可完成部署。
2. **热插拔：Unplug（记忆的移除与灭活）**: 设计了两种删除API——物理切断（`pop_synapses`）和通过清零进行灭活清除（`clear_synapses`），并发现和解决了路由中的“零向量黑洞问题”。
3. **零遗忘的实验证明**: 证明了通过在特定任务中强制切断不需要的路径的硬掩码机制（Pre-filtering），基础模型在移植/移除前后的输出Loss在小数点后也能做到完全一致（数学上的零干扰）。

## 2. 背景：SRA 架构

SRA是一种模仿大脑空间隔离和动态路由的持续学习架构。我们概述了理解热插拔所需的组成部分（详细信息请参阅 [Suzuki, 2026]）。

### 2.1 路由器 (Dynamic Synaptic Formation)
作为SRA心脏的路由器，是一个决定“对于输入信息应该激发哪些神经回路”的单层线性层。它计算输入的隐藏状态 $h$ 与每个突触的特征向量（嵌入）$e_i$ 之间的余弦相似度，选择最匹配的 Top-k 个突触。

### 2.2 微型突触 (Functional Modules)
每个突触由独立的、极小的多头注意力机制（Multi-Head Attention）和多层感知机（MLP）组成。只有被路由器指示“激发”的突触才会执行计算，其他参数不会受到干扰。

### 2.3 共享躯干 (Shared Trunk) 与功能定位
热插拔能够成立的绝对关键条件是 **共享躯干方法 (Shared Trunk)**。所有的专用突触都共享同一个预训练的基础模型（相当于视觉皮层或基础语言区的Embedding/Attention层），并且仅独立地学习突触部分。这防止了模型之间内部表征的差异（Representation Divergence），使得回路的物理移植成为可能。

## 3. 热插拔：Plug-In (模块的外科手术式移植)

热插拔的第一项操作是将独立训练的专用模块（新技能）**移植（Plug-In）**到基础模型中。这种移植操作完全可以通过PyTorch的张量操作完成。

### 3.1 方法

```python
# hotswap_model: 生产环境中的基础模型（已添加大脑的空槽）
# plugin_math: 数学团队独立训练的专用LLM（新的数学回路）

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # 复制路由器的嵌入向量（激发条件）
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # 复制Expert (TinySynapse)的权重 (w1, w2)
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

只需将已训练的专用模型的张量直接赋值给基础模型的空槽。由于SRA在保持基础模型的共享知识完全冻结的状态下仅学习突触，因此这种物理复制是有效的。

### 3.2 实现独立的并行开发
这使得“代码团队”和“数学团队”可以基于同一个基础模型，互不干扰地独立训练各自的专用突触。完成后，只需将权重张量内存复制到生产环境的基础模型中，即可完成部署。

## 4. 零遗忘 (Zero Forgetting)：通过强制路径切断的硬掩码机制

### 4.1 挑战：路由器的混淆
如果只是简单地移植张量，路由器可能会混淆新旧突触，对基础模型的现有能力产生负面影响（遗忘）。

### 4.2 启发自向量数据库的预过滤 (Pre-filtering)
SRA路由器的本质类似于内存中的最大内积搜索（MIPS）引擎。我将向量数据库的**预过滤（Pre-filtering）**技术集成到路由器的前向传播中，设计了一种在推理时物理切断（硬掩码）“当前任务不需要的神经回路路径”的机制。

```python
# 路由器层的前向传播
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtering: 用 -∞ 强制切断未授权突触的路径
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-K 路由
vals, idx = torch.topk(logits, k, dim=-1)
```

通过这个 `-inf` 掩码，路由器只从授权的突触中选择回路。无论移植了多少其他模型的权重并共存，它们在计算图上都会被完全忽略，从而保证了**基础模型的 Loss 与移植前相比，在小数点后也能完全一致（数学上等于零的干扰）**。

## 5. 热插拔：Unplug (特定记忆的移除与灭活)

热插拔的第二项操作是从基础模型中**拔出（Unplug）**不再需要的模块。虽然单体模型中的机器遗忘（Machine Unlearning）很困难，但SRA通过“物理切断突触连接”的方法解决了这个问题。

### 5.1 方法①：物理切断 (`pop_synapses`)
如果后来添加的突触变得不再需要，可以通过从张量末尾进行切片来物理丢弃它们。

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

VRAM使用量将减少，模型将完全恢复到添加突触之前的状态。这实际上是一种物理切断特定神经回路的操作。

### 5.2 方法②：通过灭活进行清除 (`clear_synapses`)
如果想要删除中间的突触而不是末尾的，物理删除会导致索引偏移，从而使大脑的控制系统崩溃。因此，我们将突触的内容清零，使其“灭活”（变成空槽）。

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

通过仅使槽的内容无效而不改变张量大小，我们保持了完美的完整性。

## 6. 余弦相似度陷阱：零向量的黑洞问题

### 6.1 问题的发现与原因
在实现灭活清除时，我遇到了一个严重的问题，即**输出完全崩溃**。
SRA的路由器使用余弦相似度，但清零后的突触的嵌入向量即使经过归一化也仍然是 $\mathbf{0}$，导致余弦相似度为 $0.0$。
因为相似度的范围是 $[-1.0, 1.0]$，如果正常突触的分数为负数（例如：$-0.5$），**被灭活的空突触（$0.0$）将获得更高的分数，导致路由器优先选择空突触。**

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

这就像一个“黑洞”现象，数据被吸入了本应该被擦除的突触中。

### 6.2 解决方案：使用 -∞ 掩码进行完全封锁
为了解决这个问题，我在路由器的前向传播中添加了掩码处理，以检测清零的突触并完全封锁其路径。

```python
# 检测被清零（灭活）的突触
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

使用 $-\infty$ 掩码，在数学上就不可能选择被灭活的突触，从而实现了空槽的安全再利用。

## 7. 模块化AI的完整生命周期

SRA实现了以下模块化AI的完整生命周期：

```
学习 (Train) → 移植 (Plug-In) → 运行 (Serve)
       ↓                                ↓
 独立的并行开发                     移除 (Unplug)
                                        ↓
                                空槽再利用 (Reuse)
                                        ↓
                                新的移植 (Plug-In)
```

这使得模型的无限扩展和持续维护成为可能，就像生物的大脑终其一生都在不断学习新技能一样。

## 8. 讨论 (Discussion)

### 8.1 表征的差异 (Representation Divergence)
热插拔能够成立的绝对条件是所有的专用突触都派生自同一个预训练的基础模型。在完全独立训练的模型之间，由于内部表征的差异，路由会崩溃。

### 8.2 替代方案：超级路由器 (Super Router)
为了放宽共享躯干的限制，我们正在验证一种方法，即使用Gumbel-Softmax通过更高级别的超级路由器来封装和捆绑整个独立模型。

### 8.3 安全风险
由于热插拔具有从运行系统外部动态加载权重文件的特性，它引入了诸如 (1) Pickle利用、(2) 恶意权重注入（后门注入）和 (3) 路由劫持等风险。必须强制使用 `safetensors` 格式并通过密码学签名来验证突触。

## 9. 结论 (Conclusion)

在本文中，我们利用SRA的功能定位机制，建立了一种能够动态移植和移除特定神经回路的“热插拔LLM”方法。我们仅通过物理张量复制成功移植了新技能（Plug-In），并通过物理切断和清零灭活成功地移除了特定的记忆（Unplug）。

通过硬掩码机制从数学上保证了零遗忘，并解决了余弦相似度中的“黑洞问题”，这种方法代表了使AI模型从“单体黑盒”进化为“可以通过物理插拔部分智能来进行安全控制的生命级模块化系统”的一个极具前景的方向。

## 参考文献 (References)

- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.
