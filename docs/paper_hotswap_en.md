# Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion via Synaptic Routing Architecture

**Jun Suzuki**, Independent Researcher

## Abstract
Large Language Models (LLMs) store all knowledge densely within a single monolithic parameter space, making it extremely difficult to add or remove specific knowledge and imposing severe constraints on operational flexibility. In this paper, I leverage the modularity of the Synaptic Routing Architecture (SRA) to propose and validate **Hot-Swap** — a bidirectional module exchange operation for neural networks. Hot-Swap enables **Plug-In**: physically inserting independently trained specialized synapses into a pre-trained base model **without any retraining**, and **Unplug**: surgically removing knowledge that is no longer needed. Experiments demonstrate that a hard-masking mechanism inspired by vector database pre-filtering techniques achieves **Zero Forgetting**, where the base model's output loss matches exactly to the decimal point before and after insertion. Additionally, I discover and resolve the "Black Hole Problem" of zero vectors in cosine similarity encountered during the Unplug operation, establishing the complete lifecycle of modular AI: Train → Plug-In → Unplug → Reuse.

## 1. Introduction

### 1.1 Operational Limitations of Monolithic Models
Since the introduction of "Attention Is All You Need," the Transformer architecture has established a dominant position across many domains including natural language processing. However, monolithic LLMs with hundreds of billions of parameters face the following critical operational challenges:

1. **Catastrophic Forgetting**: Fine-tuning a general-purpose model on a specific domain (e.g., internal regulations, specialized code) destroys or degrades its original general capabilities.
2. **Escalating Training Costs**: Each knowledge addition requires retraining the entire model (or adapters such as LoRA), making fully parallel development by multiple teams impractical.
3. **Impossibility of Knowledge Deletion**: "Machine Unlearning" — selectively forgetting specific knowledge — is extremely difficult in monolithic models where parameters are deeply intertwined, and retraining attempts often destroy unrelated capabilities.

### 1.2 Contributions
I previously proposed the Synaptic Routing Architecture (SRA) [Suzuki, 2026], a sparse architecture composed of tiny independent modules (synapses) and a lightweight router. While prior work demonstrated the router's autonomous task-separation capability, this paper focuses on the **operational innovations** enabled by SRA's modularity — the bidirectional **Hot-Swap** of modules (insertion and removal) — reporting three contributions:

1. **Hot-Swap: Plug-In (Insertion)**: Implementation and validation of a method where deploying independently trained specialized models requires only physical tensor copy into the base model's empty slots.
2. **Hot-Swap: Unplug (Removal)**: Design of two removal APIs — physical detachment (`pop_synapses`) and zero-clear purging (`clear_synapses`) — and the discovery and resolution of the "Black Hole Problem" of zero vectors in cosine similarity.
3. **Experimental Proof of Zero Forgetting**: Demonstration that a hard-masking mechanism inspired by vector DB pre-filtering ensures the base model's output loss remains identical to the decimal point before and after both insertion and removal.

## 2. Background: SRA Architecture

SRA (Synaptic Routing Architecture) is a dynamic, sparse architecture inspired by the biological brain. This section outlines the components essential for understanding Hot-Swap (see [Suzuki, 2026] for details).

### 2.1 Router
The router, the heart of SRA, is a **single linear layer** without any Attention mechanism. It computes the cosine similarity between the input hidden state $h$ and each synapse's feature vector (embedding) $e_i$, selecting the Top-k synapses.

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

where $\alpha$ is a scaling factor.

### 2.2 Tiny Synapses
Each synapse is an independent, tiny module consisting of a small Multi-Head Attention layer and an MLP. Only synapses selected by the router execute computations, achieving high computational efficiency.

### 2.3 Shared Trunk
A critical prerequisite for Hot-Swap is the **Shared Trunk** approach. All specialized synapses are derived from the same pre-trained base model (Embedding layers, Attention layers), independently training only the synapse components. This prevents divergence in internal vector representations (Representation Divergence) across models and enables synapse transplantation via physical copy.

## 3. Hot-Swap: Plug-In (Module Insertion)

The first operation of Hot-Swap is **Plug-In** — inserting independently trained specialized modules into the base model. This section demonstrates that the insertion process consists entirely of PyTorch tensor operations and is extremely simple.

### 3.1 Method

```python
# hotswap_model: Production base model (with empty slots added)
# plugin_math: Math-specialized LLM independently trained by the math team

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # Copy the router's embedding vectors
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Copy Expert (TinySynapse) weights (w1, w2)
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

The trained specialized model's tensors are simply assigned directly to specific indices (empty slots) of the base model's tensors. Because SRA trains only synapses while keeping the base model's shared knowledge (Attention layers, etc.) completely frozen via the Shared Trunk approach, this physical copy operation is valid.

### 3.2 Enabling Independent Parallel Development
This approach allows a "Code team" and a "Math team" to independently train their specialized synapses based on the same base model with zero mutual interference. After training, deployment is completed simply by memory-copying the weight tensors into the production base model's empty slots.

## 4. Zero Forgetting: Hard Masking Inspired by Vector DB Pre-filtering

### 4.1 Challenge: Router Confusion
Simply performing a physical tensor copy risks the router confusing old and new synapses, potentially altering the base model's output.

### 4.2 Metadata Filtering in Vector Databases
Modern vector databases such as Pinecone and Weaviate employ metadata filtering alongside cosine-similarity-based semantic search.

- **Post-filtering**: Excludes non-matching results after Top-K search. Prone to "K-NN depletion" where insufficient results remain after filtering.
- **Pre-filtering**: Restricts the search space via metadata masks **before** search, performing Top-K only among qualifying candidates. Noise is completely eliminated.

### 4.3 SRA's Pre-execution Hard Mask
The SRA router is essentially an **in-memory vector search engine (Maximum Inner Product Search: MIPS)** computing dot products between input vectors and synapse embedding vectors.

I incorporated vector DB Pre-filtering directly into the router's forward pass. At inference time, a metadata mask specifying the "set of synapses permitted for the current task" is provided to the model.

```python
# Router forward pass
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtering: Set logits of unauthorized synapses to -infinity
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-K routing
vals, idx = torch.topk(logits, k, dim=-1)
```

This `masked_fill` Pre-filtering ensures the router selects experts only from among permitted synapses. Regardless of how many other models' weights coexist, they are completely ignored in the computation graph, guaranteeing that **the base model's loss matches exactly to the decimal point before and after composition (mathematically zero interference)**.

### 4.4 Experimental Results
I compared the base model's (Code/Math/Text 3-domain language model) Validation Loss before and after Plug-In of independently trained specialized synapses. The loss matched exactly to the decimal point, empirically demonstrating Zero Forgetting.

## 5. Hot-Swap: Unplug (Module Removal)

The second operation of Hot-Swap is **Unplug** — removing modules that are no longer needed from the base model. If knowledge can be "plugged in," the ability to "unplug" it is equally essential. Machine Unlearning in monolithic models is extremely difficult due to the complex entanglement of parameters, but SRA's modular structure solves this problem through physical operations.

### 5.2 Approach 1: Physical Removal (pop_synapses)
When Hot-Swapped synapses are no longer needed, they are physically sliced off from the end of the tensor.

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**Advantage**: VRAM usage is physically reduced, and the model can be fully restored to its pre-addition state — like uninstalling an OS driver, a physical part of the AI's brain can be removed.

### 5.3 Approach 2: Zero-Clear Purging (clear_synapses)
When unplugging synapses at intermediate indices rather than the end, physical deletion would shift all subsequent synapse indices, breaking the metadata mask control system. Instead, the synapse contents are zero-cleared to create an "empty slot."

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

By invalidating only the slot contents without changing tensor size, index integrity is perfectly preserved. Empty slots can later be reused by overwriting with new synapses via Hot-Swap.

## 6. The Cosine Similarity Trap: The Zero Vector Black Hole Problem

### 6.1 Discovery
Upon implementing zero-clear purging for the Unplug operation, I encountered a critical bug where **output completely collapsed**.

### 6.2 Root Cause Analysis
The SRA router performs routing using cosine similarity. A zero-cleared synapse's embedding vector becomes $\mathbf{0}$, which remains $\mathbf{0}$ even after normalization. The cosine similarity between any input vector $h$ and the zero vector is $0.0$.

The problem arises because cosine similarity ranges over $[-1.0, 1.0]$. If a valid synapse's cosine similarity is negative (e.g., $-0.5$),**the empty synapse ($0.0$) becomes mathematically higher-scoring, causing the router to preferentially select the empty synapse**.

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

Data is "sucked into and vanishes at" what should be a nonexistent empty slot — a black hole-like behavior.

### 6.3 Solution: Complete Blockade via -∞ Masking
I added mask processing to detect and exclude zero-cleared synapses in the router's forward pass.

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Detect zero-cleared synapses
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

The $-\infty$ mask makes it mathematically impossible for empty synapses to be selected, regardless of how low other synapses' scores may be.

## 7. The Complete Lifecycle of Modular AI

The mechanisms described above enable SRA to realize the complete lifecycle of modular AI:

```
Train → Hot-Swap (Compose) → Serve
  ↓                            ↓
Independent                  Purge (Delete)
Parallel Dev                    ↓
                          Slot Reuse
                              ↓
                        New Hot-Swap
```

1. **Train**: Multiple teams share a base model and independently develop their specialized synapses in parallel.
2. **Compose**: Trained tensors are physically copied into the production base model for deployment.
3. **Serve**: Inference runs with Zero Forgetting guaranteed by hard-mask Pre-filtering.
4. **Delete**: Unnecessary synapses are physically removed or zero-cleared for purging.
5. **Reuse**: Empty slots are reused by Hot-Swapping new specialized synapses.

## 8. Discussion

### 8.1 Representation Divergence
The **absolute prerequisite** for Hot-Swap is that all specialized synapses are derived from the same pre-trained base model (Shared Trunk). Transplanting synapses between models trained completely independently causes routing collapse due to divergence in internal vector representations.

### 8.2 Super Router as an Alternative
To relax the Shared Trunk constraint, an approach has been validated where independent whole models are encapsulated and orchestrated by a Super Router using Gumbel-Softmax. This approach achieves perfect $1.0$ vs $0.0$ Hard Routing, enabling complete dynamic switching of computational resources even between models with different architectures.

### 8.3 Security Risks
The Hot-Swap capability introduces new security threat vectors due to its property of dynamically loading weight files from outside a running system. Key risks include: (1) arbitrary code execution via Pickle exploits, (2) malicious weight injection (Backdoor Injection), (3) routing hijacking via router key forgery, and (4) DoS attacks via swap thrashing. Mitigations such as mandatory `safetensors` format, cryptographic synapse signing, and rate limiting are recommended.

### 8.4 Current Limitations and Future Work
This research is at the experimental stage with small-scale models ($d_\text{model}=128$, $n_\text{layers}=4$). Validation on 10B-class large LLMs remains an important future challenge. Additionally, the router synchronization problem — the potential need for router key adaptation learning when adding synapses with entirely new capabilities — requires further investigation.

## 9. Conclusion

In this paper, I proposed and validated methods for making LLMs "Hotswappable" (dynamically pluggable) by leveraging the modularity of SRA (Synaptic Routing Architecture). Hot-Swap's Plug-In operation completes deployment through physical tensor copy alone, while the Unplug operation establishes two removal approaches: physical detachment and zero-clear purging. Through a hard-masking mechanism inspired by vector database Pre-filtering, Zero Forgetting is mathematically guaranteed. By discovering and resolving the "Black Hole Problem" of zero vectors in cosine similarity encountered during Unplug, safe slot reuse is achieved.

In an era where models continue to grow larger and more opaque, the "Hotswappable LLM" approach — enabling physical plug-and-unplug control of intelligence components — represents a highly promising direction for model maintainability, safety, and operational efficiency.

## References

- Suzuki, J. (2026). [All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

The complete Hot-Swap and Synapse Deletion processes described in this paper can be interactively experienced in the following Google Colab notebooks.

- **Hot-Swap Synapse Composition Demo (Base Training → Independent Training → Composition → Zero Forgetting Proof)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **Synapse Deletion Demo (Physical Removal → Zero-Clear → Black Hole Problem Resolution)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[The Future of SRA: Dynamic Hot-Swap and Extensibility](./sra_future_hotswap_ja.md)** — Discussion on cassette-style synapse operation, personalization, and distributed learning.
- **[Security Risks in SRA Hot-Swap](./sra_security_risks_hotswap_ja.md)** — Threat vectors including Pickle Exploit, Backdoor Injection, DoS attacks, and mitigation strategies.
- **[Representation Divergence and Hierarchical Routing](./sra_representation_divergence_ja.md)** — Shared Trunk approach and Super Router solutions.
- **[Hard Routing Comparison for SRA Hierarchical Router](./sra_hierarchical_hard_routing_ja.md)** — Comparative experiments of Soft / STE / Gumbel-Softmax.
