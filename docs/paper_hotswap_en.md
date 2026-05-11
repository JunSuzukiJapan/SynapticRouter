# Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion via Synaptic Routing Architecture

**Jun Suzuki**, Independent Researcher

## Abstract
Humans can postnatally learn new skills (e.g., riding a bicycle or a new language) and have them function as independent circuits in the brain without destroying existing knowledge. However, current massive language models (LLMs) possess a monolithic structure that densely stores all knowledge in a single parameter space, making the independent addition of postnatal knowledge or the deletion of specific memories extremely difficult, imposing severe constraints on operational flexibility.

In this paper, we leverage the functional localization mechanism of the Synaptic Routing Architecture (SRA) to propose and validate a method for the dynamic insertion and removal of neural circuits (modules)—namely, **Hot-Swap**. Hot-Swap is a bidirectional module exchange operation where specialized synapses, trained independently from the base model, can be **surgically transplanted (Plug-In) into the production model without any retraining**, and when no longer needed, **specific memories can be safely disconnected (Unplug)**. Experimental results show that, using a neural circuit hard-masking mechanism inspired by pre-filtering techniques in vector databases, we achieved **Zero Forgetting**, where the base model's output remains identical down to the decimal point even after transplantation. Additionally, we discovered and solved the "black hole problem" of zero vectors peculiar to cosine similarity during the unplug operation, realizing a complete lifecycle of modular AI—"Learn → Transplant → Remove → Reuse"—that transcends biological neural plasticity.

## 1. Introduction

### 1.1 The Limitations of Monolithic Models and Postnatal Memory Control
Since "Attention Is All You Need," the Transformer architecture has established a dominant position in natural language processing. However, monolithic LLMs with tens to hundreds of billions of parameters face the following severe problems regarding postnatal knowledge control:

1. **Catastrophic Forgetting**: When a general-purpose model is fine-tuned on a specific domain (e.g., company regulations, specialized code), its original general capabilities are destroyed or degraded.
2. **Bloated Training Costs**: Adding new knowledge requires retraining or merging the entire model, making complete parallel development across multiple teams difficult.
3. **Impossibility of Knowledge Deletion**: "Machine Unlearning," forgetting only specific knowledge, is a daunting task in monolithic models where parameters are intricately intertwined; attempting to retrain often destroys unrelated capabilities.

### 1.2 Contributions of this Paper
The Synaptic Routing Architecture (SRA) [Suzuki, 2026] I propose is an architecture where "functional localization" emerges autonomously according to tasks by simultaneously learning pathway selection and module representation end-to-end. This paper focuses on the **operational innovation** brought about by SRA's biological modularity—**Hot-Swap**, which allows neural circuits to be transplanted and removed at will—and reports the following three contributions:

1. **Hot-Swap: Plug-In (Surgical Transplantation)**: Implementation and validation of a method where deployment is completed simply by physically copying the weight tensors of an independently trained specialized module into the empty slots of a base model.
2. **Hot-Swap: Unplug (Memory Removal and Inactivation)**: The design of two deletion APIs—physical disconnection (`pop_synapses`) and inactivation purging via zero-clearing (`clear_synapses`)—and the discovery and resolution of the "zero vector black hole problem" in routing.
3. **Experimental Proof of Zero Forgetting**: Demonstration that the output Loss of the base model perfectly matches down to the decimal point before and after transplantation/removal, thanks to a hard-mask mechanism (Pre-filtering) that forcibly shuts down unnecessary pathways in specific tasks.

## 2. Background: SRA Architecture

SRA is a continual learning architecture that mimics the spatial isolation and dynamic routing of the brain. We outline the components necessary to understand Hot-Swap (see [Suzuki, 2026] for details).

### 2.1 Router (Dynamic Synaptic Formation)
The router, the heart of SRA, is a single linear layer that determines "which neural circuits should fire" for the input information. It calculates the cosine similarity between the input hidden state $h$ and the feature vector (embedding) $e_i$ of each synapse, selecting the Top-k synapses.

### 2.2 Tiny Synapses (Functional Modules)
Each synapse consists of an independent, extremely small Multi-Head Attention and MLP. Only the synapses instructed to "fire" by the router execute computations; the other parameters receive no interference.

### 2.3 Shared Trunk and Functional Localization
An absolutely critical condition for Hot-Swap to work is the **Shared Trunk approach**. All specialized synapses share the same pre-trained base model (Embedding/Attention layers corresponding to the visual cortex or basic language areas) and only train the synapse portions independently. This prevents Representation Divergence between models, enabling the physical transplantation of circuits.

## 3. Hot-Swap: Plug-In (Surgical Transplantation of Modules)

The first operation of Hot-Swap is to **Transplant (Plug-In)** an independently trained specialized module (a new skill) into the base model. This transplantation operation is completed solely through PyTorch tensor operations.

### 3.1 Method

```python
# hotswap_model: Base model in production (with empty brain slots added)
# plugin_math: Specialized LLM trained independently by the math team (new math circuit)

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # Copy the router's embedding vector (firing condition)
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Copy the weights (w1, w2) of the Expert (TinySynapse)
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

You simply directly assign the tensors of the trained specialized model to the empty slots of the base model. Because SRA trains only the synapses while keeping the shared knowledge of the base model completely frozen, this physical copy works.

### 3.2 Realizing Independent Parallel Development
This allows the "Code Team" and "Math Team" to independently train only their specialized synapses based on the same base model without any mutual interference. Once completed, deployment is finished just by memory-copying the weight tensors into the base model in the production environment.

## 4. Zero Forgetting: Hard-Mask Mechanism by Forcible Pathway Shutdown

### 4.1 The Problem: Router Confusion
Simply transplanting tensors risks the router confusing old and new synapses, negatively affecting (forgetting) the base model's existing capabilities.

### 4.2 Pre-filtering Inspired by Vector DBs
The essence of the SRA router is akin to an in-memory Maximum Inner Product Search engine. I incorporated the **Pre-filtering** technique of vector databases into the router's forward pass, designing a mechanism to physically shut down (hard-mask) "pathways to neural circuits unnecessary for the current task" during inference.

```python
# Forward pass of the Router layer
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtering: Forcibly shut down pathways of unauthorized synapses with -∞
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-K Routing
vals, idx = torch.topk(logits, k, dim=-1)
```

Thanks to this `-inf` mask, the router selects circuits only from the authorized synapses. No matter how many weights of other models are transplanted and coexist, they are completely ignored on the computational graph, guaranteeing that **the base model's Loss perfectly matches down to the decimal point compared to before the transplant (mathematically zero interference).**

## 5. Hot-Swap: Unplug (Removal/Inactivation of Specific Memories)

The second operation of Hot-Swap is to **Unplug** modules that are no longer needed from the base model. While Machine Unlearning in monolithic models is difficult, SRA solves this problem with the approach of "physical disconnection of synaptic connections."

### 5.1 Approach 1: Physical Disconnection (`pop_synapses`)
If a post-added synapse becomes unnecessary, it is physically discarded by slicing from the end of the tensor.

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

VRAM usage is reduced, and the model is completely restored to its state prior to adding the synapses. This is literally an operation to physically cut specific neural circuits.

### 5.2 Approach 2: Purging by Inactivation (`clear_synapses`)
If you want to delete a synapse in the middle rather than at the end, physical deletion would cause index shifts, collapsing the brain's control system. Therefore, we zero-clear the contents of the synapse to "inactivate" it (make it an empty slot).

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

By invalidating only the slot's contents without changing the tensor size, perfect consistency is maintained.

## 6. The Cosine Similarity Trap: The Zero Vector Black Hole Problem

### 6.1 Discovery and Cause of the Problem
When implementing the inactivation purge, I encountered a severe problem where **the output completely collapsed.**
SRA's router uses cosine similarity, but the embedding vector of a zero-cleared synapse is $\mathbf{0}$ even after normalization, resulting in a cosine similarity of $0.0$.
Since the range of similarity is $[-1.0, 1.0]$, if the score of a normal synapse is negative (e.g., $-0.5$), **the inactivated empty synapse ($0.0$) will have a higher score, causing the router to preferentially select the empty synapse.**

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

This was a "black hole"-like behavior where data was sucked into synapses that were supposed to be erased.

### 6.2 Solution: Complete Blockade with a -∞ Mask
To solve this problem, I added a masking process to the router's forward pass to detect zero-cleared synapses and completely block their pathways.

```python
# Detect zero-cleared (inactivated) synapses
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

With the $-\infty$ mask, it becomes mathematically impossible for inactivated synapses to be selected, achieving the safe reuse of slots.

## 7. The Complete Lifecycle of Modular AI

SRA realizes the following complete lifecycle of modular AI:

```
Learn (Train) → Transplant (Plug-In) → Operate (Serve)
       ↓                                   ↓
Independent Parallel Dev.            Remove (Unplug)
                                           ↓
                                Slot Reuse (Reuse)
                                           ↓
                                New Transplant (Plug-In)
```

This enables the infinite expansion and continuous maintenance of models, just as the biological brain continues to learn new skills throughout its life.

## 8. Discussion

### 8.1 Representation Divergence
An absolute condition for Hot-Swap is that all specialized synapses derive from the same pre-trained base model. Between models trained completely independently, routing collapses due to the divergence of internal representations.

### 8.2 Alternative: Super Router
To relax the shared trunk constraint, an approach is being investigated where entire independent models are encapsulated and bundled by a higher-level Super Router using Gumbel-Softmax.

### 8.3 Security Risks
Due to its nature of dynamically loading weight files from outside a running system, Hot-Swap introduces risks such as (1) Pickle Exploits, (2) Backdoor Injection, and (3) Routing Hijacking. Mandating the `safetensors` format and verifying synapses with cryptographic signatures are necessary.

## 9. Conclusion

In this paper, we established a "Hotswappable LLM" method that can dynamically transplant and remove specific neural circuits by leveraging SRA's functional localization mechanism. We successfully Plugged-In new skills solely through physical tensor copying, and Unplugged specific memories using physical disconnection and zero-clearing purging.

By mathematically guaranteeing Zero Forgetting with a hard-mask mechanism and solving the "black hole problem" in cosine similarity, this approach represents an extremely promising direction for evolving AI models from "monolithic black boxes" into "vital modular systems where parts of intelligence can be physically inserted and removed for safe control."

## References

- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.
