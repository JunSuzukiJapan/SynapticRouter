import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import PAD

class TinySynapse(nn.Module):
    """A small synapse module with self-attention for token mixing."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads=2, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.state = nn.Parameter(torch.zeros(dim))

    def forward(self, h, key_padding_mask=None, encoder_len=0, seq=None):
        B, T, D = h.shape
        attn_mask = torch.zeros((T, T), dtype=torch.bool, device=h.device)
        if encoder_len > 0:
            # source positions may attend only to source positions
            attn_mask[:encoder_len, encoder_len:] = True
            # target positions may attend to all source positions and past target positions
            future_target = torch.triu(torch.ones((T - encoder_len, T - encoder_len), dtype=torch.bool, device=h.device), diagonal=1)
            attn_mask[encoder_len:, encoder_len:] = future_target
        h_attn, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm1(h + h_attn)
        out = self.net(h)
        return self.norm2(out + h) + self.state


class VectorDBSynapse(nn.Module):
    """A synapse that acts as a Vector Database, returning fixed facts based on cosine similarity."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("keys", torch.empty(0, dim))
        self.register_buffer("values", torch.empty(0, dim))
        
    def add_knowledge(self, key_tensor: torch.Tensor, value_tensor: torch.Tensor):
        """Adds a new key-value pair to the database."""
        assert key_tensor.shape[-1] == self.dim and value_tensor.shape[-1] == self.dim
        if key_tensor.dim() == 1:
            key_tensor = key_tensor.unsqueeze(0)
            value_tensor = value_tensor.unsqueeze(0)
        self.keys = torch.cat([self.keys, key_tensor.to(self.keys.device)], dim=0)
        self.values = torch.cat([self.values, value_tensor.to(self.values.device)], dim=0)
        
    def forward(self, h, key_padding_mask=None, encoder_len=0, seq=None):
        # h: (B, T, D)
        if self.keys.size(0) == 0:
            return torch.zeros_like(h)
        
        # Calculate cosine similarity with all keys
        h_norm = F.normalize(h, p=2, dim=-1)
        k_norm = F.normalize(self.keys, p=2, dim=-1)
        sim = torch.einsum("btd,nd->btn", h_norm, k_norm)  # (B, T, N)
        
        # Get the index of the most similar key
        idx = sim.argmax(dim=-1)  # (B, T)
        
        # Fetch corresponding values
        out = self.values[idx]  # (B, T, D)
        return out


class RealCalculatorSynapse(nn.Module):
    """A synapse that performs deterministic rule-based calculation using Python's eval().
    It extracts the math expression from the input sequence, evaluates it, and produces 
    an output vector that forces the target tokens to be the answer.
    """
    def __init__(self, unembed_weight: torch.Tensor, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer("unembed_weight", unembed_weight.clone()) # (VocabSize, D)
        
    def forward(self, h, key_padding_mask=None, encoder_len=0, seq=None):
        out = torch.zeros_like(h)
        if seq is None or encoder_len == 0:
            return out
            
        B, T, D = h.shape
        import re
        
        for b in range(B):
            # Decode the source sequence
            src_tokens = seq[b, :encoder_len].tolist()
            # We assume char-level tokenization where token ID = ord(char)
            src_str = "".join([chr(c) for c in src_tokens if 32 <= c <= 126])
            
            # Find a math expression like "12+34" or "15*3"
            match = re.search(r'(\d+\s*[\+\-\*\/]\s*\d+)', src_str)
            if match:
                try:
                    # Evaluate the math expression
                    result = str(eval(match.group(1)))
                    
                    # For each character in the result, map it to the target sequence
                    # We also add BOS (which is 1) and EOS (which is 2) at the ends
                    from constants import BOS, EOS, PAD
                    result_tokens = [BOS] + [ord(c) for c in result] + [EOS]
                    for tgt_idx in range(encoder_len, T):
                        # Pad the rest of the sequence with PAD
                        ans_token = result_tokens[tgt_idx - encoder_len] if tgt_idx - encoder_len < len(result_tokens) else PAD
                        if ans_token < self.unembed_weight.size(0):
                            # Create a massive embedding that will force the unembed layer to pick this token.
                            # Since final output is base + out, and logits = (base + out) @ W^T,
                            # adding a huge vector parallel to W[ans_token] guarantees its selection.
                            vec = self.unembed_weight[ans_token]
                            out[b, tgt_idx] = vec * 100.0 / (vec.norm() + 1e-5)
                except Exception:
                    pass
                    
        return out


class Router(nn.Module):
    def __init__(self, dim: int, num_synapses: int, k: int):
        super().__init__()
        self.k = k
        self.num_synapses = num_synapses
        self.dim = dim
        self.synapse_emb = nn.Parameter(torch.zeros(num_synapses, dim))
        nn.init.orthogonal_(self.synapse_emb)
        self.synapse_emb.data = self.synapse_emb.data / math.sqrt(dim)
        self.scale = math.sqrt(dim)

    def add_synapses(self, num_new: int):
        if not hasattr(self, "frozen_synapse_emb"):
            self.register_buffer("frozen_synapse_emb", self.synapse_emb.data.clone())
        else:
            self.frozen_synapse_emb = torch.cat([self.frozen_synapse_emb, self.synapse_emb.data.clone()], dim=0)
            
        new_emb = torch.zeros(num_new, self.dim, device=self.synapse_emb.device)
        nn.init.orthogonal_(new_emb)
        new_emb = new_emb / self.scale
        self.synapse_emb = nn.Parameter(new_emb)
        self.num_synapses += num_new

    def add_custom_synapse(self, initial_embedding: torch.Tensor):
        """Adds a single custom synapse with a specific initial router embedding."""
        if not hasattr(self, "frozen_synapse_emb"):
            self.register_buffer("frozen_synapse_emb", self.synapse_emb.data.clone())
        else:
            self.frozen_synapse_emb = torch.cat([self.frozen_synapse_emb, self.synapse_emb.data.clone()], dim=0)
            
        emb = initial_embedding.to(self.synapse_emb.device)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
            
        self.synapse_emb = nn.Parameter(emb)
        self.num_synapses += 1

    def get_full_emb(self):
        if hasattr(self, "frozen_synapse_emb"):
            return torch.cat([self.frozen_synapse_emb, self.synapse_emb], dim=0)
        return self.synapse_emb

    def clear_synapses(self, indices_to_clear: list[int]):
        """Zero-clears the specified synapses, making them empty slots."""
        n_frozen = self.frozen_synapse_emb.size(0) if hasattr(self, "frozen_synapse_emb") else 0
        for idx in indices_to_clear:
            if idx < 0 or idx >= self.num_synapses:
                continue
            if idx < n_frozen:
                self.frozen_synapse_emb.data[idx].zero_()
            else:
                self.synapse_emb.data[idx - n_frozen].zero_()

    def pop_synapses(self, num_drop: int):
        """Physically removes the last num_drop synapses."""
        if num_drop <= 0:
            return
        assert num_drop < self.num_synapses, "Cannot drop all synapses."
        self.num_synapses -= num_drop
        
        n_frozen = self.frozen_synapse_emb.size(0) if hasattr(self, "frozen_synapse_emb") else 0
        n_active = self.synapse_emb.size(0)
        
        if num_drop <= n_active:
            if num_drop == n_active:
                self.synapse_emb = nn.Parameter(torch.empty(0, self.dim, device=self.synapse_emb.device))
            else:
                self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
        else:
            self.synapse_emb = nn.Parameter(torch.empty(0, self.dim, device=self.synapse_emb.device))
            drop_from_frozen = num_drop - n_active
            if drop_from_frozen == n_frozen:
                self.register_buffer("frozen_synapse_emb", torch.empty(0, self.dim, device=self.frozen_synapse_emb.device))
            else:
                self.register_buffer("frozen_synapse_emb", self.frozen_synapse_emb[:-drop_from_frozen].clone())

    def forward(self, h, k_override=None, allowed_mask=None):
        # h: (B, T, D)
        k = self.k if k_override is None else k_override
        full_emb = self.get_full_emb()
        
        # Phase 2 Fix: Cosine Similarity Routing
        # By normalizing both the input and the router embeddings, we ensure that new synapses 
        # cannot arbitrarily grow their weight magnitudes to hijack routing from older domains.
        h_norm = F.normalize(h, p=2, dim=-1)
        emb_norm = F.normalize(full_emb, p=2, dim=-1)
        logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale
        
        # Mask out zero-cleared synapses so they are never routed to
        is_cleared = (full_emb == 0).all(dim=-1)
        if is_cleared.any():
            logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
        
        # Phase 3: Metadata-driven Zero-Shot routing (Hard Masking)
        if allowed_mask is not None:
            # allowed_mask shape should be (B, num_synapses) or (B, 1, num_synapses)
            if allowed_mask.dim() == 2:
                allowed_mask = allowed_mask.unsqueeze(1)
            logits = logits.masked_fill(~allowed_mask, float('-inf'))
            
        if self.training and k == 1:
            weights_full = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
            idx = weights_full.argmax(dim=-1, keepdim=True)
            weights = weights_full.gather(dim=-1, index=idx)
            return idx, weights, logits
        else:
            vals, idx = torch.topk(logits, k, dim=-1)
            weights = F.softmax(vals, dim=-1)
            return idx, weights, logits


class SRABlock(nn.Module):
    def __init__(self, dim: int, num_synapses: int, k: int, syn_hidden: int):
        super().__init__()
        self.synapses = nn.ModuleList([TinySynapse(dim, syn_hidden) for _ in range(num_synapses)])
        self.router = Router(dim, num_synapses, k)
        self.norm = nn.LayerNorm(dim)

    def add_custom_synapse(self, module: nn.Module, initial_embedding: torch.Tensor):
        """Adds a custom module as a new synapse."""
        self.synapses.append(module)
        self.router.add_custom_synapse(initial_embedding)

    def forward(self, h, dense=False, key_padding_mask=None, encoder_len=0, allowed_mask=None, seq=None):
        base = h
        h = self.norm(h)
        k_override = self.router.num_synapses if dense else None
        idx, weights, logits = self.router(h, k_override=k_override, allowed_mask=allowed_mask)
        B, T, D = h.shape
        out = torch.zeros_like(h)
        syn_outputs = []  # record synapse outputs

        for syn_id, syn in enumerate(self.synapses):
            y = syn(h, key_padding_mask=key_padding_mask, encoder_len=encoder_len, seq=seq)  # (B, T, D)
            syn_outputs.append(y.detach())
            mask = (idx == syn_id).float()  # (B, T, k)
            coeff = (mask * weights).sum(dim=-1).unsqueeze(-1)  # (B, T, 1)
            out = out + coeff * y
        return base + out, logits, syn_outputs


class SRAModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, num_synapses: int, k: int, syn_hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        self.rel_pos = nn.Embedding(129, dim)
        self.seg = nn.Embedding(2, dim)
        self.blocks = nn.ModuleList([SRABlock(dim, num_synapses, k, syn_hidden) for _ in range(layers)])
        self.out = nn.Linear(dim, vocab_size)

    def add_custom_synapse(self, module_factory, initial_embedding_factory):
        """Adds custom synapses to all blocks. 
        module_factory: function that returns a new module.
        initial_embedding_factory: function that returns a new embedding tensor.
        """
        for block in self.blocks:
            module = module_factory()
            emb = initial_embedding_factory()
            block.add_custom_synapse(module, emb)

    def forward(self, x, y_in, dense=False, allowed_mask=None):
        # encoder-ish summary from input + autoregressive-ish target prefix conditioning
        seq = torch.cat([x, y_in], dim=1)
        mask = seq == PAD
        segment_ids = torch.cat(
            [torch.zeros_like(x), torch.ones_like(y_in)],
            dim=1,
        )
        target_rel_pos = torch.cat(
            [torch.zeros_like(x), torch.arange(1, y_in.size(1) + 1, device=seq.device).unsqueeze(0).repeat(x.size(0), 1)],
            dim=1,
        )
        h = (
            self.embed(seq)
            + self.pos[:, :seq.size(1)]
            + self.seg(segment_ids)
            + self.rel_pos(target_rel_pos)
        )
        router_logits = []
        all_synapse_outputs = []  # record all synapse outputs from all blocks
        for block in self.blocks:
            h, logits, syn_outs = block(h, dense=dense, key_padding_mask=mask, encoder_len=x.size(1), allowed_mask=allowed_mask, seq=seq)
            router_logits.append(logits)
            all_synapse_outputs.append(syn_outs)
        # predict only target positions
        h_tgt = h[:, x.size(1):]
        return self.out(h_tgt), router_logits, all_synapse_outputs


class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        self.rel_pos = nn.Embedding(129, dim)
        self.seg = nn.Embedding(2, dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=2, dim_feedforward=hidden, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x, y_in, dense=False):
        seq = torch.cat([x, y_in], dim=1)
        mask = seq == PAD
        segment_ids = torch.cat(
            [torch.zeros_like(x), torch.ones_like(y_in)],
            dim=1,
        )
        target_rel_pos = torch.cat(
            [torch.zeros_like(x), torch.arange(1, y_in.size(1) + 1, device=seq.device).unsqueeze(0).repeat(x.size(0), 1)],
            dim=1,
        )
        h = (
            self.embed(seq)
            + self.pos[:, :seq.size(1)]
            + self.seg(segment_ids)
            + self.rel_pos(target_rel_pos)
        )
        T = h.size(1)
        encoder_len = x.size(1)
        attn_mask = torch.zeros((T, T), dtype=torch.bool, device=h.device)
        attn_mask[:encoder_len, encoder_len:] = True
        future_target = torch.triu(torch.ones((T - encoder_len, T - encoder_len), dtype=torch.bool, device=h.device), diagonal=1)
        attn_mask[encoder_len:, encoder_len:] = future_target
        
        h = self.transformer(h, mask=attn_mask, src_key_padding_mask=mask)
        return self.out(h[:, x.size(1):]), [], []


class BaselineMLP(nn.Module):
    def __init__(self, vocab_size: int, dim: int, layers: int, hidden: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        self.rel_pos = nn.Embedding(129, dim)
        self.seg = nn.Embedding(2, dim)
        
        self.net = nn.ModuleList()
        for _ in range(layers):
            self.net.append(nn.Linear(dim, hidden))
            self.net.append(nn.GELU())
            self.net.append(nn.Linear(hidden, dim))
            self.net.append(nn.LayerNorm(dim))
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x, y_in, dense=False):
        seq = torch.cat([x, y_in], dim=1)
        segment_ids = torch.cat(
            [torch.zeros_like(x), torch.ones_like(y_in)],
            dim=1,
        )
        target_rel_pos = torch.cat(
            [torch.zeros_like(x), torch.arange(1, y_in.size(1) + 1, device=seq.device).unsqueeze(0).repeat(x.size(0), 1)],
            dim=1,
        )
        h = (
            self.embed(seq)
            + self.pos[:, :seq.size(1)]
            + self.seg(segment_ids)
            + self.rel_pos(target_rel_pos)
        )
        for i in range(0, len(self.net), 4):
            lin1, gelu, lin2, norm = self.net[i:i+4]
            h = norm(h + lin2(gelu(lin1(h))))
        return self.out(h[:, x.size(1):]), [], []
