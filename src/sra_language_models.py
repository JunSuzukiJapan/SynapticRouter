import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sra_reference import Router

class CausalMoESRABlock(nn.Module):
    def __init__(self, dim, num_synapses, k, syn_hidden):
        super().__init__()
        self.k = k
        self.num_synapses = num_synapses
        # Shared attention
        self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        
        self.router = Router(dim, num_synapses, k)
        self.norm2 = nn.LayerNorm(dim)
        
        # Expert MLPs
        self.w1 = nn.Parameter(torch.randn(num_synapses, dim, syn_hidden) / math.sqrt(dim))
        self.b1 = nn.Parameter(torch.zeros(num_synapses, syn_hidden))
        self.w2 = nn.Parameter(torch.randn(num_synapses, syn_hidden, dim) / math.sqrt(syn_hidden))
        self.b2 = nn.Parameter(torch.zeros(num_synapses, dim))
        self.state = nn.Parameter(torch.zeros(num_synapses, dim))

    def forward(self, h, dense=False, key_padding_mask=None):
        base = h
        B, T, D = h.shape
        
        # 1. Causal Shared Attention
        attn_mask = torch.triu(torch.ones((T, T), dtype=torch.bool, device=h.device), diagonal=1)
            
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        h = self.norm1(h + attn_out)
        
        # 2. MoE Routing
        h_routed = h
        h_routed = self.norm2(h_routed)
        k_override = self.num_synapses if dense else self.k
        idx, weights, logits = self.router(h_routed, k_override=k_override)
        
        # Gather/Scatter
        h_flat = h_routed.view(B*T, D)
        idx_flat = idx.view(B*T, -1)
        weights_flat = weights.view(B*T, -1)
        out_flat = torch.zeros_like(h_flat)
        
        for e in range(self.num_synapses):
            mask = (idx_flat == e) # (B*T, k)
            if not mask.any(): continue
            
            token_indices = mask.any(dim=-1).nonzero().squeeze(-1) # (num_tokens,)
            h_sub = h_flat[token_indices] # (num_tokens, D)
            
            w1_ex = self.w1[e] # (D, H)
            b1_ex = self.b1[e] # (H)
            w2_ex = self.w2[e] # (H, D)
            b2_ex = self.b2[e] # (D)
            state_ex = self.state[e] # (D)
            
            hidden = F.gelu(torch.matmul(h_sub, w1_ex) + b1_ex)
            expert_out = torch.matmul(hidden, w2_ex) + b2_ex + state_ex # (num_tokens, D)
            
            expert_weights = weights_flat[mask] # (num_tokens,)
            out_flat[token_indices] += expert_out * expert_weights.unsqueeze(-1)
            
        out = out_flat.view(B, T, D)
        return base + out, logits

class MoESRALanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, layers, num_synapses, k, syn_hidden, pad_idx=None, max_seq_len=512):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([CausalMoESRABlock(dim, num_synapses, k, syn_hidden) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x, dense=False):
        B, T = x.shape
        mask = (x == self.pad_idx) if self.pad_idx is not None else None
        
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.embed(x) + self.pos(positions)
        
        router_logits = []
        for block in self.blocks:
            h, logits = block(h, dense=dense, key_padding_mask=mask)
            router_logits.append(logits)
            
        h = self.norm(h)
        return self.out(h), router_logits
