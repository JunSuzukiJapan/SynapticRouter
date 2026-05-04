import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import PAD
from sra_reference import Router, TinySynapse

# ==============================================================================
# 1. BatchedSRAModel (Approach 1)
# ==============================================================================
class BatchedTinySynapse(nn.Module):
    """N synapses evaluated in parallel using einsum."""
    def __init__(self, num_synapses, dim, hidden):
        super().__init__()
        self.num_synapses = num_synapses
        self.dim = dim
        self.hidden = hidden
        self.num_heads = 2
        self.head_dim = dim // 2
        
        # MHA weights
        self.qkv_w = nn.Parameter(torch.randn(num_synapses, dim, 3 * dim) / math.sqrt(dim))
        self.qkv_b = nn.Parameter(torch.zeros(num_synapses, 3 * dim))
        self.out_w = nn.Parameter(torch.randn(num_synapses, dim, dim) / math.sqrt(dim))
        self.out_b = nn.Parameter(torch.zeros(num_synapses, dim))
        
        # MLP weights
        self.w1 = nn.Parameter(torch.randn(num_synapses, dim, hidden) / math.sqrt(dim))
        self.b1 = nn.Parameter(torch.zeros(num_synapses, hidden))
        self.w2 = nn.Parameter(torch.randn(num_synapses, hidden, dim) / math.sqrt(hidden))
        self.b2 = nn.Parameter(torch.zeros(num_synapses, dim))
        
        self.norm1_w = nn.Parameter(torch.ones(num_synapses, dim))
        self.norm1_b = nn.Parameter(torch.zeros(num_synapses, dim))
        self.norm2_w = nn.Parameter(torch.ones(num_synapses, dim))
        self.norm2_b = nn.Parameter(torch.zeros(num_synapses, dim))
        
        self.state = nn.Parameter(torch.zeros(num_synapses, dim))

    def _layer_norm(self, x, weight, bias):
        # x: (B, N, T, D), weight/bias: (N, D)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + 1e-5)
        return x_norm * weight.unsqueeze(0).unsqueeze(2) + bias.unsqueeze(0).unsqueeze(2)

    def forward(self, h, key_padding_mask=None, encoder_len=0):
        # h: (B, T, D)
        B, T, D = h.shape
        N = self.num_synapses
        
        # Expand to (B, N, T, D)
        x = h.unsqueeze(1).expand(B, N, T, D)
        
        # MHA
        qkv = torch.einsum('bntd,ndh->bnth', x, self.qkv_w) + self.qkv_b.unsqueeze(0).unsqueeze(2)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(B, N, T, self.num_heads, self.head_dim).transpose(2, 3) # (B, N, H, T, HD)
        k = k.view(B, N, T, self.num_heads, self.head_dim).transpose(2, 3)
        v = v.view(B, N, T, self.num_heads, self.head_dim).transpose(2, 3)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (B, N, H, T, T)
        
        # apply masks
        attn_mask = torch.zeros((T, T), dtype=torch.bool, device=h.device)
        if encoder_len > 0:
            attn_mask[:encoder_len, encoder_len:] = True
            future_target = torch.triu(torch.ones((T - encoder_len, T - encoder_len), dtype=torch.bool, device=h.device), diagonal=1)
            attn_mask[encoder_len:, encoder_len:] = future_target
            
        if attn_mask.any():
            scores = scores.masked_fill(attn_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0), float('-inf'))
            
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) -> (B, 1, 1, 1, T)
            scores = scores.masked_fill(key_padding_mask.view(B, 1, 1, 1, T), float('-inf'))
            
        attn = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(attn, v) # (B, N, H, T, HD)
        attn_out = attn_out.transpose(2, 3).reshape(B, N, T, D)
        attn_out = torch.einsum('bntd,ndh->bnth', attn_out, self.out_w) + self.out_b.unsqueeze(0).unsqueeze(2)
        
        h_norm1 = self._layer_norm(x + attn_out, self.norm1_w, self.norm1_b)
        
        # MLP
        mlp_h = F.gelu(torch.einsum('bntd,ndh->bnth', h_norm1, self.w1) + self.b1.unsqueeze(0).unsqueeze(2))
        mlp_out = torch.einsum('bnth,nhd->bntd', mlp_h, self.w2) + self.b2.unsqueeze(0).unsqueeze(2)
        
        out = self._layer_norm(h_norm1 + mlp_out, self.norm2_w, self.norm2_b) + self.state.unsqueeze(0).unsqueeze(2)
        return out # (B, N, T, D)

class BatchedSRABlock(nn.Module):
    def __init__(self, dim, num_synapses, k, syn_hidden):
        super().__init__()
        self.num_synapses = num_synapses
        self.synapses = BatchedTinySynapse(num_synapses, dim, syn_hidden)
        self.router = Router(dim, num_synapses, k)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, dense=False, key_padding_mask=None, encoder_len=0):
        base = h
        h = self.norm(h)
        k_override = self.router.num_synapses if dense else None
        idx, weights, logits = self.router(h, k_override=k_override) # idx: (B, T, k)
        
        B, T, D = h.shape
        # all synapses evaluated
        y = self.synapses(h, key_padding_mask=key_padding_mask, encoder_len=encoder_len) # (B, N, T, D)
        
        # Gather the top-k outputs
        y_trans = y.transpose(1, 2) # (B, T, N, D)
        idx_expanded = idx.unsqueeze(-1).expand(B, T, idx.size(-1), D)
        selected_y = torch.gather(y_trans, 2, idx_expanded) # (B, T, k, D)
        
        # multiply by weights: (B, T, k, 1)
        coeff = weights.unsqueeze(-1)
        out = (selected_y * coeff).sum(dim=2) # (B, T, D)
        
        # syn_outputs list for stats
        syn_outputs = [y[:, i].detach() for i in range(self.num_synapses)]
        return base + out, logits, syn_outputs

class BatchedSRAModel(nn.Module):
    def __init__(self, vocab_size, dim, layers, num_synapses, k, syn_hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        self.rel_pos = nn.Embedding(129, dim)
        self.seg = nn.Embedding(2, dim)
        self.blocks = nn.ModuleList([BatchedSRABlock(dim, num_synapses, k, syn_hidden) for _ in range(layers)])
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x, y_in, dense=False):
        seq = torch.cat([x, y_in], dim=1)
        mask = seq == PAD
        segment_ids = torch.cat([torch.zeros_like(x), torch.ones_like(y_in)], dim=1)
        target_rel_pos = torch.cat([torch.zeros_like(x), torch.arange(1, y_in.size(1) + 1, device=seq.device).unsqueeze(0).repeat(x.size(0), 1)], dim=1)
        h = self.embed(seq) + self.pos[:, :seq.size(1)] + self.seg(segment_ids) + self.rel_pos(target_rel_pos)
        
        router_logits = []
        all_synapse_outputs = []
        for block in self.blocks:
            h, logits, syn_outs = block(h, dense=dense, key_padding_mask=mask, encoder_len=x.size(1))
            router_logits.append(logits)
            all_synapse_outputs.append(syn_outs)
        return self.out(h[:, x.size(1):]), router_logits, all_synapse_outputs

# ==============================================================================
# 2. MoESRAModel (Approach 2)
# ==============================================================================
class MoESRABlock(nn.Module):
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

    def forward(self, h, dense=False, key_padding_mask=None, encoder_len=0):
        base = h
        B, T, D = h.shape
        
        # 1. Shared Attention
        attn_mask = torch.zeros((T, T), dtype=torch.bool, device=h.device)
        if encoder_len > 0:
            attn_mask[:encoder_len, encoder_len:] = True
            future_target = torch.triu(torch.ones((T - encoder_len, T - encoder_len), dtype=torch.bool, device=h.device), diagonal=1)
            attn_mask[encoder_len:, encoder_len:] = future_target
            
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
        
        for i in range(idx_flat.size(1)): # iterate over k
            expert_idx = idx_flat[:, i]
            expert_weights = weights_flat[:, i].unsqueeze(-1)
            
            w1_ex = self.w1[expert_idx] # (B*T, D, H)
            b1_ex = self.b1[expert_idx] # (B*T, H)
            w2_ex = self.w2[expert_idx] # (B*T, H, D)
            b2_ex = self.b2[expert_idx] # (B*T, D)
            state_ex = self.state[expert_idx]
            
            # bmm needs (B*T, 1, D) @ (B*T, D, H) -> (B*T, 1, H)
            hidden = torch.bmm(h_flat.unsqueeze(1), w1_ex).squeeze(1) + b1_ex
            hidden = F.gelu(hidden)
            expert_out = torch.bmm(hidden.unsqueeze(1), w2_ex).squeeze(1) + b2_ex + state_ex
            
            out_flat = out_flat + expert_out * expert_weights
            
        out = out_flat.view(B, T, D)
        # Dummy syn_outputs for stats to avoid breaking logging
        dummy_syn_outs = [torch.zeros(B, T, D, device=h.device) for _ in range(self.num_synapses)]
        return base + out, logits, dummy_syn_outs

class MoESRAModel(nn.Module):
    def __init__(self, vocab_size, dim, layers, num_synapses, k, syn_hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        self.rel_pos = nn.Embedding(129, dim)
        self.seg = nn.Embedding(2, dim)
        self.blocks = nn.ModuleList([MoESRABlock(dim, num_synapses, k, syn_hidden) for _ in range(layers)])
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x, y_in, dense=False):
        seq = torch.cat([x, y_in], dim=1)
        mask = seq == PAD
        segment_ids = torch.cat([torch.zeros_like(x), torch.ones_like(y_in)], dim=1)
        target_rel_pos = torch.cat([torch.zeros_like(x), torch.arange(1, y_in.size(1) + 1, device=seq.device).unsqueeze(0).repeat(x.size(0), 1)], dim=1)
        h = self.embed(seq) + self.pos[:, :seq.size(1)] + self.seg(segment_ids) + self.rel_pos(target_rel_pos)
        
        router_logits = []
        all_synapse_outputs = []
        for block in self.blocks:
            h, logits, syn_outs = block(h, dense=dense, key_padding_mask=mask, encoder_len=x.size(1))
            router_logits.append(logits)
            all_synapse_outputs.append(syn_outs)
        return self.out(h[:, x.size(1):]), router_logits, all_synapse_outputs

# ==============================================================================
# 3. SeqSRAModel (Approach 3)
# ==============================================================================
class SeqRouter(nn.Module):
    def __init__(self, dim, num_synapses, k):
        super().__init__()
        self.k = k
        self.num_synapses = num_synapses
        self.synapse_emb = nn.Parameter(torch.zeros(num_synapses, dim))
        nn.init.orthogonal_(self.synapse_emb)
        self.synapse_emb.data = self.synapse_emb.data / math.sqrt(dim)
        self.scale = math.sqrt(dim)

    def forward(self, h, mask=None, k_override=None):
        k = self.k if k_override is None else k_override
        if mask is not None:
            h_masked = h.masked_fill(mask.unsqueeze(-1), 0.0)
            lens = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            h_pool = h_masked.sum(dim=1) / lens
        else:
            h_pool = h.mean(dim=1)
            
        logits = torch.einsum("bd,nd->bn", h_pool, self.synapse_emb) / self.scale
        vals, idx = torch.topk(logits, k, dim=-1)
        weights = F.softmax(vals, dim=-1)
        return idx, weights, logits

class SeqSRABlock(nn.Module):
    def __init__(self, dim, num_synapses, k, syn_hidden):
        super().__init__()
        self.num_synapses = num_synapses
        self.synapses = nn.ModuleList([TinySynapse(dim, syn_hidden) for _ in range(num_synapses)])
        self.router = SeqRouter(dim, num_synapses, k)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, dense=False, key_padding_mask=None, encoder_len=0):
        base = h
        h = self.norm(h)
        k_override = self.num_synapses if dense else None
        idx, weights, logits = self.router(h, mask=key_padding_mask, k_override=k_override)
        
        B, T, D = h.shape
        out = torch.zeros_like(h)
        syn_outputs = []
        
        for syn_id, syn in enumerate(self.synapses):
            batch_mask = (idx == syn_id) # (B, k)
            if not batch_mask.any():
                syn_outputs.append(torch.zeros_like(h))
                continue
                
            b_indices = batch_mask.any(dim=-1).nonzero().squeeze(-1)
            h_sub = h[b_indices]
            mask_sub = key_padding_mask[b_indices] if key_padding_mask is not None else None
            
            y_sub = syn(h_sub, key_padding_mask=mask_sub, encoder_len=encoder_len)
            
            # extract weights for chosen items
            weight_sub = weights[batch_mask].view(-1, 1, 1)
            
            out[b_indices] += y_sub * weight_sub
            
            y_full = torch.zeros_like(h)
            y_full[b_indices] = y_sub.detach()
            syn_outputs.append(y_full)
            
        logits_expanded = logits.unsqueeze(1).expand(B, T, self.num_synapses)
        return base + out, logits_expanded, syn_outputs

class SeqSRAModel(nn.Module):
    def __init__(self, vocab_size, dim, layers, num_synapses, k, syn_hidden):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.pos = nn.Parameter(torch.randn(1, 128, dim) * 0.02)
        self.rel_pos = nn.Embedding(129, dim)
        self.seg = nn.Embedding(2, dim)
        self.blocks = nn.ModuleList([SeqSRABlock(dim, num_synapses, k, syn_hidden) for _ in range(layers)])
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, x, y_in, dense=False):
        seq = torch.cat([x, y_in], dim=1)
        mask = seq == PAD
        segment_ids = torch.cat([torch.zeros_like(x), torch.ones_like(y_in)], dim=1)
        target_rel_pos = torch.cat([torch.zeros_like(x), torch.arange(1, y_in.size(1) + 1, device=seq.device).unsqueeze(0).repeat(x.size(0), 1)], dim=1)
        h = self.embed(seq) + self.pos[:, :seq.size(1)] + self.seg(segment_ids) + self.rel_pos(target_rel_pos)
        
        router_logits = []
        all_synapse_outputs = []
        for block in self.blocks:
            h, logits, syn_outs = block(h, dense=dense, key_padding_mask=mask, encoder_len=x.size(1))
            router_logits.append(logits)
            all_synapse_outputs.append(syn_outs)
        return self.out(h[:, x.size(1):]), router_logits, all_synapse_outputs
