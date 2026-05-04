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

    def forward(self, h, key_padding_mask=None, encoder_len=0):
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


class Router(nn.Module):
    def __init__(self, dim: int, num_synapses: int, k: int):
        super().__init__()
        self.k = k
        self.num_synapses = num_synapses
        # Initialize with orthogonal matrix to balance initial routing
        self.synapse_emb = nn.Parameter(torch.zeros(num_synapses, dim))
        nn.init.orthogonal_(self.synapse_emb)
        self.synapse_emb.data = self.synapse_emb.data / math.sqrt(dim)
        self.scale = math.sqrt(dim)

    def forward(self, h, k_override=None):
        # h: (B, T, D)
        k = self.k if k_override is None else k_override
        logits = torch.einsum("btd,nd->btn", h, self.synapse_emb) / self.scale
        vals, idx = torch.topk(logits, k, dim=-1)
        weights = F.softmax(vals, dim=-1)
        return idx, weights, logits


class SRABlock(nn.Module):
    def __init__(self, dim: int, num_synapses: int, k: int, syn_hidden: int):
        super().__init__()
        self.synapses = nn.ModuleList([TinySynapse(dim, syn_hidden) for _ in range(num_synapses)])
        self.router = Router(dim, num_synapses, k)
        self.norm = nn.LayerNorm(dim)

    def forward(self, h, dense=False, key_padding_mask=None, encoder_len=0):
        base = h
        h = self.norm(h)
        k_override = self.router.num_synapses if dense else None
        idx, weights, logits = self.router(h, k_override=k_override)
        B, T, D = h.shape
        out = torch.zeros_like(h)
        syn_outputs = []  # record synapse outputs

        for syn_id, syn in enumerate(self.synapses):
            y = syn(h, key_padding_mask=key_padding_mask, encoder_len=encoder_len)  # (B, T, D)
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

    def forward(self, x, y_in, dense=False):
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
            h, logits, syn_outs = block(h, dense=dense, key_padding_mask=mask, encoder_len=x.size(1))
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
