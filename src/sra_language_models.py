import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from sra_reference import Router


class CausalMoESRABlock(nn.Module):
    """
    MoE-SRA ブロック（MPS/CUDA 対応ベクトル化版）

    Expert ループを全て einsum/scatter_add でベクトル化し、
    Python for-loop を排除して MPS 上での速度を最大化する。
    """
    def __init__(self, dim, num_synapses, k, syn_hidden):
        super().__init__()
        self.k = k
        self.num_synapses = num_synapses
        self.dim = dim
        self.syn_hidden = syn_hidden

        # Shared causal attention（num_heads を dim に合わせて自動調整）
        num_heads = max(1, min(8, dim // 64))
        # dim が num_heads で割り切れない場合の保険
        while dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        self.router = Router(dim, num_synapses, k)
        self.norm2 = nn.LayerNorm(dim)

        # Expert MLP パラメータ (num_synapses, ...)
        self.w1 = nn.Parameter(torch.randn(num_synapses, dim, syn_hidden) / math.sqrt(dim))
        self.b1 = nn.Parameter(torch.zeros(num_synapses, syn_hidden))
        self.w2 = nn.Parameter(torch.randn(num_synapses, syn_hidden, dim) / math.sqrt(syn_hidden))
        self.b2 = nn.Parameter(torch.zeros(num_synapses, dim))
        self.state = nn.Parameter(torch.zeros(num_synapses, dim))

    def _moe_forward(self, h_flat, idx_flat, weights_flat):
        """
        Expert ごとに matmul を実行するメモリ効率版。
        全トークンを一度に展開するのではなく、Expert 単位でバッチ matmul する。

        h_flat      : (N, D)   N = B*T
        idx_flat    : (N, k)
        weights_flat: (N, k)
        """
        N, D = h_flat.shape
        out_flat = torch.zeros_like(h_flat)

        for e in range(self.num_synapses):
            # このExpertを参照しているトークンのマスク: (N, k) → (N,)
            token_mask = (idx_flat == e).any(dim=-1)  # (N,)
            if not token_mask.any():
                continue

            # 担当トークンを抽出
            h_sub = h_flat[token_mask]                  # (M, D)
            idx_sub = idx_flat[token_mask]              # (M, k)
            w_sub = weights_flat[token_mask]            # (M, k)

            # Expert MLP: h_sub → hidden → expert_out
            hidden = F.gelu(h_sub @ self.w1[e] + self.b1[e])   # (M, H)
            expert_out_raw = hidden @ self.w2[e] + self.b2[e] + self.state[e]  # (M, D)
            
            # API Standardization Phase 1: L2 Normalization
            # L2 norm keeps the vector length to 1, then we scale by sqrt(dim) to match expected variance
            expert_out = F.normalize(expert_out_raw, p=2, dim=-1) * math.sqrt(self.dim)

            # このExpertへの重みを取り出して加算（同一トークンが複数スロットで同Expertを選ぶ場合も合算）
            coeff = (w_sub * (idx_sub == e).float()).sum(dim=-1, keepdim=True)  # (M, 1)
            out_flat[token_mask] += expert_out * coeff

        return out_flat

    def forward(self, h, dense=False, key_padding_mask=None):
        base = h
        B, T, D = h.shape

        # 1. Causal Shared Attention
        attn_mask = torch.triu(
            torch.ones((T, T), dtype=torch.bool, device=h.device), diagonal=1
        )
        attn_out, _ = self.attn(
            h, h, h,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        h = self.norm1(h + attn_out)

        # 2. MoE Routing
        h_routed = self.norm2(h)
        k_override = self.num_synapses if dense else self.k
        idx, weights, logits = self.router(h_routed, k_override=k_override)

        # 3. ベクトル化 Expert 計算
        h_flat = h_routed.view(B * T, D)
        idx_flat = idx.view(B * T, -1)
        weights_flat = weights.view(B * T, -1)

        out_flat = self._moe_forward(h_flat, idx_flat, weights_flat)
        out = out_flat.view(B, T, D)

        return base + out, logits


class MoESRALanguageModel(nn.Module):
    def __init__(self, vocab_size, dim, layers, num_synapses, k, syn_hidden,
                 pad_idx=None, max_seq_len=512):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        self.pos = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            CausalMoESRABlock(dim, num_synapses, k, syn_hidden)
            for _ in range(layers)
        ])
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
