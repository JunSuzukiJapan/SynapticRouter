"""
SRA Encoder-Decoder 翻訳モデル
- SRAEncoderLayer : 双方向 Self-Attention + MoE-SRA FFN
- SRADecoderLayer : Causal Self-Attention + Cross-Attention + MoE-SRA FFN
- SRATranslationModel : Encoder + Decoder + weight-tied output projection
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sra_reference import Router


# ─────────────────────────────────────────────────────────────
# MoE-SRA FFN Block（Encoder / Decoder 共通）
# ─────────────────────────────────────────────────────────────
class SRAFFNBlock(nn.Module):
    """
    MoE-SRA FFN。Expert ループはメモリ効率優先の boolean-mask 版。
    Pre-LN 構成（LayerNorm → routing → 残差加算）。
    """
    def __init__(self, dim: int, num_synapses: int, k: int, syn_hidden: int):
        super().__init__()
        self.num_synapses = num_synapses
        self.k = k
        self.norm = nn.LayerNorm(dim)
        self.router = Router(dim, num_synapses, k)

        self.w1 = nn.Parameter(torch.randn(num_synapses, dim, syn_hidden) / math.sqrt(dim))
        self.b1 = nn.Parameter(torch.zeros(num_synapses, syn_hidden))
        self.w2 = nn.Parameter(torch.randn(num_synapses, syn_hidden, dim) / math.sqrt(syn_hidden))
        self.b2 = nn.Parameter(torch.zeros(num_synapses, dim))
        self.state = nn.Parameter(torch.zeros(num_synapses, dim))

    def _expert_forward(self, h_flat, idx_flat, weights_flat):
        N, D = h_flat.shape
        out = torch.zeros_like(h_flat)
        for e in range(self.num_synapses):
            mask = (idx_flat == e).any(dim=-1)
            if not mask.any():
                continue
            h_sub = h_flat[mask]
            w_sub = weights_flat[mask]
            idx_sub = idx_flat[mask]
            hidden = F.gelu(h_sub @ self.w1[e] + self.b1[e])
            expert_out = hidden @ self.w2[e] + self.b2[e] + self.state[e]
            coeff = (w_sub * (idx_sub == e).float()).sum(dim=-1, keepdim=True)
            out[mask] += expert_out * coeff
        return out

    def forward(self, h: torch.Tensor, dense: bool = False):
        """
        h: (B, T, D)
        Returns: (h + ffn_out, router_logits)
        """
        B, T, D = h.shape
        h_norm = self.norm(h)
        k_use = self.num_synapses if dense else self.k
        idx, weights, logits = self.router(h_norm, k_override=k_use)
        out_flat = self._expert_forward(
            h_norm.view(B * T, D),
            idx.view(B * T, -1),
            weights.view(B * T, -1),
        )
        return h + out_flat.view(B, T, D), logits


# ─────────────────────────────────────────────────────────────
# Encoder Layer（双方向）
# ─────────────────────────────────────────────────────────────
class SRAEncoderLayer(nn.Module):
    def __init__(self, dim: int, num_synapses: int, k: int, syn_hidden: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_attn = nn.LayerNorm(dim)
        self.ffn = SRAFFNBlock(dim, num_synapses, k, syn_hidden)

    def forward(self, h, src_key_padding_mask=None, dense=False):
        # 双方向 Self-Attention（causal mask なし）
        attn_out, _ = self.attn(
            h, h, h,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        h = self.norm_attn(h + attn_out)
        h, logits = self.ffn(h, dense=dense)
        return h, logits


# ─────────────────────────────────────────────────────────────
# Decoder Layer（Causal Self-Attn + Cross-Attn）
# ─────────────────────────────────────────────────────────────
class SRADecoderLayer(nn.Module):
    def __init__(self, dim: int, num_synapses: int, k: int, syn_hidden: int, num_heads: int = 4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_self = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.ffn = SRAFFNBlock(dim, num_synapses, k, syn_hidden)

    def forward(self, h, enc_out, causal_mask=None, src_key_padding_mask=None, dense=False):
        T = h.shape[1]
        # Causal mask を動的生成
        if causal_mask is None:
            causal_mask = torch.triu(
                torch.ones(T, T, dtype=torch.bool, device=h.device), diagonal=1
            )
        # Causal Self-Attention
        sa_out, _ = self.self_attn(h, h, h, attn_mask=causal_mask, need_weights=False)
        h = self.norm_self(h + sa_out)
        # Cross-Attention（encoder 表現に強制 attend）
        ca_out, _ = self.cross_attn(
            h, enc_out, enc_out,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        h = self.norm_cross(h + ca_out)
        # MoE-SRA FFN
        h, logits = self.ffn(h, dense=dense)
        return h, logits


# ─────────────────────────────────────────────────────────────
# SRA Translation Model（Encoder-Decoder）
# ─────────────────────────────────────────────────────────────
class SRATranslationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int,
        enc_layers: int,
        dec_layers: int,
        num_synapses: int,
        k: int,
        syn_hidden: int,
        num_heads: int = 4,
        pad_idx: int = 0,
        max_src_len: int = 80,
        max_tgt_len: int = 80,
    ):
        super().__init__()
        self.dim = dim
        self.pad_idx = pad_idx
        self.scale = math.sqrt(dim)

        # 共有埋め込み（src/tgt の語彙は同じ tiktoken vocab）
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        self.src_pos = nn.Embedding(max_src_len, dim)
        self.tgt_pos = nn.Embedding(max_tgt_len, dim)

        # Encoder
        self.enc_layers = nn.ModuleList([
            SRAEncoderLayer(dim, num_synapses, k, syn_hidden, num_heads)
            for _ in range(enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(dim)

        # Decoder
        self.dec_layers = nn.ModuleList([
            SRADecoderLayer(dim, num_synapses, k, syn_hidden, num_heads)
            for _ in range(dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(dim)

        # Output projection（embedding と重み共有）
        self.out = nn.Linear(dim, vocab_size, bias=False)
        self.out.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.src_pos.weight, std=0.02)
        nn.init.normal_(self.tgt_pos.weight, std=0.02)

    # ----------------------------------------------------------
    def encode(self, src: torch.Tensor, dense: bool = False):
        """
        src: (B, S) token IDs
        Returns: enc_out (B, S, D), src_pad_mask (B, S), router_logits list
        """
        B, S = src.shape
        src_pad_mask = (src == self.pad_idx)

        pos = torch.arange(S, device=src.device).unsqueeze(0)
        h = self.embed(src) * self.scale + self.src_pos(pos)

        enc_logits = []
        for layer in self.enc_layers:
            h, logits = layer(h, src_key_padding_mask=src_pad_mask, dense=dense)
            enc_logits.append(logits)

        return self.enc_norm(h), src_pad_mask, enc_logits

    def decode(self, tgt: torch.Tensor, enc_out: torch.Tensor,
               src_pad_mask: torch.Tensor, dense: bool = False):
        """
        tgt: (B, T) token IDs（decoder input, BOS-shifted）
        enc_out: (B, S, D)
        src_pad_mask: (B, S) bool
        Returns: logits (B, T, V), router_logits list
        """
        B, T = tgt.shape
        pos = torch.arange(T, device=tgt.device).unsqueeze(0)
        h = self.embed(tgt) * self.scale + self.tgt_pos(pos)

        causal_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=tgt.device), diagonal=1
        )

        dec_logits = []
        for layer in self.dec_layers:
            h, logits = layer(
                h, enc_out,
                causal_mask=causal_mask,
                src_key_padding_mask=src_pad_mask,
                dense=dense,
            )
            dec_logits.append(logits)

        h = self.dec_norm(h)
        return self.out(h), dec_logits

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, dense: bool = False):
        """
        src: (B, S)  source token IDs
        tgt: (B, T)  target token IDs（BOS から始まる）
        Returns: logits (B, T, V), all_router_logits
        """
        enc_out, src_pad_mask, enc_logits = self.encode(src, dense=dense)
        logits, dec_logits = self.decode(tgt, enc_out, src_pad_mask, dense=dense)
        return logits, enc_logits + dec_logits

    @torch.no_grad()
    def greedy_decode(self, src: torch.Tensor, bos_id: int, eos_id: int,
                      max_len: int = 80) -> torch.Tensor:
        """評価用 greedy decoding"""
        enc_out, src_pad_mask, _ = self.encode(src, dense=False)
        B = src.shape[0]
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_len - 1):
            logits, _ = self.decode(generated, enc_out, src_pad_mask, dense=False)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=1)
            if (next_tok == eos_id).all():
                break
        return generated  # (B, T_gen)
