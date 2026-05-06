"""
SRA Seq2Seq 翻訳モデル — 推論・評価スクリプト
- Beam search 推論（beam_width=4）
- 全方向 BLEU 評価（EN↔FR, EN↔JA, FR↔JA）
- ルーティング分析（言語方向別シナプス使用率）
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import heapq
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F
import tiktoken
import numpy as np

from sra_seq2seq_model import SRATranslationModel
from data_loader_seq2seq import build_special_tokens

# ──────────────────────────────────────────────
# テストセット
# ──────────────────────────────────────────────
TEST_PAIRS = [
    # EN → FR
    ("en", "fr", "I eat apples.",             "Je mange des pommes."),
    ("en", "fr", "She reads books.",           "Elle lit des livres."),
    ("en", "fr", "Good morning.",              "Bonjour."),
    ("en", "fr", "Thank you very much.",       "Merci beaucoup."),
    ("en", "fr", "The dog runs fast.",         "Le chien court vite."),
    ("en", "fr", "He drives a car.",           "Il conduit une voiture."),
    ("en", "fr", "We love music.",             "Nous aimons la musique."),
    ("en", "fr", "The cat sleeps.",            "Le chat dort."),
    # FR → EN
    ("fr", "en", "Je mange des pommes.",       "I eat apples."),
    ("fr", "en", "Elle lit des livres.",        "She reads books."),
    ("fr", "en", "Bonjour.",                   "Good morning."),
    ("fr", "en", "Merci beaucoup.",            "Thank you very much."),
    ("fr", "en", "Il conduit une voiture.",    "He drives a car."),
    # EN → JA
    ("en", "ja", "I eat apples.",              "私はりんごを食べます。"),
    ("en", "ja", "Good morning.",              "おはようございます。"),
    ("en", "ja", "Thank you very much.",       "ありがとうございます。"),
    ("en", "ja", "She reads books.",           "彼女は本を読みます。"),
    # JA → EN
    ("ja", "en", "私 は りんご を 食べる。",   "I eat apples."),
    ("ja", "en", "おはようございます。",       "Good morning."),
    ("ja", "en", "彼女 は 本 を 読む。",       "She reads books."),
    # Zero-shot (学習に含まれない方向)
    ("fr", "ja", "Je mange des pommes.",       "私はりんごを食べます。"),
    ("ja", "fr", "私 は りんご を 食べる。",   "Je mange des pommes."),
]


# ──────────────────────────────────────────────
# BLEU 計算（1〜2 gram）
# ──────────────────────────────────────────────
def ngrams(tokens, n):
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def bleu_score(hyp: str, ref: str, max_n: int = 2) -> float:
    hyp_tok = hyp.lower().split()
    ref_tok = ref.lower().split()
    if not hyp_tok:
        return 0.0
    scores = []
    for n in range(1, max_n + 1):
        h_ng = ngrams(hyp_tok, n)
        r_ng = ngrams(ref_tok, n)
        if not h_ng:
            scores.append(0.0)
            continue
        ref_cnt: dict = {}
        for g in r_ng:
            ref_cnt[g] = ref_cnt.get(g, 0) + 1
        match = 0
        for g in h_ng:
            if ref_cnt.get(g, 0) > 0:
                match += 1
                ref_cnt[g] -= 1
        scores.append(match / len(h_ng))
    bp = min(1.0, len(hyp_tok) / max(len(ref_tok), 1))
    return bp * math.exp(sum(math.log(s + 1e-9) for s in scores) / max_n)


# ──────────────────────────────────────────────
# Beam search
# ──────────────────────────────────────────────
@torch.no_grad()
def beam_decode(
    model: SRATranslationModel,
    src: torch.Tensor,          # (1, S)
    bos_id: int,
    eos_id: int,
    pad_id: int,
    beam_width: int = 4,
    max_len: int = 80,
) -> List[int]:
    """単一文の beam search デコード"""
    device = src.device
    enc_out, src_pad_mask, _ = model.encode(src)  # (1, S, D)

    # (score, token_list)
    beams: List[Tuple[float, List[int]]] = [(0.0, [bos_id])]
    completed: List[Tuple[float, List[int]]] = []

    for _ in range(max_len):
        candidates: List[Tuple[float, List[int]]] = []

        for score, tokens in beams:
            if tokens[-1] == eos_id:
                completed.append((score, tokens))
                continue

            tgt = torch.tensor([tokens], dtype=torch.long, device=device)
            logits, _ = model.decode(tgt, enc_out, src_pad_mask)  # (1, T, V)
            log_probs = F.log_softmax(logits[0, -1], dim=-1)      # (V,)

            top_vals, top_ids = torch.topk(log_probs, beam_width)
            for lp, tok in zip(top_vals.tolist(), top_ids.tolist()):
                candidates.append((score + lp, tokens + [tok]))

        if not candidates:
            break

        # top-k を残す
        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]

        # 全ビームが EOS に達したら終了
        if all(b[1][-1] == eos_id for b in beams):
            break

    # 完了ビームと未完了ビームをマージ
    completed += beams
    if not completed:
        return [bos_id]

    # 長さ正規化スコアで最良を選択
    best_score, best_tokens = max(
        completed,
        key=lambda x: x[0] / max(len(x[1]), 1),
    )
    return best_tokens


# ──────────────────────────────────────────────
# 推論ループ
# ──────────────────────────────────────────────
def run_inference(args):
    device = "cpu"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    sp = build_special_tokens(tokenizer)
    vocab_size = tokenizer.n_vocab + 100

    model = SRATranslationModel(
        vocab_size=vocab_size,
        dim=args.dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        num_synapses=args.synapses,
        k=args.k,
        syn_hidden=args.syn_hidden,
        num_heads=args.num_heads,
        pad_idx=sp["[PAD]"],
        max_src_len=args.src_max_len,
        max_tgt_len=args.tgt_max_len,
    )

    ckpt = torch.load(args.model_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded: {args.model_path}")
    if "step" in ckpt:
        print(f"  Trained to step={ckpt['step']}, best_bleu={ckpt.get('best_bleu', 'N/A'):.4f}")

    EOS = sp["[EOS]"]
    PAD = sp["[PAD]"]

    def translate(src_lang, tgt_lang, src_text):
        src_tag = [sp[f"[{src_lang.upper()}]"]]
        src_ids = src_tag + tokenizer.encode(src_text)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
        bos_id = sp[f"[TARGET_{tgt_lang.upper()}]"]

        if args.beam_width > 1:
            gen_ids = beam_decode(
                model, src_tensor, bos_id=bos_id, eos_id=EOS, pad_id=PAD,
                beam_width=args.beam_width, max_len=args.tgt_max_len,
            )
        else:
            gen = model.greedy_decode(src_tensor, bos_id=bos_id, eos_id=EOS,
                                      max_len=args.tgt_max_len)
            gen_ids = gen[0].tolist()

        # BOS / EOS 除去
        gen_ids = [i for i in gen_ids if i not in (bos_id, EOS, PAD)]
        try:
            return tokenizer.decode(gen_ids)
        except Exception:
            return "<decode error>"

    print("\n" + "=" * 70)
    print("  翻訳精度テスト")
    print("=" * 70)

    direction_bleus: dict = {}
    results = []

    for src_lang, tgt_lang, src_text, ref_text in TEST_PAIRS:
        pred = translate(src_lang, tgt_lang, src_text)
        bleu = bleu_score(pred, ref_text)
        direction = f"{src_lang.upper()}→{tgt_lang.upper()}"
        direction_bleus.setdefault(direction, []).append(bleu)
        results.append((direction, src_text, ref_text, pred, bleu))

        mark = "✅" if bleu > 0.3 else ("△" if bleu > 0.05 else "❌")
        print(f"\n[{direction}] {mark} BLEU={bleu:.3f}")
        print(f"  入力 : {src_text}")
        print(f"  正解 : {ref_text}")
        print(f"  生成 : {pred[:80]}")

    print("\n" + "=" * 70)
    print("  方向別 平均 BLEU")
    print("=" * 70)
    all_bleus = []
    for direction, bleus in sorted(direction_bleus.items()):
        avg = np.mean(bleus)
        all_bleus.extend(bleus)
        bar = "█" * int(avg * 30)
        print(f"  {direction:10s}  avg={avg:.3f}  {bar}")

    overall = np.mean(all_bleus)
    print(f"\n  全体平均 BLEU: {overall:.3f}")
    print(f"  BLEU>0.3  (高品質): {sum(1 for r in results if r[4] > 0.3)}/{len(results)}")
    print(f"  BLEU>0.1  (部分一致): {sum(1 for r in results if r[4] > 0.1)}/{len(results)}")
    print("=" * 70)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path",   type=str,   default="../../../sra_seq2seq_best.pt")
    p.add_argument("--dim",          type=int,   default=256)
    p.add_argument("--enc-layers",   type=int,   default=3)
    p.add_argument("--dec-layers",   type=int,   default=3)
    p.add_argument("--synapses",     type=int,   default=12)
    p.add_argument("--k",            type=int,   default=3)
    p.add_argument("--syn-hidden",   type=int,   default=512)
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--src-max-len",  type=int,   default=80)
    p.add_argument("--tgt-max-len",  type=int,   default=80)
    p.add_argument("--beam-width",   type=int,   default=4)
    run_inference(p.parse_args())
