"""
翻訳精度分析スクリプト
- 学習済みモデルで自己回帰生成して翻訳文を出力
- 日英仏の全方向ペアをテスト
- 簡易BLEUおよびトークン一致率を計算
"""
import torch
import torch.nn.functional as F
import tiktoken
import numpy as np
import sys
import os
import re

sys.path.insert(0, os.path.dirname(__file__))
from sra_language_models import MoESRALanguageModel

# ============================================================
# 設定
# ============================================================
DEVICE = "cpu"  # 推論はCPUで十分（MPS対応でも可）
SEQ_LEN = 64
DIM = 128
LAYERS = 2
NUM_SYNAPSES = 8
K = 2
SYN_HIDDEN = 256
MAX_NEW_TOKENS = 40

LANG_TAGS = {"en": "[ENG]", "ja": "[JPN]", "fr": "[FRA]"}
TARGET_TAGS = {"en": "[TARGET_ENG]", "ja": "[TARGET_JPN]", "fr": "[TARGET_FRA]"}

# ============================================================
# テストセット（正解付き）
# ============================================================
TEST_PAIRS = [
    # EN → JA
    ("en", "ja", "I eat apples.", "私はりんごを食べる。"),
    ("en", "ja", "She reads books.", "彼女は本を読む。"),
    ("en", "ja", "Good morning.", "おはようございます。"),
    ("en", "ja", "Thank you very much.", "ありがとうございました。"),
    # EN → FR
    ("en", "fr", "I eat apples.", "Je mange des pommes."),
    ("en", "fr", "She reads books.", "Elle lit des livres."),
    ("en", "fr", "Good morning.", "Bonjour."),
    ("en", "fr", "Thank you very much.", "Merci beaucoup."),
    # JA → EN
    ("ja", "en", "私 は りんご を 食べる。", "I eat apples."),
    ("ja", "en", "彼女 は 本 を 読む。", "She reads books."),
    ("ja", "en", "おはようございます。", "Good morning."),
    # FR → EN
    ("fr", "en", "Je mange des pommes.", "I eat apples."),
    ("fr", "en", "Elle lit des livres.", "She reads books."),
    ("fr", "en", "Bonjour.", "Good morning."),
    # JA → FR
    ("ja", "fr", "私 は りんご を 食べる。", "Je mange des pommes."),
    # FR → JA
    ("fr", "ja", "Je mange des pommes.", "私はりんごを食べる。"),
]

# ============================================================
# モデルロード
# ============================================================
tokenizer = tiktoken.get_encoding("cl100k_base")
vocab_size = tokenizer.n_vocab + 100

model = MoESRALanguageModel(
    vocab_size=vocab_size, dim=DIM, layers=LAYERS,
    num_synapses=NUM_SYNAPSES, k=K, syn_hidden=SYN_HIDDEN,
    pad_idx=0, max_seq_len=SEQ_LEN
).to(DEVICE)

ckpt_path = os.path.join(os.path.dirname(__file__), "..", "sra_translation_local.pt")
ckpt = torch.load(ckpt_path, map_location=DEVICE)
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)
model.eval()
print("Model loaded OK\n")

# ============================================================
# 自己回帰生成（推論）
# ============================================================
def translate(src_lang: str, tgt_lang: str, src_text: str) -> str:
    prompt_str = f"{LANG_TAGS[src_lang]} {src_text} [SEP] {TARGET_TAGS[tgt_lang]} "
    prompt_tokens = tokenizer.encode(prompt_str, allowed_special="all")
    prompt_tokens = prompt_tokens[:SEQ_LEN]

    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=DEVICE)
    generated = list(prompt_tokens)

    with torch.no_grad():
        for _ in range(MAX_NEW_TOKENS):
            cur = torch.tensor([generated[-SEQ_LEN:]], dtype=torch.long, device=DEVICE)
            logits, _ = model(cur, dense=False)
            next_token_logits = logits[0, -1, :]  # (vocab_size,)
            next_token = next_token_logits.argmax().item()
            generated.append(next_token)

            # 簡易EOS判定（tiktoken はEOSがないので改行 or [EOS]文字列でストップ）
            try:
                partial = tokenizer.decode(generated[len(prompt_tokens):])
                if "[EOS]" in partial or "\n\n" in partial:
                    break
            except Exception:
                break

    # デコード
    output_tokens = generated[len(prompt_tokens):]
    try:
        text = tokenizer.decode(output_tokens)
        # [EOS] 以降を切り捨て
        text = text.split("[EOS]")[0].strip()
    except Exception:
        text = "<decode error>"
    return text

# ============================================================
# 簡易BLEU (n=1,2 のみ)
# ============================================================
def ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def bleu_score(hypothesis: str, reference: str, max_n=2) -> float:
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = reference.lower().split()
    if not hyp_tokens:
        return 0.0
    scores = []
    for n in range(1, max_n+1):
        hyp_ng = ngrams(hyp_tokens, n)
        ref_ng = ngrams(ref_tokens, n)
        if not hyp_ng:
            scores.append(0.0)
            continue
        ref_set = {}
        for g in ref_ng:
            ref_set[g] = ref_set.get(g, 0) + 1
        match = 0
        for g in hyp_ng:
            if ref_set.get(g, 0) > 0:
                match += 1
                ref_set[g] -= 1
        scores.append(match / len(hyp_ng))
    # Brevity penalty
    bp = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
    bleu = bp * np.exp(np.mean([np.log(s + 1e-9) for s in scores]))
    return bleu

# ============================================================
# 評価実行
# ============================================================
print("=" * 70)
print("  翻訳精度テスト")
print("=" * 70)

results = []
direction_stats = {}

for src_lang, tgt_lang, src_text, ref_text in TEST_PAIRS:
    pred_text = translate(src_lang, tgt_lang, src_text)
    bleu = bleu_score(pred_text, ref_text)
    direction = f"{src_lang.upper()}→{tgt_lang.upper()}"

    results.append({
        "direction": direction,
        "src": src_text,
        "ref": ref_text,
        "pred": pred_text,
        "bleu": bleu,
    })

    if direction not in direction_stats:
        direction_stats[direction] = []
    direction_stats[direction].append(bleu)

    # 表示
    match_mark = "✅" if bleu > 0.3 else ("△" if bleu > 0.05 else "❌")
    print(f"\n[{direction}] {match_mark} BLEU={bleu:.3f}")
    print(f"  入力 : {src_text}")
    print(f"  正解 : {ref_text}")
    print(f"  生成 : {pred_text[:80]}")

# ============================================================
# サマリー
# ============================================================
print("\n" + "=" * 70)
print("  方向別 平均BLEU")
print("=" * 70)
all_bleus = []
for direction, bleus in sorted(direction_stats.items()):
    avg = np.mean(bleus)
    all_bleus.extend(bleus)
    bar = "█" * int(avg * 20)
    print(f"  {direction:10s}  avg BLEU={avg:.3f}  {bar}")

overall_bleu = np.mean(all_bleus)
print(f"\n  全体平均 BLEU: {overall_bleu:.3f}")
print(f"  テストケース数: {len(results)}")
print(f"  BLEU>0.3 (高品質): {sum(1 for r in results if r['bleu'] > 0.3)}/{len(results)}")
print(f"  BLEU>0.05 (部分一致): {sum(1 for r in results if r['bleu'] > 0.05)}/{len(results)}")
print("=" * 70)
