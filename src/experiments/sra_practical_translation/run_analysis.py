"""
学習済みモデルのシナプス使用率とコサイン類似度を数値で出力する分析スクリプト (v2対応)
"""
import sys
import os
import torch
import torch.nn.functional as F
import tiktoken
import numpy as np

# src/ を sys.path に追加して共通モジュールを参照
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from sra_language_models import MoESRALanguageModel

tokenizer = tiktoken.get_encoding("cl100k_base")
vocab_size = tokenizer.n_vocab + 100

model = MoESRALanguageModel(
    vocab_size=vocab_size, dim=256, layers=4,
    num_synapses=16, k=4, syn_hidden=512,
    pad_idx=0, max_seq_len=96
)

ckpt_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "sra_translation_v2.pt")
ckpt = torch.load(ckpt_path, map_location="cpu")
state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)
model.eval()
print("Model loaded OK")

LANG_TAGS = {"en": "[ENG]", "ja": "[JPN]", "fr": "[FRA]"}
TEST_SENTENCES = {
    "en": ["I eat apples.", "She reads books.", "They play games.", "The dog chases the cat."],
    "ja": ["私 は りんご を 食べる。", "彼女 は 本 を 読む。", "彼ら は ゲーム を する。", "犬 は 猫 を 追いかける。"],
    "fr": ["Je mange des pommes.", "Elle lit des livres.", "Ils jouent aux jeux.", "Le chien chasse le chat."]
}
langs = list(TEST_SENTENCES.keys())
num_layers = 4

layer_usage = {lang: [] for lang in langs}
with torch.no_grad():
    for lang, sentences in TEST_SENTENCES.items():
        lang_usages = []
        for text in sentences:
            tokens = tokenizer.encode(f"{LANG_TAGS[lang]} {text}", allowed_special="all")
            tokens = tokens[:96]
            x = torch.tensor([tokens], dtype=torch.long)
            _, router_logits = model(x, dense=False)
            seq_usages = []
            for r in router_logits:
                probs = F.softmax(r, dim=-1).mean(dim=1).squeeze(0)
                seq_usages.append(probs.numpy())
            lang_usages.append(seq_usages)
        avg = np.array(lang_usages).mean(axis=0)  # (num_layers, num_synapses)
        layer_usage[lang] = avg

print("\n=== レイヤー別 シナプス平均活性確率 ===")
for layer_idx in range(num_layers):
    print(f"\n--- Layer {layer_idx+1} ---")
    for lang in langs:
        usage = layer_usage[lang][layer_idx]
        top3 = np.argsort(usage)[::-1][:3]
        print(f"  {lang.upper()}: top-3シナプス={top3.tolist()} 確率={usage[top3].round(4).tolist()}")
        print(f"        全8シナプス: {usage.round(4).tolist()}")

print("\n=== 言語ペア間コサイン類似度（シナプス使用プロファイル） ===")
for layer_idx in range(num_layers):
    print(f"\n--- Layer {layer_idx+1} ---")
    for i, l1 in enumerate(langs):
        for j, l2 in enumerate(langs):
            if i >= j:
                continue
            v1 = layer_usage[l1][layer_idx]
            v2 = layer_usage[l2][layer_idx]
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            print(f"  {l1.upper()} vs {l2.upper()}: cosine_similarity = {cos:.4f}")

print("\n=== エントロピー（シナプス使用の多様性） ===")
for layer_idx in range(num_layers):
    print(f"\n--- Layer {layer_idx+1} ---")
    for lang in langs:
        usage = layer_usage[lang][layer_idx]
        p = np.clip(usage, 1e-9, 1.0)
        p = p / p.sum()
        entropy = -(p * np.log(p)).sum()
        max_entropy = np.log(16)
        print(f"  {lang.upper()}: entropy={entropy:.4f} (max={max_entropy:.4f}, ratio={entropy/max_entropy:.2%})")

print("\nDone.")
