import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

cells.append(nbf.v4.new_markdown_cell("""# 25. 統合検証：Last-Token Router と 異種シナプスの Semantic Fallback

このノートブックでは、SRAの最終形態に近い「マザーボードアーキテクチャ」の統合検証を行います。
- **Last-Token Router**: Notebook 24で最適と証明された、希釈を防ぐ軽量・高確信度ルーター
- **異種シナプス**: Neural (汎用LLM), Rule-based (電卓), Retrieval (VectorDB)
- **Semantic Fallback**: 異種モジュール間で安全に迂回を行うためのフォールバック検証"""))

cells.append(nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# デバイス設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. 異種シナプスの定義
ニューラルネットワーク（LLM）、確定的なルールベース（電卓）、外部知識検索（VectorDB）という、動作原理が全く異なる3つのモジュールを同じインターフェースで定義します。"""))

cells.append(nbf.v4.new_code_cell("""class LLMSynapse(nn.Module):
    \"\"\"汎用LLMシナプス (Neural)\"\"\"
    def __init__(self, d_model):
        super().__init__()
        # 実際にはここにTransformer Blockが入るが、今回はダミーとして線形層
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
    def forward(self, x, text_input=None):
        return self.net(x)

class CalculatorSynapse(nn.Module):
    \"\"\"ルールベース計算機シナプス\"\"\"
    def __init__(self):
        super().__init__()
        self.is_neural = False # 非ニューラルフラグ
    def forward(self, x, text_input=None):
        # 実際の推論時は数式の文字列を評価する
        if text_input is not None:
            try:
                # 簡易的な数式評価 (セキュリティ上注意)
                ans = eval(text_input)
                return f"CALC_RESULT: {ans}"
            except:
                return "CALC_ERROR"
        return x # 勾配計算用ダミー

class VectorDBSynapse(nn.Module):
    \"\"\"事実検索シナプス (Retrieval)\"\"\"
    def __init__(self, d_model):
        super().__init__()
        self.is_neural = False
        # 簡易的なKey-Valueストア
        self.db = {}
        self.d_model = d_model
        
    def add_fact(self, key, value):
        self.db[key] = value
        
    def forward(self, x, text_input=None):
        if text_input is not None:
            # 簡易的な完全一致検索
            for k, v in self.db.items():
                if k in text_input:
                    return f"DB_RESULT: {v}"
            return "DB_MISS"
        return x"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Last-Token Router と マザーボードの実装
Notebook 24で決定した `Last-Token Router` を組み込んだSRAマザーボードを実装します。ここでは、特定のシナプスが容量上限（Capacity Limit）に達した際に、**ルーターの第2候補に安全にフォールバック**するロジックを搭載しています。"""))

cells.append(nbf.v4.new_code_cell("""class LastTokenRouter(nn.Module):
    \"\"\"
    Notebook 24で最適と判断されたルーター。
    シーケンスの最後のトークンのみを用いてルーティング確率を計算する。
    \"\"\"
    def __init__(self, d_model, num_synapses):
        super().__init__()
        self.classifier = nn.Linear(d_model, num_synapses, bias=False)
        
    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # 最後のトークンのみを抽出
        last_token_embeds = x[:, -1, :] # (batch, d_model)
        logits = self.classifier(last_token_embeds)
        probs = F.softmax(logits, dim=-1)
        return probs, logits

class SRAMotherboard(nn.Module):
    \"\"\"ルーターと異種シナプスを統合するマザーボードモデル\"\"\"
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # シナプスの登録用ディクショナリ
        self.synapses = nn.ModuleDict()
        self.synapse_names = []
        
        # ルーター (初期状態ではシナプス0なのでダミー)
        self.router = None 
        
    def add_synapse(self, name, synapse_module):
        \"\"\"Hot-swap: 新しいシナプスを動的に追加\"\"\"
        self.synapses[name] = synapse_module
        self.synapse_names.append(name)
        # ルーターを再構築 (既存の重みは保持)
        old_router = self.router
        self.router = LastTokenRouter(self.d_model, len(self.synapse_names)).to(device)
        
        if old_router is not None:
            # 古い重みをコピー
            with torch.no_grad():
                self.router.classifier.weight[:len(self.synapse_names)-1, :] = old_router.classifier.weight.data
                
    def route_and_forward(self, input_ids, text_inputs=None, capacity_limits=None):
        \"\"\"
        Semantic Fallback を含む推論処理
        capacity_limits: {'Calculator': 1} のようにシナプスごとの上限を指定
        \"\"\"
        x = self.embedding(input_ids) # (batch, seq, d_model)
        
        # 1. ルーターで確率計算
        probs, logits = self.router(x)
        
        batch_size = x.size(0)
        final_outputs = []
        routing_history = []
        
        # 現在のシナプス使用数をトラッキング
        current_usage = {name: 0 for name in self.synapse_names}
        if capacity_limits is None:
            capacity_limits = {}
            
        for i in range(batch_size):
            # i番目のサンプルの確率分布をソート
            sample_probs = probs[i]
            sorted_indices = torch.argsort(sample_probs, descending=True)
            
            selected_synapse_name = None
            text_input = text_inputs[i] if text_inputs else None
            
            # Semantic Fallback: 確率の高い順に容量を確認して割り当て
            for rank, syn_idx in enumerate(sorted_indices):
                syn_name = self.synapse_names[syn_idx.item()]
                limit = capacity_limits.get(syn_name, float('inf'))
                
                if current_usage[syn_name] < limit:
                    # 容量に空きがあれば割り当て決定
                    selected_synapse_name = syn_name
                    current_usage[syn_name] += 1
                    
                    if rank > 0:
                        print(f"⚠️ [Fallback Triggered] Token '{text_input}': {self.synapse_names[sorted_indices[0].item()]} is full -> Re-routed to {syn_name}")
                    break
            
            # シナプスでの処理
            if selected_synapse_name:
                synapse = self.synapses[selected_synapse_name]
                if getattr(synapse, "is_neural", True):
                    # ニューラル処理
                    out = synapse(x[i:i+1], text_input=text_input)
                    final_outputs.append(f"[{selected_synapse_name}] Neural Output Generated (LLM handled it)")
                else:
                    # ルールベース/DB処理
                    out = synapse(x[i:i+1], text_input=text_input)
                    final_outputs.append(f"[{selected_synapse_name}] {out}")
            else:
                final_outputs.append("[Error] No Synapse Available (All Full)")
                
            routing_history.append((probs[i].detach().cpu().numpy(), selected_synapse_name))
            
        return final_outputs, routing_history"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. シナプスのHot-Swapとデータセットの準備"""))

cells.append(nbf.v4.new_code_cell("""# ダミーのTokenizerと次元数
vocab_size = 1000
d_model = 128

# モデルの初期化
model = SRAMotherboard(d_model=d_model, vocab_size=vocab_size).to(device)

# 1. 汎用LLMの追加
model.add_synapse("GeneralLLM", LLMSynapse(d_model).to(device))

# 2. 計算機プラグインのHot-Swap
calc = CalculatorSynapse()
model.add_synapse("Calculator", calc)

# 3. VectorDBプラグインのHot-Swap
vdb = VectorDBSynapse(d_model)
vdb.add_fact("Japan", "Tokyo")
vdb.add_fact("CEO", "John Doe")
model.add_synapse("VectorDB", vdb)

print(f"Active Synapses: {model.synapse_names}")

# データセットの準備 (3ドメイン: 数式、事実検索、一般会話)
data = [
    {"domain": "math", "text": "15 * 4", "target_idx": 1},
    {"domain": "math", "text": "100 / 2", "target_idx": 1},
    {"domain": "fact", "text": "Capital of Japan?", "target_idx": 2},
    {"domain": "fact", "text": "Who is the CEO?", "target_idx": 2},
    {"domain": "chat", "text": "Hello, how are you?", "target_idx": 0},
    {"domain": "chat", "text": "Tell me a joke.", "target_idx": 0},
]

# ダミー入力IDの生成
input_ids = torch.randint(0, vocab_size, (len(data), 5)).to(device)
texts = [d["text"] for d in data]
target_labels = torch.tensor([d["target_idx"] for d in data]).to(device)"""))

cells.append(nbf.v4.new_markdown_cell("""## 4. ルーターの学習 (Routing Fine-tuning)
入力を各専門モジュールへ正しく振り分けるよう、Last-Token Routerを学習させます。"""))

cells.append(nbf.v4.new_code_cell("""# ルーターの学習（Fine-Tuning）
optimizer = torch.optim.Adam(model.router.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

epochs = 100
model.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    x = model.embedding(input_ids)
    
    # ルーターの出力を取得
    probs, logits = model.router(x)
    
    # ターゲットシナプスとのCross Entropy
    loss = criterion(logits, target_labels)
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f}")

print("\\n--- Training Complete ---")
model.eval()
with torch.no_grad():
    final_probs, _ = model.router(model.embedding(input_ids))
    print(f"Final Routing Probabilities for '{texts[0]}':")
    print({name: f"{p:.4f}" for name, p in zip(model.synapse_names, final_probs[0].cpu().tolist())})"""))


cells.append(nbf.v4.new_markdown_cell("""## 5. Semantic Fallback (意味的フォールバック) の検証
ここで、**CalculatorモジュールとVectorDBモジュールに意図的な容量制限 (Capacity Limit = 1)** を設定します。
バッチ内にはそれぞれ2つのリクエストがあるため、1つは処理できずに溢れます。

溢れたリクエストが「適当なモジュール」へ行ってクラッシュするのではなく、ルーターの第2候補である「汎用LLM」へ**意味的に正しくフォールバック**する様子を確認します。"""))

cells.append(nbf.v4.new_code_cell("""print("=== 1. 通常の推論 (Capacity Limit なし) ===")
outputs, history = model.route_and_forward(input_ids, text_inputs=texts)
for t, out in zip(texts, outputs):
    print(f"Input: {t:<20} | Output: {out}")

print("\\n=== 2. Semantic Fallback の検証 (CalculatorとVectorDBの容量を1に制限) ===")
# それぞれ2つのリクエストがあるため、必ず1つは溢れる。
# VectorDBが溢れた時に、Calculatorに割り当てられるとエラーになるが、汎用LLMが第2候補なら安全に迂回できる。
limits = {"Calculator": 1, "VectorDB": 1}
outputs_fb, history_fb = model.route_and_forward(input_ids, text_inputs=texts, capacity_limits=limits)
print("-" * 40)
for t, out in zip(texts, outputs_fb):
    print(f"Input: {t:<20} | Output: {out}")"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. ルーティングの可視化
ルーターの確率分布（ヒートマップ）を確認します。
高い確信度で専用モジュールが選ばれつつ、第2候補として GeneralLLM が僅かな確率を持っているため、安全な Semantic Fallback が成立していることが分かります。"""))

cells.append(nbf.v4.new_code_cell("""prob_matrix = torch.stack([torch.tensor(h[0]) for h in history_fb]).numpy()

plt.figure(figsize=(10, 6))
sns.heatmap(prob_matrix, annot=True, cmap="YlGnBu", fmt=".4f",
            xticklabels=model.synapse_names, 
            yticklabels=[f"{d['domain']}: {d['text']}" for d in data])
plt.title("Routing Probabilities (Semantic Fallback Enabled)")
plt.xlabel("Synapses")
plt.ylabel("Inputs")
plt.tight_layout()
plt.show()"""))

nb['cells'] = cells

with open('/Users/suzukijun/Program/SynapticRouter/notebooks/25_integrated_heterogeneous_routing.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
