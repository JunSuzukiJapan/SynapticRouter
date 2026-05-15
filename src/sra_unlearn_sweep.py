import argparse
import os
import sys
import random
import copy
import collections
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F

# 依存パスの追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sra_gpu_models import MoESRAModel
from src.sra_experiment import make_optimizer, load_balance_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

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

# モデル設定
MAX_SEQ_LEN = 24
DIM = 64
LAYERS = 2
NUM_SYNAPSES = 32
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

def train_model(epochs=1500):
    print(f"Training new model for {epochs} epochs...")
    model = MoESRAModel(vocab_size=128, dim=DIM, layers=LAYERS, num_synapses=NUM_SYNAPSES, k=K, syn_hidden=SYN_HIDDEN).to(device)
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
        if (epoch + 1) % 300 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")
    return model

def get_task_routing_vector(model, task_fn, samples=50):
    model.eval()
    counts = torch.zeros(NUM_SYNAPSES, device=device)
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

def run_unlearn_experiment(model, task_vectors, target_domain, threshold):
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
    
    # モデルのコピー作成とアンラーニング
    model_copy = copy.deepcopy(model)
    if len(safe_to_delete) > 0:
        delete_synapses(model_copy, safe_to_delete)
        
    # 精度計測
    target_acc = evaluate_domain(model_copy, target_tasks)
    other_acc = evaluate_domain(model_copy, other_tasks)
    
    return {
        "target_domain": target_domain,
        "threshold": threshold,
        "deleted_synapses": len(safe_to_delete),
        "target_acc": target_acc,
        "other_acc": other_acc
    }

def main():
    parser = argparse.ArgumentParser(description="Automated Unlearning Sweep Tool for SRA")
    parser.add_argument("--model-path", type=str, default="", help="Path to trained model (if empty, trains a new one)")
    parser.add_argument("--target-domain", type=str, default="all", help="Target domain to unlearn (e.g., 'dna', 'math') or 'all'")
    parser.add_argument("--threshold", type=float, default=0.05, help="Threshold for protecting shared synapses")
    parser.add_argument("--sweep", action="store_true", help="Automatically sweep across multiple thresholds")
    args = parser.parse_args()

    # モデルの準備
    if args.model_path and os.path.exists(args.model_path):
        model = MoESRAModel(vocab_size=128, dim=DIM, layers=LAYERS, num_synapses=NUM_SYNAPSES, k=K, syn_hidden=SYN_HIDDEN).to(device)
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded model from {args.model_path}")
    else:
        model = train_model(epochs=1500)
        
    print("\nCalculating Task Routing Vectors...")
    task_vectors = {name: get_task_routing_vector(model, fn) for name, fn in ALL_TASKS.items()}
    
    # スイープ対象の設定
    domains_to_test = DOMAINS if args.target_domain == "all" else [args.target_domain]
    thresholds_to_test = [0.0, 0.01, 0.05, 0.10, 0.15, 0.20] if args.sweep else [args.threshold]
    
    print("\n" + "="*80)
    print(f"{'Domain':<10} | {'Threshold':<10} | {'Deleted Sy.':<12} | {'Target Acc':<12} | {'Other Acc':<12}")
    print("-" * 80)
    
    results = []
    
    for domain in domains_to_test:
        # ベースライン精度計測
        target_tasks = [t for t in ALL_TASKS if t.startswith(f"{domain}_")]
        other_tasks = [t for t in ALL_TASKS if t not in target_tasks]
        base_target_acc = evaluate_domain(model, target_tasks)
        base_other_acc = evaluate_domain(model, other_tasks)
        
        print(f"{domain:<10} | {'Baseline':<10} | {'0':<12} | {base_target_acc*100:>5.1f}%      | {base_other_acc*100:>5.1f}%")
        
        for thresh in thresholds_to_test:
            res = run_unlearn_experiment(model, task_vectors, domain, thresh)
            results.append(res)
            
            # 変化を色付けしたいがCLIなのでシンプルに
            t_acc = res['target_acc']*100
            o_acc = res['other_acc']*100
            print(f"{domain:<10} | {thresh:<10.3f} | {res['deleted_synapses']:<12} | {t_acc:>5.1f}%      | {o_acc:>5.1f}%")
            
        print("-" * 80)

if __name__ == "__main__":
    main()
