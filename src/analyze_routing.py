import torch
import torch.nn.functional as F
import numpy as np

from constants import VOCAB_SIZE, TASK_ORDER, BOS
from sra_experiment import make_multitask_batch
from sra_gpu_models import MoESRAModel

def analyze_routing(model_path="sra_mtl_model.pt", batch_size=256):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load model with correct dimensions
    model = MoESRAModel(
        vocab_size=VOCAB_SIZE,
        dim=64,
        layers=2,
        num_synapses=16,
        k=2,
        syn_hidden=128
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded {model_path} successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.eval()
    
    # 2. Collect routing data per task
    task_distributions = {}  # task_name -> [num_layers, num_synapses]
    
    with torch.no_grad():
        for task in TASK_ORDER:
            # タスクごとにバッチを生成
            x, y, _ = make_multitask_batch([task], batch_size, min_len=4, max_len=10, device=device)
            y_in = torch.cat([torch.full((y.size(0), 1), BOS, dtype=torch.long, device=device), y[:, :-1]], dim=1)
            
            # 順伝播
            logits, router_logits, _ = model(x, y_in)
            
            # router_logits is a list of tensors for each layer, shape [batch, seq_len, num_synapses]
            layer_usages = []
            for r_logits in router_logits:
                # Top-1 usage
                chosen = r_logits.argmax(dim=-1).flatten()
                n = r_logits.size(-1)
                hist = torch.bincount(chosen, minlength=n).float()
                usage = (hist / hist.sum()).cpu().numpy()
                layer_usages.append(usage)
            
            task_distributions[task] = np.array(layer_usages)
            
    # 3. Print raw distributions (top 3 experts per layer)
    print("\n--- Top Synapses per Task and Layer ---")
    num_layers = task_distributions[TASK_ORDER[0]].shape[0]
    num_synapses = task_distributions[TASK_ORDER[0]].shape[1]
    
    for task in TASK_ORDER:
        print(f"\nTask: [{task.upper()}]")
        dist = task_distributions[task]
        for l in range(num_layers):
            probs = dist[l]
            top_indices = np.argsort(probs)[::-1]
            top_strs = [f"Syn{i}({probs[i]:.2f})" for i in top_indices[:5] if probs[i] > 0]
            print(f"  Layer {l}: {', '.join(top_strs)}")

    # 4. Calculate Cosine Similarity between tasks
    print("\n--- Routing Cosine Similarity Between Tasks ---")
    
    # 各レイヤーごとに類似度を計算
    for l in range(num_layers):
        print(f"\nLayer {l}:")
        print(f"{'':>10}", end="")
        for t2 in TASK_ORDER:
            print(f"{t2:>10}", end="")
        print()
        
        for t1 in TASK_ORDER:
            print(f"{t1:>10}", end="")
            vec1 = task_distributions[t1][l]
            norm1 = np.linalg.norm(vec1)
            vec1 = vec1 / norm1 if norm1 > 0 else vec1
            for t2 in TASK_ORDER:
                vec2 = task_distributions[t2][l]
                norm2 = np.linalg.norm(vec2)
                vec2 = vec2 / norm2 if norm2 > 0 else vec2
                
                sim = np.dot(vec1, vec2)
                print(f"{sim:10.3f}", end="")
            print()

if __name__ == "__main__":
    analyze_routing()
