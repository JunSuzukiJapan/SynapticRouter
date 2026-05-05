import torch
import torch.nn.functional as F
from sra_language_models import MoESRALanguageModel
from constants import VOCAB_SIZE, TOKENS, ID2TOK
from sra_gridworld import generate_trajectory

def analyze_routing(model_path, device="cpu"):
    model = MoESRALanguageModel(
        vocab_size=VOCAB_SIZE,
        dim=128,
        layers=2,
        num_synapses=16,
        k=2,
        syn_hidden=256,
        pad_idx=0,
        max_seq_len=200
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print("=== Routing Analysis for Decision Transformer ===\n")
    
    def decode(ids):
        return " ".join(ID2TOK.get(i, "?") for i in ids if i != 0)
        
    for task_type in ["treasure", "escape"]:
        traj = generate_trajectory(task_type, max_steps=5)
        # Just use the expert's full trajectory to see how the router reacts to each token
        x = torch.tensor([traj], dtype=torch.long, device=device)
        
        # Forward pass
        logits, router_logits = model(x)
        
        # router_logits is a list of [B, T, num_synapses] for each layer
        # Let's look at the last layer
        layer_id = -1
        routing = router_logits[layer_id].argmax(dim=-1)[0] # Shape [T]
        
        print(f"Task: {task_type.upper()}")
        print(f"{'Token':<15} | {'Synapse':<10}")
        print("-" * 30)
        
        for i, token_id in enumerate(traj):
            token_str = ID2TOK.get(token_id, "?")
            syn = routing[i].item()
            print(f"{token_str:<15} | {syn}")
            
        print("\n")

if __name__ == "__main__":
    analyze_routing("sra_dt_model.pt")
