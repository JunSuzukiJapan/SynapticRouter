import argparse
import random
import time
import torch
import torch.nn.functional as F

from sra_language_models import MoESRALanguageModel
from sra_experiment import make_optimizer, load_balance_loss, usage_stats, specialization_loss, freeze_router
from sra_gridworld import make_dt_batch, generate_trajectory
from constants import VOCAB_SIZE, TOKENS, ID2TOK

@torch.no_grad()
def evaluate_dt(model, batches, batch_size, max_steps, device):
    model.eval()
    results = {}
    
    for task_type in ["treasure", "escape"]:
        total_loss = 0.0
        total_tok = 0
        task_usage = None
        
        for _ in range(batches):
            x, y, tasks = make_dt_batch(batch_size, max_steps, device)
            # Filter for the current task_type
            idx = [i for i, t in enumerate(tasks) if t == task_type]
            if not idx:
                continue
            x_task = x[idx]
            y_task = y[idx]
            
            logits, router_logits = model(x_task)
            
            # Action prediction accuracy isn't trivial to compute since actions are mixed with states
            # Let's just compute the LM loss for the trajectory
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_task.view(-1), ignore_index=-100, reduction="sum")
            
            valid_toks = (y_task != -100).sum().item()
            total_loss += loss.item()
            total_tok += valid_toks
            
            if router_logits:
                u = usage_stats(router_logits)
                task_usage = u if task_usage is None else task_usage + u
                
        results[task_type] = {
            "val_loss": total_loss / max(total_tok, 1),
            "usage": (task_usage / batches) if task_usage is not None else None
        }
    return results

def train(args):
    start_time = time.time()
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    model = MoESRALanguageModel(
        vocab_size=VOCAB_SIZE,
        dim=args.dim,
        layers=args.layers,
        num_synapses=args.synapses,
        k=args.k,
        syn_hidden=args.syn_hidden,
        pad_idx=0,
        max_seq_len=200 # More than enough for 10 steps of 4 states + 1 action + 1 reward
    ).to(device)

    opt = make_optimizer(model, args.lr)
    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps
    
    print(f"--- Decision Transformer Training ---")
    print(f"Device: {device}, Total Steps: {args.steps}")
    
    for step in range(1, args.steps + 1):
        if step <= args.warmup_steps:
            phase = "warmup"
        elif step <= phase1_end:
            phase = "joint"
        elif step <= phase2_end:
            phase = "stabilize"
        else:
            phase = "specialize"

        model.train()
        dense = step <= args.warmup_steps
        if step == phase1_end + 1:
            freeze_router(model)
            opt = make_optimizer(model, args.lr)
            print(f"Phase transition: stabilization after step {phase1_end}")

        x, y, batch_tasks = make_dt_batch(args.batch_size, args.max_steps, device)
        logits, router_logits = model(x, dense=dense)
        
        ce = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1), ignore_index=-100)
        lb = load_balance_loss(router_logits)
        loss = ce + args.load_balance * lb
        
        if phase == "specialize":
            spec_loss = specialization_loss(router_logits)
            loss = loss + args.specialization_weight * spec_loss
        else:
            spec_loss = torch.tensor(0.0, device=device)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            eval_results = evaluate_dt(model, 2, args.batch_size, args.max_steps, device)
            
            log_str = f"step={step:4d} phase={phase} loss={loss.item():.4f} ce={ce.item():.4f} lb={lb.item():.4f}"
            if phase == "specialize":
                log_str += f" spec={spec_loss.item():.4f}"
                
            print(log_str)
            for t, res in eval_results.items():
                top_usage = ", ".join(f"{v:.2f}" for v in res["usage"].tolist()[:min(8, len(res["usage"]))]) if res["usage"] is not None else ""
                print(f"  [{t}] val_loss={res['val_loss']:.4f} usage[:8]=[{top_usage}]")
            
    torch.save(model.state_dict(), args.save)
    print(f"Saved: {args.save}")
    
    print(f"\n--- Inference Examples ---")
    model.eval()
    
    def decode(ids):
        return " ".join(ID2TOK.get(i, "?") for i in ids if i != 0)
        
    for task_type in ["treasure", "escape"]:
        # Get a real trajectory
        traj = generate_trajectory(task_type, max_steps=args.max_steps)
        # Give the model the task token, initial reward, and first state
        prompt = traj[:6]
        
        print(f"\n[{task_type.upper()}] Prompt: {decode(prompt)}")
        
        # Autoregressively generate the rest
        current_prompt = prompt.copy()
        for _ in range(30): # Generate up to 30 tokens
            x_t = torch.tensor([current_prompt], dtype=torch.long, device=device)
            logits, _ = model(x_t)
            probs = torch.softmax(logits[0, -1] / 0.1, dim=-1) # low temperature for argmax-like behavior
            next_token = torch.multinomial(probs, 1).item()
            current_prompt.append(next_token)
            
            if ID2TOK.get(next_token) in ["<R_POS>", "<R_NEG>"]:
                # The environment terminated
                break
                
        print(f"Generated: {decode(current_prompt[len(prompt):])}")
        print(f"Full Traj: {decode(current_prompt)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--syn-hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--load-balance", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--joint-steps", type=int, default=1000)
    p.add_argument("--stabilize-steps", type=int, default=200)
    p.add_argument("--specialization-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="sra_dt_model.pt")
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())
