import argparse
import random
import time
import torch
import torch.nn.functional as F

from constants import PAD, BOS, EOS, VOCAB_SIZE, TASK_ORDER
from sra_experiment import make_multitask_batch, make_optimizer, specialization_loss, freeze_router, usage_stats, load_balance_loss, decode, generate_prediction, usage_entropy, synapse_stats
from sra_gpu_models import MoESRAModel, BatchedSRAModel, SeqSRAModel
from sra_reference import SRAModel

@torch.no_grad()
def evaluate_multitask(model, tasks, batches, batch_size, min_len, max_len, device):
    model.eval()
    results = {}
    
    for task in tasks:
        total_loss, total_tok, correct_seq, total_seq = 0.0, 0, 0, 0
        task_usage = None
        for _ in range(batches):
            x, y, _ = make_multitask_batch([task], batch_size, min_len, max_len, device)
            y_in = torch.cat([torch.full((y.size(0), 1), BOS, dtype=torch.long, device=device), y[:, :-1]], dim=1)
            logits, router_logits, _ = model(x, y_in)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD, reduction="sum")
            pred = logits.argmax(dim=-1)
            mask = y != PAD
            seq_ok = ((pred == y) | ~mask).all(dim=1)
            correct_seq += seq_ok.sum().item()
            total_seq += y.size(0)
            total_loss += loss.item()
            total_tok += mask.sum().item()
            
            if router_logits:
                u = usage_stats(router_logits)
                task_usage = u if task_usage is None else task_usage + u
                
        results[task] = {
            "val_loss": total_loss / max(total_tok, 1),
            "seq_acc": correct_seq / max(total_seq, 1),
            "usage": (task_usage / batches) if task_usage is not None else None
        }
    return results

def train_multitask(args):
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
    
    if args.model_type == "sra":
        model = SRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)
    elif args.model_type == "batched_sra":
        model = BatchedSRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)
    elif args.model_type == "moe_sra":
        model = MoESRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)
    elif args.model_type == "seq_sra":
        model = SeqSRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    opt = make_optimizer(model, args.lr)
    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps
    
    tasks = args.tasks.split(",") if args.tasks else TASK_ORDER
    print(f"--- Multitask Training ---")
    print(f"Tasks: {tasks}")
    print(f"Device: {device}, Model: {args.model_type}, Total Steps: {args.steps}")
    
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

        x, y, batch_tasks = make_multitask_batch(tasks, args.batch_size, args.min_len, args.max_len, device)
        y_in = torch.cat([torch.full((y.size(0), 1), BOS, dtype=torch.long, device=device), y[:, :-1]], dim=1)
        logits, router_logits, all_syn_outputs = model(x, y_in, dense=dense)
        
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD)
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
            eval_results = evaluate_multitask(model, tasks, 10, args.batch_size, args.min_len, args.max_len, device)
            
            log_str = f"step={step:5d} phase={phase} loss={loss.item():.4f} ce={ce.item():.4f} lb={lb.item():.4f}"
            if phase == "specialize":
                log_str += f" spec={spec_loss.item():.4f}"
                
            print(log_str)
            for t in tasks:
                res = eval_results[t]
                top_usage = ", ".join(f"{v:.2f}" for v in res["usage"].tolist()[:min(8, len(res["usage"]))]) if res["usage"] is not None else ""
                print(f"  [{t}] val_loss={res['val_loss']:.4f} seq_acc={res['seq_acc']:.3f} usage[:8]=[{top_usage}]")
            
    torch.save(model.state_dict(), args.save)
    print(f"Saved: {args.save}")

    # quick inference examples
    print(f"\n--- Inference Examples ---")
    model.eval()
    for task in tasks:
        x, y, _ = make_multitask_batch([task], 1, args.min_len, args.max_len, device)
        y_in = torch.full((1, 1), BOS, dtype=torch.long, device=device)
        outs = []
        for _t in range(y.size(1) + 2):
            logits, _, _ = model(x, y_in)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            outs.append(nxt.item())
            y_in = torch.cat([y_in, nxt], dim=1)
            if nxt.item() == EOS:
                break
        print(f"[{task.upper()}]")
        print("  x=     ", decode(x[0].tolist()))
        print("  target=", decode(y[0].tolist()))
        print("  pred=  ", decode(outs))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["sra", "batched_sra", "moe_sra", "seq_sra"], default="moe_sra")
    p.add_argument("--tasks", type=str, default="copy,reverse,paren,addmod", help="Comma separated list of tasks")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-len", type=int, default=4)
    p.add_argument("--max-len", type=int, default=10)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--syn-hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--load-balance", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--joint-steps", type=int, default=3500)
    p.add_argument("--stabilize-steps", type=int, default=500)
    p.add_argument("--specialization-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="sra_mtl_model.pt")
    p.add_argument("--cpu", action="store_true")
    train_multitask(p.parse_args())
