# Synaptic Routing Architecture (SRA) minimal experiment
# Tasks: copy, reverse, balanced parentheses, mod addition
# Requires: pip install torch

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import (
    PAD, BOS, EOS, SEP, TOKENS, ID2TOK, VOCAB_SIZE,
    TASK_ORDER, TASK_DEFAULT_STEPS, encode_symbols
)


def make_sample(task: str, min_len: int, max_len: int, mod: int = 10, use_task_token: bool = False) -> Tuple[list, list]:
    n = random.randint(min_len, max_len)
    
    task_token = [TOKENS[f"<TASK_{task.upper()}>"]] if use_task_token else []
    
    if task == "copy":
        seq = [str(random.randint(0, mod - 1)) for _ in range(n)]
        return [BOS] + task_token + encode_symbols(seq) + [SEP], encode_symbols(seq) + [EOS]
    if task == "reverse":
        seq = [str(random.randint(0, mod - 1)) for _ in range(n)]
        return [BOS] + task_token + encode_symbols(seq) + [SEP], encode_symbols(list(reversed(seq))) + [EOS]
    if task == "paren":
        seq = [random.choice(["(", ")"]) for _ in range(n)]
        bal = 0
        ok = True
        for s in seq:
            bal += 1 if s == "(" else -1
            if bal < 0:
                ok = False
        ok = ok and bal == 0
        return [BOS] + task_token + encode_symbols(seq) + [SEP], [TOKENS["Y"] if ok else TOKENS["N"], EOS]
    if task == "addmod":
        a = random.randint(0, mod - 1)
        b = random.randint(0, mod - 1)
        return [BOS] + task_token + [TOKENS[str(a)], TOKENS[str(b)], SEP], [TOKENS[str((a + b) % mod)], EOS]
    raise ValueError(f"unknown task: {task}")


def make_batch(task: str, batch_size: int, min_len: int, max_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    pairs = [make_sample(task, min_len, max_len) for _ in range(batch_size)]
    max_x = max(len(x) for x, _ in pairs)
    max_y = max(len(y) for _, y in pairs)
    x = torch.full((batch_size, max_x), PAD, dtype=torch.long)
    y = torch.full((batch_size, max_y), PAD, dtype=torch.long)
    for i, (xi, yi) in enumerate(pairs):
        x[i, :len(xi)] = torch.tensor(xi)
        y[i, :len(yi)] = torch.tensor(yi)
    return x.to(device), y.to(device)


def make_multitask_batch(tasks: list, batch_size: int, min_len: int, max_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor, list]:
    batch_tasks = [random.choice(tasks) for _ in range(batch_size)]
    pairs = [make_sample(t, min_len, max_len, use_task_token=True) for t in batch_tasks]
    max_x = max(len(x) for x, _ in pairs)
    max_y = max(len(y) for _, y in pairs)
    x = torch.full((batch_size, max_x), PAD, dtype=torch.long)
    y = torch.full((batch_size, max_y), PAD, dtype=torch.long)
    for i, (xi, yi) in enumerate(pairs):
        x[i, :len(xi)] = torch.tensor(xi)
        y[i, :len(yi)] = torch.tensor(yi)
    return x.to(device), y.to(device), batch_tasks

from sra_reference import (
    TinySynapse, Router, SRABlock, SRAModel,
    BaselineTransformer, BaselineMLP
)


def make_optimizer(model, lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.01)


def specialization_loss(router_logits):
    """Promote specialization by maximizing entropy of synapse usage."""
    logits = router_logits[-1]
    probs = F.softmax(logits, dim=-1).mean(dim=(0, 1))
    entropy = -(probs * torch.log(probs + 1e-9)).sum()
    return -entropy  # negative for maximization in loss


def freeze_router(model):
    for block in model.blocks:
        for name, p in block.router.named_parameters():
            if 'synapse_emb' not in name:
                p.requires_grad = False


def usage_stats(router_logits):
    # returns mean usage distribution from final layer argmax for monitoring
    logits = router_logits[-1].detach()
    chosen = logits.argmax(dim=-1).flatten()
    n = logits.size(-1)
    hist = torch.bincount(chosen, minlength=n).float()
    return (hist / hist.sum()).cpu()


def usage_entropy(usage):
    p = usage.clamp(min=1e-9)
    return -(p * torch.log(p)).sum().item()


@torch.no_grad()
def synapse_stats(all_synapse_outputs):
    """Compute statistics on synapse outputs."""
    stats = []
    for layer_id, layer_outputs in enumerate(all_synapse_outputs):
        layer_stats = []
        for syn_id, syn_out in enumerate(layer_outputs):
            norm = syn_out.norm(dim=-1).mean().item()
            layer_stats.append(norm)
        stats.append(layer_stats)
    return stats


@torch.no_grad()
def generate_prediction(model, x, max_len, device):
    model.eval()
    y_in = torch.full((1, 1), BOS, dtype=torch.long, device=device)
    out = []
    for _ in range(max_len + 5):
        logits, _, _ = model(x, y_in)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        out.append(nxt.item())
        y_in = torch.cat([y_in, nxt], dim=1)
        if nxt.item() == EOS:
            break
    return out


def generate_self_conditioned_prefix(model, x, max_len, device):
    y_in = torch.full((x.size(0), 1), BOS, dtype=torch.long, device=device)
    for _ in range(max_len - 1):
        logits, _, _ = model(x, y_in)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        y_in = torch.cat([y_in, nxt], dim=1)
    return y_in


def load_balance_loss(router_logits):
    """Enforce balanced usage of synapses."""
    loss = 0.0
    for logits in router_logits:
        probs = F.softmax(logits, dim=-1).mean(dim=(0, 1))
        uniform = torch.full_like(probs, 1.0 / probs.numel())
        loss = loss + ((probs - uniform) ** 2).mean()
    return loss


@torch.no_grad()
def evaluate(model, task, batches, batch_size, min_len, max_len, device):
    model.eval()
    total_loss, total_tok, correct_seq, total_seq = 0.0, 0, 0, 0
    for _ in range(batches):
        x, y = make_batch(task, batch_size, min_len, max_len, device)
        y_in = torch.cat([torch.full((y.size(0), 1), BOS, dtype=torch.long, device=device), y[:, :-1]], dim=1)
        logits, _, _ = model(x, y_in)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD, reduction="sum")
        pred = logits.argmax(dim=-1)
        mask = y != PAD
        seq_ok = ((pred == y) | ~mask).all(dim=1)
        correct_seq += seq_ok.sum().item()
        total_seq += y.size(0)
        total_loss += loss.item()
        total_tok += mask.sum().item()
    return total_loss / max(total_tok, 1), correct_seq / max(total_seq, 1)


def train_single(args):
    start_time = time.time()
    if args.cpu:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    if getattr(args, "profile", False):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        elif device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.model_type == "sra":
        model = SRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)
    elif args.model_type == "batched_sra":
        from sra_gpu_models import BatchedSRAModel
        model = BatchedSRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)
    elif args.model_type == "moe_sra":
        from sra_gpu_models import MoESRAModel
        model = MoESRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)
    elif args.model_type == "seq_sra":
        from sra_gpu_models import SeqSRAModel
        model = SeqSRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)
    elif args.model_type == "transformer":
        model = BaselineTransformer(VOCAB_SIZE, args.dim, args.layers, getattr(args, "baseline_hidden", 256)).to(device)
    elif args.model_type == "mlp":
        model = BaselineMLP(VOCAB_SIZE, args.dim, args.layers, getattr(args, "baseline_hidden", 256)).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    opt = make_optimizer(model, args.lr)
    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps
    print(
        f"device={device} model={args.model_type} task={args.task} k={args.k} batch_size={args.batch_size} dim={args.dim} "
        f"layers={args.layers} synapses={args.synapses} baseline_hidden={getattr(args, 'baseline_hidden', 0)} "
        f"steps={args.steps} save={args.save} warmup={args.warmup_steps} joint={args.joint_steps} stabilize={args.stabilize_steps}"
    )

    final_val_loss, final_seq_acc = 0.0, 0.0
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
        if step == phase1_end + 1 and "sra" in args.model_type:
            freeze_router(model)
            opt = make_optimizer(model, args.lr)
            print(f"phase transition: stabilization after step {phase1_end}")

        x, y = make_batch(args.task, args.batch_size, args.min_len, args.max_len, device)
        y_in = torch.cat([torch.full((y.size(0), 1), BOS, dtype=torch.long, device=device), y[:, :-1]], dim=1)
        logits, router_logits, all_syn_outputs = model(x, y_in, dense=dense)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD)
        
        if "sra" in args.model_type:
            lb = load_balance_loss(router_logits)
            loss = ce + args.load_balance * lb
        else:
            lb = torch.tensor(0.0, device=device)
            loss = ce

        if args.self_gen_weight > 0 and phase != "warmup":
            y_in_self = generate_self_conditioned_prefix(model, x, y.size(1), device)
            logits_self, _, _ = model(x, y_in_self)
            ce_self = F.cross_entropy(logits_self.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD)
            loss = loss + args.self_gen_weight * ce_self
        else:
            ce_self = torch.tensor(0.0, device=device)

        if phase == "specialize" and "sra" in args.model_type:
            spec_loss = specialization_loss(router_logits)
            loss = loss + args.specialization_weight * spec_loss
        else:
            spec_loss = torch.tensor(0.0, device=device)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.log_every == 0 or step == 1:
            val_loss, seq_acc = evaluate(model, args.task, 20, args.batch_size, args.min_len, args.max_len, device)
            final_val_loss, final_seq_acc = val_loss, seq_acc
            
            if "sra" in args.model_type:
                usage = usage_stats(router_logits)
                entropy = usage_entropy(usage)
                top_usage = ", ".join(f"{v:.2f}" for v in usage.tolist()[:min(8, len(usage))])
                syn_norms = synapse_stats(all_syn_outputs)
                syn_str = " | ".join(f"L{i}:[" + ", ".join(f"{n:.3f}" for n in norms) + "]" for i, norms in enumerate(syn_norms))
                log_str = (
                    f"step={step:5d} phase={phase} train_loss={loss.item():.4f} ce={ce.item():.4f} "
                    f"lb={lb.item():.4f} val_loss={val_loss:.4f} seq_acc={seq_acc:.3f} entropy={entropy:.3f} "
                    f"usage[:8]=[{top_usage}] synapses={syn_str}"
                )
            else:
                log_str = (
                    f"step={step:5d} phase={phase} train_loss={loss.item():.4f} ce={ce.item():.4f} "
                    f"val_loss={val_loss:.4f} seq_acc={seq_acc:.3f}"
                )
            
            if args.self_gen_weight > 0 and phase != "warmup":
                log_str += f" ce_self={ce_self.item():.4f}"
            if phase == "specialize" and "sra" in args.model_type:
                log_str += f" spec={spec_loss.item():.4f}"
            print(log_str)
            sample_x, sample_y = make_sample(args.task, args.min_len, args.max_len)
            sample_x_t = torch.tensor([sample_x], dtype=torch.long, device=device)
            sample_pred = generate_prediction(model, sample_x_t, len(sample_y), device)
            print("sample x=", decode(sample_x), "target=", decode(sample_y), "pred=", decode(sample_pred))

    torch.save(model.state_dict(), args.save)
    print(f"saved: {args.save}")

    # quick inference examples
    model.eval()
    for _ in range(5):
        x, y = make_batch(args.task, 1, args.min_len, args.max_len, device)
        y_in = torch.full((1, 1), BOS, dtype=torch.long, device=device)
        outs = []
        for _t in range(y.size(1) + 2):
            logits, _, _ = model(x, y_in)
            nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
            outs.append(nxt.item())
            y_in = torch.cat([y_in, nxt], dim=1)
            if nxt.item() == EOS:
                break
        if not getattr(args, "disable_inference_print", False):
            print("x=", decode(x[0].tolist()), " target=", decode(y[0].tolist()), " pred=", decode(outs))

    total_time = time.time() - start_time
    if device == "cuda":
        max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    elif device == "mps" and hasattr(torch.mps, "current_allocated_memory"):
        max_mem = torch.mps.current_allocated_memory() / (1024 ** 2)
    else:
        max_mem = 0.0
    
    return {
        "val_loss": final_val_loss,
        "seq_acc": final_seq_acc,
        "total_time": total_time,
        "max_mem_mb": max_mem
    }


def decode(ids):
    return " ".join(ID2TOK.get(i, "?") for i in ids if i != PAD)


def run_task_suite(args):
    print("Running task suite:", ", ".join(TASK_ORDER))
    for task in TASK_ORDER:
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task
        task_args.steps = TASK_DEFAULT_STEPS[task]
        if args.save == "sra_model.pt":
            task_args.save = f"sra_model_{task}.pt"
        elif args.save.endswith(".pt"):
            base, ext = args.save.rsplit(".", 1)
            task_args.save = f"{base}_{task}.{ext}"
        print(f"\n=== task suite: {task} ({task_args.steps} steps) ===")
        train_single(task_args)


def train(args):
    if args.task_suite:
        run_task_suite(args)
        return
    train_single(args)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["sra", "transformer", "mlp", "batched_sra", "moe_sra", "seq_sra"], default="sra")
    p.add_argument("--profile", action="store_true", help="Enable memory/time profiling")
    p.add_argument("--baseline-hidden", type=int, default=256, help="Hidden dim for transformer/mlp baselines")
    p.add_argument("--task", choices=["copy", "reverse", "paren", "addmod"], default="reverse")
    p.add_argument("--task-suite", action="store_true", help="Run the default minimal task suite sequentially: copy, reverse, paren, addmod.")
    p.add_argument("--steps", type=int, default=2000)
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
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--joint-steps", type=int, default=1400)
    p.add_argument("--stabilize-steps", type=int, default=300)
    p.add_argument("--specialization-weight", type=float, default=0.1)
    p.add_argument("--self-gen-weight", type=float, default=0.5)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save", type=str, default="sra_model.pt")
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())
