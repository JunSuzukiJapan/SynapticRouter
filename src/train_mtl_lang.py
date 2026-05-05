import argparse
import random
import time
import torch
import torch.nn.functional as F
import tiktoken

from sra_language_models import MoESRALanguageModel
from sra_experiment import make_optimizer, load_balance_loss, usage_stats, specialization_loss, freeze_router

def load_data(tokenizer):
    domains = ["code", "math", "text"]
    data = {}
    for d in domains:
        path = f"data/lang/{d}.txt"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        # Add repeated text to ensure we have enough data for sequences
        text = (text + "\n") * 10 
        tokens = tokenizer.encode(text, allowed_special="all")
        data[d] = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {d}: {len(tokens)} tokens")
    return data, domains

def get_batch(data, domains, batch_size, seq_len, device):
    x = torch.zeros((batch_size, seq_len), dtype=torch.long)
    y = torch.zeros((batch_size, seq_len), dtype=torch.long)
    batch_domains = []
    
    for i in range(batch_size):
        d = random.choice(domains)
        batch_domains.append(d)
        d_data = data[d]
        max_start = len(d_data) - seq_len - 1
        if max_start > 0:
            start = random.randint(0, max_start)
        else:
            start = 0
            
        chunk = d_data[start:start+seq_len+1]
        
        if len(chunk) < seq_len + 1:
            pad_len = seq_len + 1 - len(chunk)
            chunk = torch.cat([chunk, torch.zeros(pad_len, dtype=torch.long)])
            
        x[i] = chunk[:seq_len]
        y[i] = chunk[1:seq_len+1]
        
    return x.to(device), y.to(device), batch_domains

@torch.no_grad()
def evaluate(model, data, domains, batches, batch_size, seq_len, device):
    model.eval()
    results = {}
    for d in domains:
        total_loss = 0.0
        total_tok = 0
        task_usage = None
        for _ in range(batches):
            x = torch.zeros((batch_size, seq_len), dtype=torch.long)
            y = torch.zeros((batch_size, seq_len), dtype=torch.long)
            d_data = data[d]
            for i in range(batch_size):
                max_start = len(d_data) - seq_len - 1
                start = random.randint(0, max(0, max_start))
                chunk = d_data[start:start+seq_len+1]
                if len(chunk) < seq_len + 1:
                    pad_len = seq_len + 1 - len(chunk)
                    chunk = torch.cat([chunk, torch.zeros(pad_len, dtype=torch.long)])
                x[i] = chunk[:seq_len]
                y[i] = chunk[1:seq_len+1]
            
            x, y = x.to(device), y.to(device)
            logits, router_logits = model(x)
            
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction="sum")
            total_loss += loss.item()
            total_tok += y.numel()
            
            if router_logits:
                u = usage_stats(router_logits)
                task_usage = u if task_usage is None else task_usage + u
                
        results[d] = {
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
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab
    
    data, domains = load_data(tokenizer)

    model = MoESRALanguageModel(
        vocab_size=vocab_size,
        dim=args.dim,
        layers=args.layers,
        num_synapses=args.synapses,
        k=args.k,
        syn_hidden=args.syn_hidden,
        max_seq_len=args.seq_len
    ).to(device)

    opt = make_optimizer(model, args.lr)
    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps
    
    print(f"--- Multi-Domain Language Training ---")
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

        x, y, batch_domains = get_batch(data, domains, args.batch_size, args.seq_len, device)
        logits, router_logits = model(x, dense=dense)
        
        ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
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
            eval_results = evaluate(model, data, domains, 5, args.batch_size, args.seq_len, device)
            
            log_str = f"step={step:4d} phase={phase} loss={loss.item():.4f} ce={ce.item():.4f} lb={lb.item():.4f}"
            if phase == "specialize":
                log_str += f" spec={spec_loss.item():.4f}"
                
            print(log_str)
            for d in domains:
                res = eval_results[d]
                top_usage = ", ".join(f"{v:.2f}" for v in res["usage"].tolist()[:min(8, len(res["usage"]))]) if res["usage"] is not None else ""
                print(f"  [{d}] val_loss={res['val_loss']:.4f} usage[:8]=[{top_usage}]")
            
    torch.save(model.state_dict(), args.save)
    print(f"Saved: {args.save}")
    
    # Generate some samples
    print(f"\n--- Inference Examples ---")
    model.eval()
    for d in domains:
        x, _, _ = get_batch(data, [d], 1, args.seq_len // 2, device) # prompt with half length
        prompt = x[0].tolist()
        print(f"\n[{d.upper()}] Prompt: {tokenizer.decode(prompt)}")
        
        # simple greedy generation
        for _ in range(20):
            logits, _ = model(torch.tensor([prompt], device=device))
            probs = torch.softmax(logits[0, -1] / 0.8, dim=-1); next_token = torch.multinomial(probs, 1).item()
            prompt.append(next_token)
            
        print(f"Generated: {tokenizer.decode(prompt)}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--syn-hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--load-balance", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--joint-steps", type=int, default=600)
    p.add_argument("--stabilize-steps", type=int, default=100)
    p.add_argument("--specialization-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="sra_lang_model.pt")
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())
