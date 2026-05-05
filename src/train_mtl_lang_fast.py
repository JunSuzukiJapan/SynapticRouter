import argparse
import random
import time
import torch
import torch.nn.functional as F

from sra_language_models import MoESRALanguageModel
from sra_experiment import make_optimizer, load_balance_loss, usage_stats, specialization_loss, freeze_router

class CharTokenizer:
    def __init__(self, texts):
        chars = set()
        for text in texts:
            chars.update(list(text))
        self.chars = sorted(list(chars))
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, tokens):
        return "".join([self.idx2char[t] for t in tokens if t in self.idx2char])

def load_data():
    domains = ["code", "math", "text"]
    raw_texts = {}
    for d in domains:
        path = f"data/lang/{d}.txt"
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        raw_texts[d] = (text + "\n") * 50 # Repeat more to ensure large sequence pool
        
    tokenizer = CharTokenizer(raw_texts.values())
    
    data = {}
    for d in domains:
        tokens = tokenizer.encode(raw_texts[d])
        data[d] = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {d}: {len(tokens)} tokens")
        
    return data, domains, tokenizer

def get_batch(data, domains, batch_size, seq_len, device):
    x = torch.zeros((batch_size, seq_len), dtype=torch.long)
    y = torch.zeros((batch_size, seq_len), dtype=torch.long)
    batch_domains = []
    
    for i in range(batch_size):
        d = random.choice(domains)
        batch_domains.append(d)
        d_data = data[d]
        max_start = len(d_data) - seq_len - 1
        start = random.randint(0, max(0, max_start))
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
            x, y, _ = get_batch(data, [d], batch_size, seq_len, device)
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
    device = "cpu"
        
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    data, domains, tokenizer = load_data()
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size (Char-level): {vocab_size}")

    model = MoESRALanguageModel(
        vocab_size=vocab_size,
        dim=args.dim,
        layers=args.layers,
        num_synapses=args.synapses,
        k=args.k,
        syn_hidden=args.syn_hidden,
        max_seq_len=args.seq_len * 2
    ).to(device)

    opt = make_optimizer(model, args.lr)
    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps
    
    print(f"--- Fast Multi-Domain Language Training ---")
    print(f"Device: {device}, Total Steps: {args.steps}")
    
    usage_history = {d: [] for d in domains}
    
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
                if res["usage"] is not None:
                    usage_history[d].append((step, res["usage"].tolist()))
            
    print(f"Training took {time.time() - start_time:.1f}s")
    
    # Generate some samples
    print(f"\n--- Inference Examples ---")
    model.eval()
    for d in domains:
        x, _, _ = get_batch(data, [d], 1, args.seq_len // 2, device) 
        prompt = x[0].tolist()
        print(f"\n[{d.upper()}] Prompt: {repr(tokenizer.decode(prompt))}")
        
        for _ in range(30):
            logits, _ = model(torch.tensor([prompt], device=device))
            probs = torch.softmax(logits[0, -1] / 0.7, dim=-1) # Temperature 0.7
            next_token = torch.multinomial(probs, 1).item()
            prompt.append(next_token)
            
        print(f"Generated: {repr(tokenizer.decode(prompt))}")
        
    # Print usage diff to show specialization
    print(f"\n--- Routing Analysis ---")
    for d in domains:
        if usage_history[d]:
            start_u = usage_history[d][1][1] if len(usage_history[d]) > 1 else usage_history[d][0][1]
            end_u = usage_history[d][-1][1]
            print(f"[{d}]")
            print(f"  Early Usage: {', '.join([f'{v:.2f}' for v in start_u])}")
            print(f"  Final Usage: {', '.join([f'{v:.2f}' for v in end_u])}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--syn-hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--load-balance", type=float, default=0.5)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--joint-steps", type=int, default=1200)
    p.add_argument("--stabilize-steps", type=int, default=200)
    p.add_argument("--specialization-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    train(p.parse_args())
