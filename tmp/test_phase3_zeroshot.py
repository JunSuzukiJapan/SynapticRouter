import sys
import os
sys.path.append(os.path.abspath("src"))

import torch
import tiktoken
import random
import torch.nn.functional as F
from sra_language_models import MoESRALanguageModel
from sra_experiment import make_optimizer, load_balance_loss, usage_stats

# 1. Data
tokenizer = tiktoken.get_encoding("cl100k_base")
vocab_size = tokenizer.n_vocab
domains = ["code", "math", "text"]
data = {}
for d in domains:
    path = f"data/lang/{d}.txt"
    with open(path, "r", encoding="utf-8") as f:
        text = (f.read() + "\n") * 5
    tokens = tokenizer.encode(text, allowed_special="all")
    data[d] = torch.tensor(tokens, dtype=torch.long)

def get_multidomain_batch(data_dict, batch_size, seq_len, subset_domains=None):
    x = torch.zeros((batch_size, seq_len), dtype=torch.long)
    y = torch.zeros((batch_size, seq_len), dtype=torch.long)
    doms = subset_domains if subset_domains else list(data_dict.keys())
    for i in range(batch_size):
        d = random.choice(doms)
        d_data = data_dict[d]
        max_start = len(d_data) - seq_len - 1
        start = random.randint(0, max(0, max_start))
        x[i] = d_data[start:start+seq_len]
        y[i] = d_data[start+1:start+seq_len+1]
    return x, y

# 2. Setup Model
dim = 128
layers = 2
num_synapses = 16
k = 2
syn_hidden = 256
max_seq_len = 64
device = "cpu"
model = MoESRALanguageModel(vocab_size, dim, layers, num_synapses, k, syn_hidden, max_seq_len=max_seq_len).to(device)

batch_size = 32
opt = make_optimizer(model, 5e-4)

# 3. Pre-train on CODE and MATH
print("=== Pre-training on CODE and MATH ===")
model.train()
for step in range(1, 201):
    x, y = get_multidomain_batch(data, batch_size, max_seq_len, ["code", "math"])
    logits, router_logits = model(x)
    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss = ce_loss + 0.1 * load_balance_loss(router_logits)
    
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if step % 50 == 0:
        print(f"Pre-train Step {step} Loss: {loss.item():.4f}")

# Evaluate pre-train
def eval_domain(d, allowed_mask=None):
    torch.manual_seed(42)
    random.seed(42)
    model.eval()
    with torch.no_grad():
        x, y = get_multidomain_batch(data, 100, max_seq_len, [d])
        logits, _ = model(x, allowed_synapses_mask=allowed_mask)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    return loss.item()

print("\n--- Losses Before Hot-Swap ---")
loss_code_before = eval_domain("code")
loss_math_before = eval_domain("math")
loss_text_before = eval_domain("text")
print(f"CODE: {loss_code_before:.4f}, MATH: {loss_math_before:.4f}, TEXT: {loss_text_before:.4f} (Untrained)")

# 4. Hot-Swap (Add 4 new synapses and freeze old ones)
print("\n=== Hot-Swapping 4 new synapses and freezing base/old synapses ===")
model.add_synapses(4, freeze_base=True)
total_synapses = model.blocks[0].router.num_synapses

opt2 = make_optimizer(model, 5e-4)

# Define masks
old_mask = torch.zeros((batch_size, total_synapses), dtype=torch.bool, device=device)
old_mask[:, 0:16] = True

new_mask = torch.zeros((batch_size, total_synapses), dtype=torch.bool, device=device)
new_mask[:, 16:20] = True

# 5. Fine-tune on TEXT only, using Metadata-driven routing (new_mask)
print("\n=== Fine-tuning on TEXT (with Metadata Mask) ===")
model.train()
for step in range(1, 151):
    x, y = get_multidomain_batch(data, batch_size, max_seq_len, ["text"])
    logits, router_logits = model(x, allowed_synapses_mask=new_mask)
    ce_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss = ce_loss
    
    opt2.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt2.step()
    if step % 50 == 0:
        print(f"FT Step {step} Loss: {loss.item():.4f}")

# 6. Final Evaluation
print("\n--- Losses After Hot-Swap (with Metadata Mask) ---")
# Evaluate with appropriate masks
eval_old_mask = torch.zeros((100, total_synapses), dtype=torch.bool, device=device)
eval_old_mask[:, 0:16] = True

eval_new_mask = torch.zeros((100, total_synapses), dtype=torch.bool, device=device)
eval_new_mask[:, 16:20] = True

loss_code_after = eval_domain("code", allowed_mask=eval_old_mask)
loss_math_after = eval_domain("math", allowed_mask=eval_old_mask)
loss_text_after = eval_domain("text", allowed_mask=eval_new_mask)

print(f"CODE: {loss_code_after:.4f} (Diff: {loss_code_after - loss_code_before:+.4f})")
print(f"MATH: {loss_math_after:.4f} (Diff: {loss_math_after - loss_math_before:+.4f})")
print(f"TEXT: {loss_text_after:.4f} (Diff: {loss_text_after - loss_text_before:+.4f})")

if abs(loss_code_after - loss_code_before) < 1e-4 and abs(loss_math_after - loss_math_before) < 1e-4:
    print("\nSUCCESS: Catastrophic Forgetting is MATHEMATICALLY ZERO!")
else:
    print("\nWARNING: Some forgetting occurred.")

# 7. Check Router Usage for TEXT
model.eval()
with torch.no_grad():
    x, y = get_multidomain_batch(data, 100, max_seq_len, ["text"])
    _, router_logits = model(x, allowed_synapses_mask=eval_new_mask)
    u = usage_stats(router_logits)
    usage = (u / u.sum()).cpu().numpy()
    top3 = usage.argsort()[-3:][::-1]
    print(f"\nTEXT top synapses used: {top3} (Note: synapses 16-19 are the newly added ones)")
