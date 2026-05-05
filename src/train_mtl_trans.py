import argparse
import random
import time
import torch
import torch.nn.functional as F
import tiktoken

from sra_language_models import MoESRALanguageModel
from sra_experiment import make_optimizer, load_balance_loss, usage_stats, specialization_loss, freeze_router

# Synthetic parallel corpus
PARALLEL_DATA = [
    {"eng": "I eat apples", "jpn": "私 は りんご を 食べる", "fra": "Je mange des pommes"},
    {"eng": "She reads books", "jpn": "彼女 は 本 を 読む", "fra": "Elle lit des livres"},
    {"eng": "We love cats", "jpn": "私たち は 猫 が 好きです", "fra": "Nous aimons les chats"},
    {"eng": "They play games", "jpn": "彼ら は ゲーム を する", "fra": "Ils jouent aux jeux"},
    {"eng": "He drinks water", "jpn": "彼 は 水 を 飲む", "fra": "Il boit de l' eau"},
    {"eng": "The dog chases the cat", "jpn": "犬 は 猫 を 追いかける", "fra": "Le chien chasse le chat"},
    {"eng": "A bird flies in the sky", "jpn": "鳥 は 空 を 飛ぶ", "fra": "Un oiseau vole dans le ciel"},
    {"eng": "You write a letter", "jpn": "あなた は 手紙 を 書く", "fra": "Tu ecris une lettre"},
    {"eng": "The teacher explains the lesson", "jpn": "先生 は 授業 を 説明 する", "fra": "Le professeur explique la lecon"},
    {"eng": "My friend buys a car", "jpn": "私 の 友達 は 車 を 買う", "fra": "Mon ami achete une voiture"},
    {"eng": "I see a bird", "jpn": "私 は 鳥 を 見る", "fra": "Je vois un oiseau"},
    {"eng": "She likes dogs", "jpn": "彼女 は 犬 が 好きです", "fra": "Elle aime les chiens"},
    {"eng": "We learn math", "jpn": "私たち は 数学 を 学ぶ", "fra": "Nous apprenons les mathematiques"},
    {"eng": "They build a house", "jpn": "彼ら は 家 を 建てる", "fra": "Ils construisent une maison"},
    {"eng": "He drives a car", "jpn": "彼 は 車 を 運転 する", "fra": "Il conduit une voiture"},
    {"eng": "The cat sleeps on the bed", "jpn": "猫 は ベッド で 眠る", "fra": "Le chat dort sur le lit"},
    {"eng": "A fish swims in the river", "jpn": "魚 は 川 を 泳ぐ", "fra": "Un poisson nage dans la riviere"},
    {"eng": "You sing a song", "jpn": "あなた は 歌 を 歌う", "fra": "Tu chantes une chanson"},
    {"eng": "The student does homework", "jpn": "生徒 は 宿題 を する", "fra": "L' etudiant fait ses devoirs"},
    {"eng": "My mother cooks dinner", "jpn": "私 の 母 は 夕食 を 作る", "fra": "Ma mere prepare le diner"},
]

LANG_TAGS = {"eng": "[ENG]", "jpn": "[JPN]", "fra": "[FRA]"}
TARGET_TAGS = {"eng": "[TARGET_ENG]", "jpn": "[TARGET_JPN]", "fra": "[TARGET_FRA]"}

def prepare_translation_batch(tokenizer, batch_size, seq_len, device):
    x = torch.full((batch_size, seq_len), 0, dtype=torch.long)
    y = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    batch_pairs = []
    
    langs = ["eng", "jpn", "fra"]
    
    for i in range(batch_size):
        src_lang = random.choice(langs)
        tgt_lang = random.choice([l for l in langs if l != src_lang])
        
        # ZERO-SHOT TEST: Exclude fra<->jpn during training
        while (src_lang == "fra" and tgt_lang == "jpn") or (src_lang == "jpn" and tgt_lang == "fra"):
            src_lang = random.choice(langs)
            tgt_lang = random.choice([l for l in langs if l != src_lang])
            
        batch_pairs.append(f"{src_lang}->{tgt_lang}")
        
        pair = random.choice(PARALLEL_DATA)
        src_text = pair[src_lang]
        tgt_text = pair[tgt_lang]
        
        prompt_str = f"{LANG_TAGS[src_lang]} {src_text} [SEP] {TARGET_TAGS[tgt_lang]} "
        target_str = f"{tgt_text} [EOS]"
        
        prompt_tokens = tokenizer.encode(prompt_str, allowed_special="all")
        target_tokens = tokenizer.encode(target_str, allowed_special="all")
        
        all_tokens = prompt_tokens + target_tokens
        
        if len(all_tokens) > seq_len + 1:
            all_tokens = all_tokens[:seq_len+1]
            
        x_seq = all_tokens[:-1]
        y_seq = all_tokens[1:]
        
        x[i, :len(x_seq)] = torch.tensor(x_seq, dtype=torch.long)
        y[i, len(prompt_tokens)-1:len(y_seq)] = torch.tensor(y_seq[len(prompt_tokens)-1:], dtype=torch.long)

    return x.to(device), y.to(device), batch_pairs

@torch.no_grad()
def evaluate_translation(model, tokenizer, batches, batch_size, seq_len, device):
    model.eval()
    langs = ["eng", "jpn", "fra"]
    results = {}
    
    for src_lang in langs:
        for tgt_lang in langs:
            if src_lang == tgt_lang:
                continue
            pair_name = f"{src_lang}->{tgt_lang}"
            total_loss = 0.0
            total_tok = 0
            task_usage = None
            
            for _ in range(batches):
                x = torch.full((batch_size, seq_len), 0, dtype=torch.long)
                y = torch.full((batch_size, seq_len), -100, dtype=torch.long)
                
                for i in range(batch_size):
                    pair = random.choice(PARALLEL_DATA)
                    src_text = pair[src_lang]
                    tgt_text = pair[tgt_lang]
                    
                    prompt_str = f"{LANG_TAGS[src_lang]} {src_text} [SEP] {TARGET_TAGS[tgt_lang]} "
                    target_str = f"{tgt_text} [EOS]"
                    
                    prompt_tokens = tokenizer.encode(prompt_str, allowed_special="all")
                    target_tokens = tokenizer.encode(target_str, allowed_special="all")
                    
                    all_tokens = prompt_tokens + target_tokens
                    if len(all_tokens) > seq_len + 1:
                        all_tokens = all_tokens[:seq_len+1]
                        
                    x_seq = all_tokens[:-1]
                    y_seq = all_tokens[1:]
                    
                    x[i, :len(x_seq)] = torch.tensor(x_seq, dtype=torch.long)
                    y[i, len(prompt_tokens)-1:len(y_seq)] = torch.tensor(y_seq[len(prompt_tokens)-1:], dtype=torch.long)
                    
                x, y = x.to(device), y.to(device)
                logits, router_logits = model(x)
                
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100, reduction="sum")
                
                valid_toks = (y != -100).sum().item()
                total_loss += loss.item()
                total_tok += valid_toks
                
                if router_logits:
                    u = usage_stats(router_logits)
                    task_usage = u if task_usage is None else task_usage + u
                    
            results[pair_name] = {
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
    vocab_size = tokenizer.n_vocab + 100 
    
    model = MoESRALanguageModel(
        vocab_size=vocab_size,
        dim=args.dim,
        layers=args.layers,
        num_synapses=args.synapses,
        k=args.k,
        syn_hidden=args.syn_hidden,
        pad_idx=0,
        max_seq_len=args.seq_len
    ).to(device)

    opt = make_optimizer(model, args.lr)
    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps
    
    print(f"--- Multilingual Translation Training ---")
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

        print(f"Step {step}: preparing batch...")
        x, y, batch_pairs = prepare_translation_batch(tokenizer, args.batch_size, args.seq_len, device)
        print(f"Step {step}: forward pass...")
        logits, router_logits = model(x, dense=dense)
        
        print(f"Step {step}: computing loss...")
        ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
        lb = load_balance_loss(router_logits)
        loss = ce + args.load_balance * lb
        
        if phase == "specialize":
            spec_loss = specialization_loss(router_logits)
            loss = loss + args.specialization_weight * spec_loss
        else:
            spec_loss = torch.tensor(0.0, device=device)

        print(f"Step {step}: backward pass...")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        print(f"Step {step}: optimizer step...")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        print(f"Step {step}: evaluate/log...")
        if step % args.log_every == 0 or step == 1:
            eval_results = evaluate_translation(model, tokenizer, 2, args.batch_size, args.seq_len, device)
            
            log_str = f"step={step:4d} phase={phase} loss={loss.item():.4f} ce={ce.item():.4f} lb={lb.item():.4f}"
            if phase == "specialize":
                log_str += f" spec={spec_loss.item():.4f}"
                
            print(log_str)
            for pair_name, res in eval_results.items():
                top_usage = ", ".join(f"{v:.2f}" for v in res["usage"].tolist()[:min(8, len(res["usage"]))]) if res["usage"] is not None else ""
                print(f"  [{pair_name}] val_loss={res['val_loss']:.4f} usage[:8]=[{top_usage}]")
            
    torch.save(model.state_dict(), args.save)
    print(f"Saved: {args.save}")
    
    print(f"\n--- Inference Examples ---")
    model.eval()
    test_cases = [
        ("eng", "jpn", "I eat apples"),
        ("jpn", "eng", "私 は りんご を 食べる"),
        ("eng", "fra", "She reads books"), 
        ("fra", "eng", "Elle lit des livres"),
        # Zero-shot tests:
        ("jpn", "fra", "彼 は 車 を 運転 する"),
        ("fra", "jpn", "Il conduit une voiture"),
    ]
    
    for src_lang, tgt_lang, text in test_cases:
        prompt_str = f"{LANG_TAGS[src_lang]} {text} [SEP] {TARGET_TAGS[tgt_lang]} "
        prompt_tokens = tokenizer.encode(prompt_str, allowed_special="all")
        
        print(f"\n[{src_lang}->{tgt_lang}] Prompt: {prompt_str}")
        
        prompt = prompt_tokens.copy()
        for _ in range(20): 
            x = torch.tensor([prompt], device=device)
            logits, _ = model(x)
            probs = torch.softmax(logits[0, -1] / 0.8, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            prompt.append(next_token)
            
            decoded = tokenizer.decode(prompt)
            if "[EOS]" in decoded:
                break
                
        print(f"Generated: {tokenizer.decode(prompt[len(prompt_tokens):])}")

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
    p.add_argument("--save", type=str, default="sra_trans_model.pt")
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())
