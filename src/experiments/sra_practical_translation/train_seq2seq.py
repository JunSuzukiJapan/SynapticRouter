"""
SRA Seq2Seq 翻訳モデル — 学習スクリプト
- Teacher forcing による標準 Seq2Seq 学習
- Label smoothing (0.1) でサンプル効率向上
- コサイン LR スケジューラ（warmup 付き）
- 1000 ステップごとに greedy BLEU を検証
- Encoder / Decoder の router_logits を両方使って load_balance_loss を計算
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import argparse
import math
import time

import torch
import torch.nn.functional as F

from sra_seq2seq_model import SRATranslationModel
from data_loader_seq2seq import Seq2SeqDataLoader, build_special_tokens
from sra_experiment import make_optimizer, load_balance_loss, freeze_router
import tiktoken


# ──────────────────────────────────────────────
# ユーティリティ
# ──────────────────────────────────────────────
def cosine_lr(step, total, lr_max, lr_min=1e-5, warmup=1000):
    if step < warmup:
        return lr_max * step / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def label_smoothed_ce(logits, targets, smoothing=0.1, ignore_index=-100):
    """
    logits : (B*T, V)
    targets: (B*T,)   -100 は無視
    """
    V = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)

    mask = targets != ignore_index
    if mask.sum() == 0:
        return logits.new_tensor(0.0)

    # NLL loss（正解クラス）
    nll = F.nll_loss(log_probs, targets.clamp(min=0), ignore_index=ignore_index, reduction="mean")

    # Smooth loss（全クラス均等）
    smooth = -log_probs[mask].mean()

    return (1.0 - smoothing) * nll + smoothing * smooth


# ──────────────────────────────────────────────
# 検証用 greedy BLEU（簡易 1-gram precision）
# ──────────────────────────────────────────────
EVAL_PAIRS = [
    ("en", "fr", "I eat apples.", "Je mange des pommes."),
    ("en", "fr", "She reads books.", "Elle lit des livres."),
    ("en", "fr", "Good morning.", "Bonjour."),
    ("en", "fr", "Thank you very much.", "Merci beaucoup."),
    ("en", "fr", "The dog runs fast.", "Le chien court vite."),
    ("fr", "en", "Je mange des pommes.", "I eat apples."),
    ("fr", "en", "Elle lit des livres.", "She reads books."),
    ("fr", "en", "Bonjour.", "Good morning."),
    ("en", "ja", "I eat apples.", "私はりんごを食べます。"),
    ("en", "ja", "Good morning.", "おはようございます。"),
    ("ja", "en", "私 は りんご を 食べる。", "I eat apples."),
]


def simple_bleu1(hyp: str, ref: str) -> float:
    h_tok = hyp.lower().split()
    r_tok = ref.lower().split()
    if not h_tok:
        return 0.0
    ref_counts: dict = {}
    for t in r_tok:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    match = 0
    for t in h_tok:
        if ref_counts.get(t, 0) > 0:
            match += 1
            ref_counts[t] -= 1
    bp = min(1.0, len(h_tok) / max(len(r_tok), 1))
    return bp * match / len(h_tok)


def evaluate(model, loader, device, step):
    """greedy decode で EVAL_PAIRS を翻訳して 1-gram BLEU を計算"""
    tokenizer = loader.tokenizer
    sp = loader.sp

    _LANG_CODE = {"en": "ENG", "ja": "JPN", "fr": "FRA"}

    model.eval()
    bleus = []
    samples = []
    for src_lang, tgt_lang, src_text, ref_text in EVAL_PAIRS:
        src_code = _LANG_CODE[src_lang]
        tgt_code = _LANG_CODE[tgt_lang]
        src_tag = [sp[f"[{src_code}]"]]
        src_ids = src_tag + tokenizer.encode(src_text)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        bos_id = sp[f"[TARGET_{tgt_code}]"]
        eos_id = sp["[EOS]"]

        gen = model.greedy_decode(src_tensor, bos_id=bos_id, eos_id=eos_id, max_len=80)
        gen_ids = gen[0].tolist()
        # EOS 以降を切り捨て
        if eos_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.index(eos_id)]
        # BOS を除去
        gen_ids = [i for i in gen_ids if i != bos_id]
        try:
            pred_text = tokenizer.decode(gen_ids)
        except Exception:
            pred_text = ""
        bleus.append(simple_bleu1(pred_text, ref_text))
        samples.append((src_lang, tgt_lang, src_text, ref_text, pred_text))

    avg_bleu = sum(bleus) / max(len(bleus), 1)
    print(f"\n  [eval step={step}] avg BLEU-1 = {avg_bleu:.4f}")
    for i, (sl, tl, src, ref, pred) in enumerate(samples[:4]):
        mark = "✅" if bleus[i] > 0.3 else ("△" if bleus[i] > 0.05 else "❌")
        print(f"    {mark} {sl.upper()}→{tl.upper()} | src: {src[:40]}")
        print(f"       ref: {ref[:50]}")
        print(f"       pred:{pred[:50]}")
    model.train()
    return avg_bleu


# ──────────────────────────────────────────────
# 学習メイン
# ──────────────────────────────────────────────
def train(args):
    start = time.time()
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    print(f"=== SRA Seq2Seq Translation Training ===")
    print(f"Device: {device}  Steps: {args.steps}  Batch: {args.batch_size}")

    torch.manual_seed(args.seed)

    loader = Seq2SeqDataLoader(
        src_max_len=args.src_max_len,
        tgt_max_len=args.tgt_max_len,
        batch_size=args.batch_size,
        device=device,
    )
    vocab_size = loader.vocab_size

    model = SRATranslationModel(
        vocab_size=vocab_size,
        dim=args.dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        num_synapses=args.synapses,
        k=args.k,
        syn_hidden=args.syn_hidden,
        num_heads=args.num_heads,
        pad_idx=loader.PAD,
        max_src_len=args.src_max_len,
        max_tgt_len=args.tgt_max_len,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    opt = make_optimizer(model, args.lr)

    # チェックポイントから再開
    start_step = 1
    best_bleu = 0.0
    if args.resume and os.path.exists(args.save):
        ckpt = torch.load(args.save, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        best_bleu = ckpt.get("best_bleu", 0.0)
        print(f"Resumed from step {start_step - 1}, best_bleu={best_bleu:.4f}")

    print(f"LR: {args.lr} (cosine → {args.lr_min}) | smoothing={args.label_smoothing} | lb={args.load_balance}")

    # ウォームアップ（dense routing）終了ステップ
    warmup_end = args.warmup_steps

    for step in range(start_step, args.steps + 1):
        # LR スケジュール
        lr = cosine_lr(step, args.steps, args.lr, args.lr_min, args.warmup_steps)
        set_lr(opt, lr)

        # warmup 中は dense（全シナプス使用）
        dense = step <= warmup_end

        model.train()
        src, tgt_in, tgt_lab, pairs = loader.get_batch()

        opt.zero_grad(set_to_none=True)

        logits, router_logits = model(src, tgt_in, dense=dense)

        # Label-smoothed CE（PAD 位置 -100 は無視）
        B, T, V = logits.shape
        ce = label_smoothed_ce(
            logits.view(B * T, V),
            tgt_lab.view(B * T),
            smoothing=args.label_smoothing,
        )

        # Load balance loss（Encoder + Decoder の全 router を使用）
        lb = load_balance_loss(router_logits)
        loss = ce + args.load_balance * lb

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        # ──── ログ ────
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start
            speed = step / max(elapsed, 1e-6)
            eta = (args.steps - step) / max(speed, 1e-6)
            eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.1f}m"
            mode = "dense" if dense else "sparse"
            print(f"step={step:6d} [{mode}] lr={lr:.2e} loss={loss.item():.4f} "
                  f"ce={ce.item():.4f} lb={lb.item():.5f} | {speed:.2f}it/s ETA={eta_str}")

        # ──── 検証 ────
        if step % args.eval_every == 0:
            bleu = evaluate(model, loader, device, step)
            if bleu > best_bleu:
                best_bleu = bleu
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "best_bleu": best_bleu,
                    "args": vars(args),
                }, args.save.replace(".pt", "_best.pt"))
                print(f"  → Best model saved (BLEU={best_bleu:.4f})")

        # ──── チェックポイント ────
        if step % args.save_every == 0:
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "best_bleu": best_bleu,
                "args": vars(args),
            }, args.save)
            print(f"  [ckpt] step={step} → {args.save}")

    total = time.time() - start
    print(f"\nTraining complete. Total time: {total/3600:.2f}h  Best BLEU-1: {best_bleu:.4f}")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps",        type=int,   default=30000)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--src-max-len",  type=int,   default=80)
    p.add_argument("--tgt-max-len",  type=int,   default=80)
    p.add_argument("--dim",          type=int,   default=256)
    p.add_argument("--enc-layers",   type=int,   default=3)
    p.add_argument("--dec-layers",   type=int,   default=3)
    p.add_argument("--synapses",     type=int,   default=12)
    p.add_argument("--k",            type=int,   default=3)
    p.add_argument("--syn-hidden",   type=int,   default=512)
    p.add_argument("--num-heads",    type=int,   default=4)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--lr-min",       type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int,   default=1000)
    p.add_argument("--load-balance", type=float, default=0.005)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--grad-clip",    type=float, default=1.0)
    p.add_argument("--log-every",    type=int,   default=200)
    p.add_argument("--eval-every",   type=int,   default=2000)
    p.add_argument("--save-every",   type=int,   default=2000)
    p.add_argument("--save",         type=str,   default="../../../sra_seq2seq.pt")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--resume",       action="store_true")
    train(p.parse_args())
