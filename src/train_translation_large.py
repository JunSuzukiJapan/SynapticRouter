import os
import argparse
import math
import time
import torch
import torch.nn.functional as F

from sra_language_models import MoESRALanguageModel
from sra_experiment import make_optimizer, load_balance_loss, specialization_loss, freeze_router
from data_loader_translation import MultilingualDataLoader


def cosine_lr(step: int, total_steps: int, lr_max: float, lr_min: float = 1e-5, warmup: int = 500) -> float:
    """コサインアニーリング付きウォームアップLRスケジューラ"""
    if step < warmup:
        return lr_max * step / warmup
    progress = (step - warmup) / max(total_steps - warmup, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def train(args):
    start_time = time.time()
    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    print(f"--- Scalable Multilingual Translation Training (v2) ---")
    print(f"Device: {device}, AMP: {args.amp}, Compile: {args.compile}")

    torch.manual_seed(args.seed)

    # DataLoader setup
    loader = MultilingualDataLoader(
        dataset_name=args.dataset,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        device=device
    )
    loader.prepare_datasets(split="train", streaming=True)
    vocab_size = loader.vocab_size

    # Model setup
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

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Optional: Compile model for speedup (PyTorch 2.x+)
    if args.compile and device == "cuda":
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}. Proceeding without compilation.")

    opt = make_optimizer(model, args.lr)
    # AMP scaler (CUDA only)
    use_amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Resume from checkpoint if exists
    start_step = 1
    if args.resume and os.path.exists(args.save):
        print(f"Resuming from checkpoint: {args.save}")
        ckpt = torch.load(args.save, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0) + 1
        if "scaler_state_dict" in ckpt and scaler.is_enabled():
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps

    # Optional WandB setup
    if args.wandb:
        import wandb
        wandb.init(project="synaptic-router-translation", config=vars(args))

    print(f"Training: {args.steps} steps | phases: warmup={args.warmup_steps} "
          f"joint={args.joint_steps} stabilize={args.stabilize_steps}")
    print(f"LR: {args.lr} (cosine decay to {args.lr_min}) | load_balance={args.load_balance}")

    router_frozen = False
    for step in range(start_step, args.steps + 1):
        # フェーズ判定
        if step <= args.warmup_steps:
            phase = "warmup"
            dense = True
        elif step <= phase1_end:
            phase = "joint"
            dense = False
        elif step <= phase2_end:
            phase = "stabilize"
            dense = False
        else:
            phase = "specialize"
            dense = False

        # ルーターフリーズ（stabilizeフェーズ開始時）
        if step == phase1_end + 1 and not router_frozen:
            freeze_router(model)
            opt = make_optimizer(model, args.lr)
            router_frozen = True
            print(f"Phase transition: stabilization after step {phase1_end}")

        # コサインLRスケジュール
        lr_now = cosine_lr(step, args.steps, args.lr, args.lr_min, args.warmup_steps)
        set_lr(opt, lr_now)

        model.train()
        x, y, batch_pairs = loader.get_batch()

        opt.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device if device != "mps" else "cpu",
                            dtype=torch.float16, enabled=use_amp):
            logits, router_logits = model(x, dense=dense)
            ce = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            lb = load_balance_loss(router_logits)
            loss = ce + args.load_balance * lb

            if phase == "specialize":
                spec_loss = specialization_loss(router_logits)
                loss = loss + args.specialization_weight * spec_loss
            else:
                spec_loss = torch.tensor(0.0, device=device)

        # Backward and step
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        # Logging
        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 1e-6)
            eta_sec = (args.steps - step) / max(steps_per_sec, 1e-6)
            eta_str = f"{eta_sec/3600:.1f}h" if eta_sec > 3600 else f"{eta_sec/60:.1f}m"

            log_str = (f"step={step:6d} phase={phase} lr={lr_now:.2e} "
                       f"loss={loss.item():.4f} ce={ce.item():.4f} lb={lb.item():.5f}")
            if phase == "specialize":
                log_str += f" spec={spec_loss.item():.4f}"
            log_str += f" | {steps_per_sec:.2f}it/s ETA={eta_str}"
            print(log_str)

            if args.wandb:
                import wandb
                wandb.log({
                    "step": step, "loss": loss.item(), "ce_loss": ce.item(),
                    "lb_loss": lb.item(), "spec_loss": spec_loss.item(),
                    "lr": lr_now, "phase": phase
                })

        # Save Checkpoint
        if step % args.save_every == 0:
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "args": vars(args)
            }
            if scaler.is_enabled():
                ckpt["scaler_state_dict"] = scaler.state_dict()
            torch.save(ckpt, args.save)
            print(f"  [ckpt] Saved at step {step} → {args.save}")

    if args.wandb:
        import wandb
        wandb.finish()

    total_time = time.time() - start_time
    print(f"\nTraining complete. Total time: {total_time/3600:.2f}h")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="opus100")
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=96)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--syn-hidden", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5, help="コサイン減衰の最小LR")
    p.add_argument("--load-balance", type=float, default=0.01,
                   help="シナプス均等化の重み（前回0.1→0.01に削減）")
    p.add_argument("--specialization-weight", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--joint-steps", type=int, default=10000)
    p.add_argument("--stabilize-steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="sra_translation_v2.pt")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--wandb", action="store_true")
    train(p.parse_args())
