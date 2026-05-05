import os
import argparse
import time
import torch
import torch.nn.functional as F

from sra_language_models import MoESRALanguageModel
from sra_experiment import make_optimizer, load_balance_loss, specialization_loss, usage_stats, freeze_router
from data_loader_translation import MultilingualDataLoader

def train(args):
    start_time = time.time()
    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    print(f"--- Scalable Multilingual Translation Training ---")
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

    # Optional: Compile model for speedup (PyTorch 2.x+)
    if args.compile and device == "cuda":
        print("Compiling model with torch.compile...")
        try:
            model = torch.compile(model)
        except Exception as e:
            print(f"torch.compile failed: {e}. Proceeding without compilation.")

    opt = make_optimizer(model, args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device == "cuda")
    
    # Resume from checkpoint if exists
    start_step = 1
    if args.resume and os.path.exists(args.save):
        print(f"Resuming from checkpoint: {args.save}")
        ckpt = torch.load(args.save, map_location=device)
        # Handle compiled model state dict mismatch if necessary (strip _orig_mod.)
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
    
    for step in range(start_step, args.steps + 1):
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

        if step == phase1_end + 1:
            freeze_router(model)
            opt = make_optimizer(model, args.lr) # re-init after freezing
            print(f"Phase transition: stabilization after step {phase1_end}")

        model.train()
        x, y, batch_pairs = loader.get_batch()
        
        opt.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=args.amp and device == "cuda"):
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
            log_str = f"step={step:6d} phase={phase} loss={loss.item():.4f} ce={ce.item():.4f} lb={lb.item():.4f}"
            if phase == "specialize":
                log_str += f" spec={spec_loss.item():.4f}"
            print(log_str)
            
            if args.wandb:
                wandb.log({
                    "step": step,
                    "loss": loss.item(),
                    "ce_loss": ce.item(),
                    "lb_loss": lb.item(),
                    "spec_loss": spec_loss.item(),
                    "phase": phase
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
            print(f"Checkpoint saved at step {step}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="iwslt2017", help="HuggingFace dataset name")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--syn-hidden", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--load-balance", type=float, default=0.1)
    p.add_argument("--specialization-weight", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--joint-steps", type=int, default=7000)
    p.add_argument("--stabilize-steps", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="sra_translation_large.pt")
    p.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile for PyTorch 2.x")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    train(p.parse_args())
