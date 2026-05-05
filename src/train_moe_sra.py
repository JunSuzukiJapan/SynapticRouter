import argparse
import time
import random
import torch
import torch.nn.functional as F

from constants import (
    PAD, BOS, EOS, VOCAB_SIZE, ID2TOK,
    TASK_ORDER, TASK_DEFAULT_STEPS
)

# MoE SRAだけを引っ張ってくる
from sra_gpu_models import MoESRAModel

from sra_experiment import (
    make_batch, make_sample, make_optimizer,
    load_balance_loss, specialization_loss,
    usage_stats, usage_entropy, synapse_stats,
    generate_self_conditioned_prefix, generate_prediction,
    evaluate, decode, freeze_router
)


def train_moe_sra(args):
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
    
    # 専用にMoE SRAだけを初期化
    model = MoESRAModel(VOCAB_SIZE, args.dim, args.layers, args.synapses, args.k, args.syn_hidden).to(device)

    opt = make_optimizer(model, args.lr)
    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps
    print(
        f"device={device} model=moe_sra task={args.task} k={args.k} batch_size={args.batch_size} dim={args.dim} "
        f"layers={args.layers} synapses={args.synapses} "
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
        if step == phase1_end + 1:
            freeze_router(model)
            opt = make_optimizer(model, args.lr)
            print(f"phase transition: stabilization after step {phase1_end}")

        x, y = make_batch(args.task, args.batch_size, args.min_len, args.max_len, device)
        y_in = torch.cat([torch.full((y.size(0), 1), BOS, dtype=torch.long, device=device), y[:, :-1]], dim=1)
        
        # モデルのフォワード実行
        logits, router_logits, all_syn_outputs = model(x, y_in, dense=dense)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD)
        
        # Load Balance Lossの追加
        lb = load_balance_loss(router_logits)
        loss = ce + args.load_balance * lb

        # Self-generation Loss
        if args.self_gen_weight > 0 and phase != "warmup":
            y_in_self = generate_self_conditioned_prefix(model, x, y.size(1), device)
            logits_self, _, _ = model(x, y_in_self)
            ce_self = F.cross_entropy(logits_self.reshape(-1, VOCAB_SIZE), y.reshape(-1), ignore_index=PAD)
            loss = loss + args.self_gen_weight * ce_self
        else:
            ce_self = torch.tensor(0.0, device=device)

        # Specialization Loss
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
            val_loss, seq_acc = evaluate(model, args.task, 20, args.batch_size, args.min_len, args.max_len, device)
            final_val_loss, final_seq_acc = val_loss, seq_acc
            
            usage = usage_stats(router_logits)
            entropy = usage_entropy(usage)
            top_usage = ", ".join(f"{v:.2f}" for v in usage.tolist()[:min(8, len(usage))])
            
            # MoEではダミーテンソル（ゼロ）を返しているため、全て0として表示される
            syn_norms = synapse_stats(all_syn_outputs)
            syn_str = " | ".join(f"L{i}:[" + ", ".join(f"{n:.3f}" for n in norms) + "]" for i, norms in enumerate(syn_norms))
            
            log_str = (
                f"step={step:5d} phase={phase} train_loss={loss.item():.4f} ce={ce.item():.4f} "
                f"lb={lb.item():.4f} val_loss={val_loss:.4f} seq_acc={seq_acc:.3f} entropy={entropy:.3f} "
                f"usage[:8]=[{top_usage}] synapses={syn_str}"
            )
            
            if args.self_gen_weight > 0 and phase != "warmup":
                log_str += f" ce_self={ce_self.item():.4f}"
            if phase == "specialize":
                log_str += f" spec={spec_loss.item():.4f}"
            print(log_str)
            
            sample_x, sample_y = make_sample(args.task, args.min_len, args.max_len)
            sample_x_t = torch.tensor([sample_x], dtype=torch.long, device=device)
            sample_pred = generate_prediction(model, sample_x_t, len(sample_y), device)
            print("sample x=", decode(sample_x), "target=", decode(sample_y), "pred=", decode(sample_pred))

    torch.save(model.state_dict(), args.save)
    print(f"saved: {args.save}")

    # 推論テスト
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
        print("x=", decode(x[0].tolist()), " target=", decode(y[0].tolist()), " pred=", decode(outs))

    return final_val_loss, final_seq_acc


def run_task_suite(args):
    print("Running MoE SRA task suite:", ", ".join(TASK_ORDER))
    for task in TASK_ORDER:
        task_args = argparse.Namespace(**vars(args))
        task_args.task = task
        task_args.steps = TASK_DEFAULT_STEPS[task]
        if args.save == "moe_sra_model.pt":
            task_args.save = f"moe_sra_model_{task}.pt"
        elif args.save.endswith(".pt"):
            base, ext = args.save.rsplit(".", 1)
            task_args.save = f"{base}_{task}.{ext}"
        print(f"\n=== task suite: {task} ({task_args.steps} steps) ===")
        train_moe_sra(task_args)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Standalone training script exclusively for MoE SRA")
    p.add_argument("--task", choices=["copy", "reverse", "paren", "addmod"], default="reverse")
    p.add_argument("--task-suite", action="store_true", help="Run the default minimal task suite sequentially")
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
    p.add_argument("--save", type=str, default="moe_sra_model.pt")
    p.add_argument("--cpu", action="store_true")
    
    args = p.parse_args()
    if args.task_suite:
        run_task_suite(args)
    else:
        train_moe_sra(args)
