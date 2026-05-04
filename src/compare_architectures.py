import argparse
from sra_experiment import train_single

def run_comparisons(args):
    # Base arguments for train_single
    class TrainArgs:
        def __init__(self):
            self.task = args.task
            self.steps = args.steps
            self.batch_size = args.batch_size
            self.min_len = args.min_len
            self.max_len = args.max_len
            self.dim = args.dim
            self.layers = args.layers
            self.lr = args.lr
            self.seed = args.seed
            self.cpu = args.cpu
            self.profile = True
            self.disable_inference_print = True
            self.log_every = args.steps  # only log final or minimal
            self.save = "dummy_model.pt"
            
            # SRA specific defaults
            self.synapses = 16
            self.k = 2
            self.syn_hidden = 128
            self.load_balance = 0.5
            self.warmup_steps = int(args.steps * 0.1)
            self.joint_steps = int(args.steps * 0.7)
            self.stabilize_steps = int(args.steps * 0.15)
            self.specialization_weight = 0.1
            self.self_gen_weight = 0.5
            
            # Baseline specific
            self.baseline_hidden = 256 # equivalent to 2 synapses

    configs = [
        {"name": "SRA (k=2)", "model_type": "sra", "k": 2},
        # {"name": "SRA (k=16)", "model_type": "sra", "k": 16}, # Skip to save time
        {"name": "Transformer", "model_type": "transformer", "baseline_hidden": 256},
        {"name": "MLP", "model_type": "mlp", "baseline_hidden": 256},
    ]

    results = []
    print(f"=== Starting Comparison for Task: {args.task} (steps: {args.steps}) ===")
    
    for cfg in configs:
        print(f"\nEvaluating: {cfg['name']} ...")
        t_args = TrainArgs()
        for k, v in cfg.items():
            if k != "name":
                setattr(t_args, k, v)
                
        metrics = train_single(t_args)
        if metrics is not None:
            metrics["name"] = cfg["name"]
            results.append(metrics)
        else:
            print(f"Warning: train_single returned None for {cfg['name']}")
        
    print("\n=== Comparison Results ===")
    print("| Architecture | Final Val Loss | Seq Accuracy | Total Time (s) | Max VRAM (MB) |")
    print("|--------------|----------------|--------------|----------------|---------------|")
    for r in results:
        print(f"| {r['name']:<12} | {r['val_loss']:.4f} | {r['seq_acc']:.3f} | {r['total_time']:.2f} | {r['max_mem_mb']:.1f} |")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["copy", "reverse", "paren", "addmod"], default="copy")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--min-len", type=int, default=4)
    p.add_argument("--max-len", type=int, default=10)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu", action="store_true")
    run_comparisons(p.parse_args())
