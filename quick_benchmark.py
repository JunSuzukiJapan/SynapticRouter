import argparse
import time
import torch
import warnings
warnings.filterwarnings("ignore")

from sra_experiment import train_single

def run_quick_comparisons(task="copy", steps=50):
    class TrainArgs:
        def __init__(self):
            self.task = task
            self.steps = steps
            self.batch_size = 128
            self.min_len = 4
            self.max_len = 10
            self.dim = 64
            self.layers = 2
            self.lr = 3e-4
            self.seed = 42
            self.cpu = False
            self.profile = True
            self.disable_inference_print = True
            self.log_every = steps
            self.save = "dummy_model.pt"
            
            # SRA specific defaults
            self.synapses = 16
            self.k = 2
            self.syn_hidden = 128
            self.load_balance = 0.5
            self.warmup_steps = int(steps * 0.1)
            self.joint_steps = int(steps * 0.7)
            self.stabilize_steps = int(steps * 0.15)
            self.specialization_weight = 0.1
            self.self_gen_weight = 0.5
            
            # Baseline specific
            self.baseline_hidden = 256 # equivalent to 2 synapses

    configs = [
        {"name": "Batched SRA", "model_type": "batched_sra", "k": 2},
        {"name": "MoE SRA", "model_type": "moe_sra", "k": 2},
        {"name": "Seq SRA", "model_type": "seq_sra", "k": 2},
        {"name": "Transformer", "model_type": "transformer", "baseline_hidden": 256},
        {"name": "MLP", "model_type": "mlp", "baseline_hidden": 256},
    ]

    results = []
    print(f"=== Quick Performance Benchmark for Task: {task} (steps: {steps}) ===")
    
    for cfg in configs:
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
        
    print("\n=== Quick Comparison Results (Speed & Memory) ===")
    print("| Architecture | Steps/Sec | Total Time (s) | Max VRAM (MB) |")
    print("|--------------|-----------|----------------|---------------|")
    for r in results:
        steps_per_sec = steps / r['total_time']
        print(f"| {r['name']:<12} | {steps_per_sec:9.2f} | {r['total_time']:14.2f} | {r['max_mem_mb']:13.1f} |")

if __name__ == "__main__":
    run_quick_comparisons()
