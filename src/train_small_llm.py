import argparse
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from sra_experiment import freeze_router, load_balance_loss, make_optimizer, specialization_loss, usage_stats
from sra_language_models import MoESRALanguageModel
from tokenizer_utils import build_tokenizer


DATASET_PRESETS = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "train_split": "train",
        "valid_split": "validation",
        "text_column": "text",
    },
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "train_split": "train",
        "valid_split": "validation",
        "text_column": "text",
    },
}


@dataclass
class CorpusBundle:
    train_tokens: torch.Tensor
    valid_tokens: torch.Tensor
    tokenizer: object
    dataset_path: str
    dataset_name: Optional[str]
    text_column: str


def choose_device(device_arg: str, force_cpu: bool) -> str:
    if force_cpu:
        return "cpu"

    mps_backend = getattr(torch.backends, "mps", None)
    mps_built = bool(mps_backend and mps_backend.is_built())
    mps_available = bool(mps_backend and mps_backend.is_available())

    if device_arg == "mps":
        if not mps_built:
            raise SystemExit("PyTorch was built without MPS support.")
        if not mps_available:
            raise SystemExit("PyTorch has MPS support, but MPS is not available in this environment.")
        return "mps"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is not available in this environment.")
        return "cuda"
    if device_arg == "cpu":
        return "cpu"

    if mps_available:
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    if mps_built and not mps_available:
        print("MPS is built into PyTorch but not available at runtime. Falling back to CPU.")
    return "cpu"


def import_datasets():
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The `datasets` package is required. Install it with `python3 -m pip install datasets`."
        ) from exc
    return load_dataset


def resolve_dataset_config(args) -> dict:
    if args.dataset_path:
        cfg = {
            "path": args.dataset_path,
            "name": args.dataset_name,
            "train_split": args.train_split,
            "valid_split": args.valid_split,
            "text_column": args.text_column,
        }
    elif args.dataset in DATASET_PRESETS:
        cfg = dict(DATASET_PRESETS[args.dataset])
    else:
        raise SystemExit(
            "Specify either `--dataset` with a known preset or `--dataset-path` for a custom Hugging Face dataset."
        )
    return cfg


def read_split_text(
    load_dataset,
    *,
    path: str,
    name: Optional[str],
    split: str,
    text_column: str,
    cache_dir: str,
    max_examples: int,
) -> str:
    ds = load_dataset(path, name=name, split=split, cache_dir=cache_dir)
    total = len(ds)
    limit = total if max_examples <= 0 else min(total, max_examples)
    parts = []
    for i in range(limit):
        sample = ds[i]
        text = sample.get(text_column)
        if text is None:
            available = ", ".join(sample.keys())
            raise KeyError(f"Column `{text_column}` not found in dataset sample. Available columns: {available}")
        if not isinstance(text, str):
            text = str(text)
        parts.append(text)
    return "\n".join(parts)


def load_corpus(args) -> CorpusBundle:
    cfg = resolve_dataset_config(args)
    load_dataset = import_datasets()
    tokenizer = build_tokenizer(args.tokenizer_type, args.tokenizer_name)

    train_text = read_split_text(
        load_dataset,
        path=cfg["path"],
        name=cfg["name"],
        split=cfg["train_split"],
        text_column=cfg["text_column"],
        cache_dir=args.cache_dir,
        max_examples=args.max_train_examples,
    )
    valid_text = read_split_text(
        load_dataset,
        path=cfg["path"],
        name=cfg["name"],
        split=cfg["valid_split"],
        text_column=cfg["text_column"],
        cache_dir=args.cache_dir,
        max_examples=args.max_valid_examples,
    )

    if args.repeat_train > 1:
        train_text = ("\n".join([train_text] * args.repeat_train)).strip()

    train_tokens = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    valid_tokens = torch.tensor(tokenizer.encode(valid_text), dtype=torch.long)

    if len(train_tokens) <= args.seq_len + 1:
        raise SystemExit("Training corpus is too short for the configured seq_len.")
    if len(valid_tokens) <= args.seq_len + 1:
        raise SystemExit("Validation corpus is too short for the configured seq_len.")

    print(
        f"Loaded dataset path={cfg['path']} name={cfg['name']} "
        f"train_tokens={len(train_tokens)} valid_tokens={len(valid_tokens)} text_column={cfg['text_column']}"
    )
    return CorpusBundle(
        train_tokens=train_tokens,
        valid_tokens=valid_tokens,
        tokenizer=tokenizer,
        dataset_path=cfg["path"],
        dataset_name=cfg["name"],
        text_column=cfg["text_column"],
    )


def sample_batch(tokens: torch.Tensor, batch_size: int, seq_len: int, device: str):
    max_start = len(tokens) - seq_len - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([tokens[s:s + seq_len] for s in starts.tolist()])
    y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts.tolist()])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, tokens: torch.Tensor, batch_size: int, seq_len: int, eval_batches: int, device: str):
    model.eval()
    total_loss = 0.0
    total_usage = None
    for _ in range(eval_batches):
        x, y = sample_batch(tokens, batch_size, seq_len, device)
        logits, router_logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        total_loss += loss.item()
        if router_logits:
            usage = usage_stats(router_logits)
            total_usage = usage if total_usage is None else total_usage + usage
    mean_usage = None if total_usage is None else total_usage / eval_batches
    return total_loss / eval_batches, mean_usage


def phase_for_step(step: int, args) -> str:
    phase1_end = args.warmup_steps + args.joint_steps
    phase2_end = phase1_end + args.stabilize_steps
    if step <= args.warmup_steps:
        return "warmup"
    if step <= phase1_end:
        return "joint"
    if step <= phase2_end:
        return "stabilize"
    return "specialize"


def save_checkpoint(path: str, model, optimizer, step: int, args, corpus: CorpusBundle):
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": cuda_rng_state,
            "args": vars(args),
            "dataset": {
                "path": corpus.dataset_path,
                "name": corpus.dataset_name,
                "text_column": corpus.text_column,
            },
        },
        path,
    )


def maybe_resume(path: str, model, optimizer, device: str):
    if not path or not os.path.exists(path):
        raise SystemExit(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
    if "python_random_state" in ckpt:
        random.setstate(ckpt["python_random_state"])
    if "torch_rng_state" in ckpt:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if ckpt.get("cuda_rng_state") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
    start_step = int(ckpt.get("step", 0))
    print(f"Resumed from {path} at step={start_step}")
    return start_step


@torch.no_grad()
def generate_text(model, tokenizer, tokens: torch.Tensor, prompt_len: int, max_new_tokens: int, device: str):
    model.eval()
    max_prompt_len = min(prompt_len, len(tokens) - 1)
    start = random.randint(0, len(tokens) - max_prompt_len - 1)
    prompt = tokens[start:start + max_prompt_len].tolist()
    generated = list(prompt)
    max_seq_len = model.pos.num_embeddings
    for _ in range(max_new_tokens):
        ctx = generated[-max_seq_len:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(x)
        probs = torch.softmax(logits[0, -1] / 0.9, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        generated.append(next_token)
    return tokenizer.decode(prompt), tokenizer.decode(generated)


def train(args):
    start_time = time.time()
    device = choose_device(args.device, args.cpu)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    corpus = load_corpus(args)

    model = MoESRALanguageModel(
        vocab_size=corpus.tokenizer.vocab_size,
        dim=args.dim,
        layers=args.layers,
        num_synapses=args.synapses,
        k=args.k,
        syn_hidden=args.syn_hidden,
        max_seq_len=args.seq_len,
    ).to(device)

    optimizer = make_optimizer(model, args.lr)
    start_step = 0
    if args.resume:
        start_step = maybe_resume(args.resume, model, optimizer, device)

    phase1_end = args.warmup_steps + args.joint_steps
    router_already_frozen = start_step > phase1_end

    print(
        f"device={device} steps={args.steps} batch_size={args.batch_size} seq_len={args.seq_len} "
        f"dim={args.dim} layers={args.layers} synapses={args.synapses} k={args.k} "
        f"checkpoint={args.checkpoint_path}"
    )

    for step in range(start_step + 1, args.steps + 1):
        phase = phase_for_step(step, args)
        dense = step <= args.warmup_steps

        if step == phase1_end + 1 and not router_already_frozen:
            freeze_router(model)
            optimizer = make_optimizer(model, args.lr)
            router_already_frozen = True
            print(f"Phase transition: stabilization after step {phase1_end}")

        model.train()
        x, y = sample_batch(corpus.train_tokens, args.batch_size, args.seq_len, device)
        logits, router_logits = model(x, dense=dense)

        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        lb = load_balance_loss(router_logits)
        loss = ce + args.load_balance * lb

        if phase == "specialize":
            spec_loss = specialization_loss(router_logits)
            loss = loss + args.specialization_weight * spec_loss
        else:
            spec_loss = torch.tensor(0.0, device=device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            val_loss, usage = evaluate(
                model, corpus.valid_tokens, args.eval_batch_size, args.seq_len, args.eval_batches, device
            )
            usage_str = ""
            if usage is not None:
                top_usage = ", ".join(f"{v:.2f}" for v in usage.tolist()[: min(8, len(usage))])
                usage_str = f" usage[:8]=[{top_usage}]"
            print(
                f"step={step:5d} phase={phase} train_loss={loss.item():.4f} ce={ce.item():.4f} "
                f"lb={lb.item():.4f} val_loss={val_loss:.4f}"
                + (f" spec={spec_loss.item():.4f}" if phase == "specialize" else "")
                + usage_str
            )

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(args.checkpoint_path, model, optimizer, step, args, corpus)
            print(f"checkpoint saved: {args.checkpoint_path}")

    save_checkpoint(args.checkpoint_path, model, optimizer, args.steps, args, corpus)
    print(f"final checkpoint saved: {args.checkpoint_path}")

    prompt, generated = generate_text(
        model,
        corpus.tokenizer,
        corpus.valid_tokens,
        prompt_len=max(16, args.seq_len // 2),
        max_new_tokens=args.generate_tokens,
        device=device,
    )
    print("\n--- Sample Generation ---")
    print(f"Prompt:\n{prompt}")
    print(f"\nGenerated:\n{generated}")
    print(f"\nTraining took {time.time() - start_time:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Train a small SRA language model on an open Hugging Face dataset.")
    p.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), default="wikitext2")
    p.add_argument("--dataset-path", type=str, default="")
    p.add_argument("--dataset-name", type=str, default=None)
    p.add_argument("--train-split", type=str, default="train")
    p.add_argument("--valid-split", type=str, default="validation")
    p.add_argument("--text-column", type=str, default="text")
    p.add_argument("--cache-dir", type=str, default="data/hf_cache")
    p.add_argument("--max-train-examples", type=int, default=20000)
    p.add_argument("--max-valid-examples", type=int, default=2000)
    p.add_argument("--repeat-train", type=int, default=1)
    p.add_argument("--tokenizer-type", choices=["byte", "tiktoken"], default="byte")
    p.add_argument("--tokenizer-name", type=str, default="cl100k_base")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=16)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--syn-hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--load-balance", type=float, default=0.3)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--joint-steps", type=int, default=1400)
    p.add_argument("--stabilize-steps", type=int, default=300)
    p.add_argument("--specialization-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-path", type=str, default="checkpoints/small_llm_sra.pt")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--generate-tokens", type=int, default=120)
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
