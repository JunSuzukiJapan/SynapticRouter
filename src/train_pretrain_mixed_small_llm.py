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
from train_small_llm import choose_device


SOURCE_PRESETS = {
    "tinystories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "data_dir": None,
        "train_split": "train",
        "valid_split": "validation",
        "text_column": "text",
        "train_examples": 120000,
        "valid_examples": 4000,
        "weight": 0.45,
    },
    "wikitext2": {
        "path": "wikitext",
        "name": "wikitext-2-raw-v1",
        "data_dir": None,
        "train_split": "train",
        "valid_split": "validation",
        "text_column": "text",
        "train_examples": 36718,
        "valid_examples": 3760,
        "weight": 0.10,
    },
    "fineweb_edu_sample": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "data_dir": None,
        "train_split": "train",
        "valid_split": None,
        "text_column": "text",
        "train_examples": 50000,
        "valid_examples": 2000,
        "weight": 0.45,
    },
    "llm_jp_synth": {
        "path": "llm-jp/scaling-data-constrained-llms",
        "name": None,
        "data_dir": None,
        "train_split": "train",
        "valid_split": None,
        "text_column": "text",
        "train_examples": 10000,
        "valid_examples": 500,
        "weight": 0.10,
    },
}

MODEL_PRESETS = {
    "8gb_safe": {"dim": 384, "layers": 6, "synapses": 32, "syn_hidden": 768, "seq_len": 256, "batch_size": 4, "eval_batch_size": 4, "grad_accum": 4},
    "8gb_max": {"dim": 512, "layers": 8, "synapses": 32, "syn_hidden": 1024, "seq_len": 256, "batch_size": 2, "eval_batch_size": 2, "grad_accum": 8},
}


@dataclass
class SourceCorpus:
    name: str
    train_tokens: torch.Tensor
    valid_tokens: torch.Tensor
    weight: float
    path: str
    hf_name: Optional[str]
    text_column: str


@dataclass
class MixedCorpusBundle:
    sources: list[SourceCorpus]
    tokenizer_type: str
    tokenizer_name: str
    vocab_size: int


def import_datasets():
    try:
        from datasets import DownloadConfig, load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "The `datasets` package is required. Install it with `python3 -m pip install datasets`."
        ) from exc
    return load_dataset, DownloadConfig


def load_dataset_cache_first(load_dataset, download_config, local_only: bool, **kwargs):
    if local_only:
        return load_dataset(
            **kwargs,
            download_config=download_config(local_files_only=True),
        )

    try:
        ds = load_dataset(
            **kwargs,
            download_config=download_config(local_files_only=True),
        )
        print(f"cache hit: {kwargs['path']} split={kwargs.get('split')}")
        return ds
    except Exception as exc:
        print(f"cache miss: {kwargs['path']} split={kwargs.get('split')} fallback=remote reason={exc.__class__.__name__}")
        return load_dataset(
            **kwargs,
            download_config=download_config(local_files_only=False),
        )


def parse_sources(spec: str) -> list[str]:
    names = [s.strip() for s in spec.split(",") if s.strip()]
    if not names:
        raise SystemExit("At least one source must be provided.")
    for name in names:
        if name not in SOURCE_PRESETS:
            raise SystemExit(f"Unknown source: {name}")
    return names


def maybe_apply_model_preset(args):
    if not args.model_preset:
        return args
    preset = MODEL_PRESETS[args.model_preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def resolve_required_args(args):
    required = ["dim", "layers", "synapses", "syn_hidden", "seq_len", "batch_size", "eval_batch_size", "grad_accum"]
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing model/training settings: {', '.join(missing)}")


def read_text_split(load_dataset, download_config, cfg: dict, split: str, max_examples: int, cache_dir: str, local_only: bool) -> str:
    actual_split = cfg[split]
    if actual_split is None:
        return ""
    ds = load_dataset_cache_first(
        load_dataset,
        download_config,
        local_only,
        path=cfg["path"],
        name=cfg["name"],
        data_dir=cfg.get("data_dir"),
        split=actual_split,
        cache_dir=cache_dir,
    )
    total = len(ds)
    limit = total if max_examples <= 0 else min(total, max_examples)
    parts = []
    for i in range(limit):
        sample = ds[i]
        text = sample.get(cfg["text_column"])
        if text is None:
            available = ", ".join(sample.keys())
            raise KeyError(f"Column `{cfg['text_column']}` not found in dataset sample. Available columns: {available}")
        if not isinstance(text, str):
            text = str(text)
        parts.append(text)
    return "\n".join(parts)


def tokenize_text(tokenizer, text: str) -> torch.Tensor:
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def load_mixed_corpus(args) -> MixedCorpusBundle:
    tokenizer = build_tokenizer(args.tokenizer_type, args.tokenizer_name)
    load_dataset, DownloadConfig = import_datasets()
    sources = []

    for source_name in parse_sources(args.sources):
        cfg = dict(SOURCE_PRESETS[source_name])
        train_text = read_text_split(
            load_dataset,
            DownloadConfig,
            cfg,
            "train_split",
            getattr(args, f"max_train_examples_{source_name}", cfg["train_examples"]),
            args.cache_dir,
            args.local_datasets_only,
        )
        valid_text = read_text_split(
            load_dataset,
            DownloadConfig,
            cfg,
            "valid_split",
            getattr(args, f"max_valid_examples_{source_name}", cfg["valid_examples"]),
            args.cache_dir,
            args.local_datasets_only,
        )

        if cfg["valid_split"] is None or len(valid_text.strip()) == 0:
            # Reuse the front slice of train text for validation if the source doesn't expose a validation split.
            valid_text = train_text[: min(len(train_text), max(20000, len(train_text) // 20))]

        train_tokens = tokenize_text(tokenizer, train_text)
        valid_tokens = tokenize_text(tokenizer, valid_text)
        if len(train_tokens) <= args.seq_len + 1:
            raise SystemExit(f"Training corpus for {source_name} is too short for seq_len={args.seq_len}")
        if len(valid_tokens) <= args.seq_len + 1:
            raise SystemExit(f"Validation corpus for {source_name} is too short for seq_len={args.seq_len}")

        weight = getattr(args, f"weight_{source_name}", cfg["weight"])
        sources.append(
            SourceCorpus(
                name=source_name,
                train_tokens=train_tokens,
                valid_tokens=valid_tokens,
                weight=weight,
                path=cfg["path"],
                hf_name=cfg["name"],
                text_column=cfg["text_column"],
            )
        )
        print(
            f"loaded source={source_name} path={cfg['path']} name={cfg['name']} data_dir={cfg.get('data_dir')} "
            f"train_tokens={len(train_tokens)} valid_tokens={len(valid_tokens)} weight={weight:.3f}"
        )

    return MixedCorpusBundle(
        sources=sources,
        tokenizer_type=args.tokenizer_type,
        tokenizer_name=args.tokenizer_name,
        vocab_size=tokenizer.vocab_size,
    )


def weighted_source_choice(sources: list[SourceCorpus]) -> SourceCorpus:
    weights = [max(0.0, s.weight) for s in sources]
    total = sum(weights)
    if total <= 0:
        raise SystemExit("Source weights must sum to a positive value.")
    return random.choices(sources, weights=weights, k=1)[0]


def sample_from_tokens(tokens: torch.Tensor, batch_size: int, seq_len: int, device: str):
    max_start = len(tokens) - seq_len - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))
    x = torch.stack([tokens[s:s + seq_len] for s in starts.tolist()])
    y = torch.stack([tokens[s + 1:s + seq_len + 1] for s in starts.tolist()])
    return x.to(device), y.to(device)


def sample_mixed_batch(bundle: MixedCorpusBundle, batch_size: int, seq_len: int, device: str):
    per_source = {src.name: [] for src in bundle.sources}
    for _ in range(batch_size):
        src = weighted_source_choice(bundle.sources)
        per_source[src.name].append(src)

    xs = []
    ys = []
    batch_sources = []
    source_map = {src.name: src for src in bundle.sources}
    for source_name, picks in per_source.items():
        if not picks:
            continue
        src = source_map[source_name]
        x, y = sample_from_tokens(src.train_tokens, len(picks), seq_len, device)
        xs.append(x)
        ys.append(y)
        batch_sources.extend([source_name] * len(picks))

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    perm = torch.randperm(x.size(0), device=device)
    return x[perm], y[perm], batch_sources


@torch.no_grad()
def evaluate(model, bundle: MixedCorpusBundle, batch_size: int, seq_len: int, eval_batches: int, device: str):
    model.eval()
    results = {}
    total_weight = sum(src.weight for src in bundle.sources)
    weighted_val = 0.0
    weighted_usage = None

    for src in bundle.sources:
        total_loss = 0.0
        total_usage = None
        for _ in range(eval_batches):
            x, y = sample_from_tokens(src.valid_tokens, batch_size, seq_len, device)
            logits, router_logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            total_loss += loss.item()
            if router_logits:
                usage = usage_stats(router_logits)
                total_usage = usage if total_usage is None else total_usage + usage
        mean_loss = total_loss / eval_batches
        mean_usage = None if total_usage is None else total_usage / eval_batches
        results[src.name] = {"val_loss": mean_loss, "usage": mean_usage, "weight": src.weight}
        weighted_val += mean_loss * src.weight
        if mean_usage is not None:
            weighted_usage = mean_usage * src.weight if weighted_usage is None else weighted_usage + mean_usage * src.weight

    if total_weight > 0:
        weighted_val /= total_weight
        if weighted_usage is not None:
            weighted_usage /= total_weight
    return weighted_val, weighted_usage, results


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


def save_checkpoint(path: str, model, optimizer, step: int, args, bundle: MixedCorpusBundle):
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
            "sources": [
                {
                    "name": src.name,
                    "path": src.path,
                    "hf_name": src.hf_name,
                    "text_column": src.text_column,
                    "weight": src.weight,
                }
                for src in bundle.sources
            ],
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

    maybe_apply_model_preset(args)
    resolve_required_args(args)
    bundle = load_mixed_corpus(args)
    tokenizer = build_tokenizer(args.tokenizer_type, args.tokenizer_name)

    model = MoESRALanguageModel(
        vocab_size=bundle.vocab_size,
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
        f"device={device} preset={args.model_preset} steps={args.steps} seq_len={args.seq_len} "
        f"batch_size={args.batch_size} grad_accum={args.grad_accum} dim={args.dim} layers={args.layers} "
        f"synapses={args.synapses} syn_hidden={args.syn_hidden} tokenizer={args.tokenizer_type}:{args.tokenizer_name}"
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
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        ce_acc = 0.0
        lb_acc = 0.0
        spec_acc = 0.0
        batch_sources = []
        for micro in range(args.grad_accum):
            x, y, sources = sample_mixed_batch(bundle, args.batch_size, args.seq_len, device)
            logits, router_logits = model(x, dense=dense)
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            lb = load_balance_loss(router_logits)
            loss = ce + args.load_balance * lb
            if phase == "specialize":
                spec_loss = specialization_loss(router_logits)
                loss = loss + args.specialization_weight * spec_loss
                spec_acc += spec_loss.item()
            else:
                spec_loss = None

            (loss / args.grad_accum).backward()
            loss_acc += loss.item()
            ce_acc += ce.item()
            lb_acc += lb.item()
            batch_sources.extend(sources)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            val_loss, usage, per_source = evaluate(
                model, bundle, args.eval_batch_size, args.seq_len, args.eval_batches, device
            )
            usage_str = ""
            if usage is not None:
                top_usage = ", ".join(f"{v:.2f}" for v in usage.tolist()[: min(8, len(usage))])
                usage_str = f" usage[:8]=[{top_usage}]"
            print(
                f"step={step:5d} phase={phase} train_loss={loss_acc/args.grad_accum:.4f} "
                f"ce={ce_acc/args.grad_accum:.4f} lb={lb_acc/args.grad_accum:.4f} "
                f"val_loss={val_loss:.4f} batch_sources={batch_sources[:min(8, len(batch_sources))]}"
                + (f" spec={spec_acc/args.grad_accum:.4f}" if phase == "specialize" else "")
                + usage_str
            )
            for source_name, metrics in per_source.items():
                print(f"  [{source_name}] val_loss={metrics['val_loss']:.4f} weight={metrics['weight']:.3f}")

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(args.checkpoint_path, model, optimizer, step, args, bundle)
            print(f"checkpoint saved: {args.checkpoint_path}")

    save_checkpoint(args.checkpoint_path, model, optimizer, args.steps, args, bundle)
    print(f"final checkpoint saved: {args.checkpoint_path}")

    sample_source = max(bundle.sources, key=lambda s: s.weight)
    prompt, generated = generate_text(
        model,
        tokenizer,
        sample_source.valid_tokens,
        prompt_len=max(16, args.seq_len // 2),
        max_new_tokens=args.generate_tokens,
        device=device,
    )
    print("\n--- Sample Generation ---")
    print(f"Prompt:\n{prompt}")
    print(f"\nGenerated:\n{generated}")
    print(f"\nTraining took {time.time() - start_time:.1f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Mixed pretraining for a larger SRA LM using multiple open datasets.")
    p.add_argument("--sources", type=str, default="tinystories,fineweb_edu_sample,wikitext2")
    p.add_argument("--cache-dir", type=str, default="data/hf_cache")
    p.add_argument("--local-datasets-only", action="store_true")
    p.add_argument("--tokenizer-type", choices=["byte", "tiktoken"], default="tiktoken")
    p.add_argument("--tokenizer-name", type=str, default="cl100k_base")
    p.add_argument("--model-preset", choices=sorted(MODEL_PRESETS.keys()), default="8gb_safe")

    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--synapses", type=int, default=None)
    p.add_argument("--syn-hidden", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--k", type=int, default=2)

    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--load-balance", type=float, default=0.2)
    p.add_argument("--warmup-steps", type=int, default=400)
    p.add_argument("--joint-steps", type=int, default=2800)
    p.add_argument("--stabilize-steps", type=int, default=600)
    p.add_argument("--specialization-weight", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-path", type=str, default="checkpoints/mixed_sra_pretrain.pt")
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--generate-tokens", type=int, default=160)
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu", action="store_true")

    # Per-source overrides
    for source_name, cfg in SOURCE_PRESETS.items():
        p.add_argument(f"--max-train-examples-{source_name}", dest=f"max_train_examples_{source_name}", type=int, default=cfg["train_examples"])
        p.add_argument(f"--max-valid-examples-{source_name}", dest=f"max_valid_examples_{source_name}", type=int, default=cfg["valid_examples"])
        p.add_argument(f"--weight-{source_name}", dest=f"weight_{source_name}", type=float, default=cfg["weight"])
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
