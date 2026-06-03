import argparse
import os
import random
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F

from sra_experiment import make_optimizer
from sra_language_models import MoESRALanguageModel
from tokenizer_utils import build_tokenizer
from train_small_llm import choose_device


IGNORE_INDEX = -100


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location="cpu")
    args = ckpt["args"]
    tokenizer = build_tokenizer(args.get("tokenizer_type", "byte"), args.get("tokenizer_name", "cl100k_base"))
    model = MoESRALanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=args["dim"],
        layers=args["layers"],
        num_synapses=args["synapses"],
        k=args["k"],
        syn_hidden=args["syn_hidden"],
        max_seq_len=args["seq_len"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt, args, model, tokenizer


def format_prompt(user_text: str) -> str:
    return (
        "System: You are a helpful assistant.\n\n"
        f"User: {user_text}\n"
        "Assistant: "
    )


def import_datasets():
    from datasets import DownloadConfig, load_dataset
    return load_dataset, DownloadConfig


def build_no_robots_examples(split: str, max_examples: int, local_only: bool) -> List[Tuple[str, str]]:
    if max_examples == 0:
        return []
    load_dataset, DownloadConfig = import_datasets()
    ds = load_dataset(
        "HuggingFaceH4/no_robots",
        split=split,
        download_config=DownloadConfig(local_files_only=local_only),
    )
    examples = []
    for row in ds:
        msgs = row["messages"]
        history = []
        for msg in msgs:
            role = msg["role"]
            content = msg["content"].strip()
            if not content:
                continue
            if role == "user":
                history.append(("user", content))
            elif role == "assistant" and history:
                prompt_turns = []
                for h_role, h_content in history:
                    tag = "User" if h_role == "user" else "Assistant"
                    prompt_turns.append(f"{tag}: {h_content}")
                prompt = "System: You are a helpful assistant.\n\n" + "\n".join(prompt_turns) + "\nAssistant: "
                examples.append((prompt, content))
                history.append(("assistant", content))
            if max_examples > 0 and len(examples) >= max_examples:
                return examples
    return examples


def build_oasst2_examples(split: str, max_examples: int, local_only: bool) -> List[Tuple[str, str]]:
    if max_examples == 0:
        return []
    load_dataset, DownloadConfig = import_datasets()
    ds = load_dataset(
        "OpenAssistant/oasst2",
        split=split,
        download_config=DownloadConfig(local_files_only=local_only),
    )
    rows = [row for row in ds]
    by_id = {row["message_id"]: row for row in rows}
    examples = []
    for row in rows:
        if row["role"] != "assistant":
            continue
        if row.get("lang") != "en" or not row.get("review_result") or row.get("deleted"):
            continue
        parent_id = row.get("parent_id")
        if not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        if parent["role"] != "prompter" or parent.get("lang") != "en":
            continue
        user_text = parent["text"].strip()
        assistant_text = row["text"].strip()
        if not user_text or not assistant_text:
            continue
        examples.append((format_prompt(user_text), assistant_text))
        if max_examples > 0 and len(examples) >= max_examples:
            break
    return examples


def encode_example(tokenizer, prompt: str, response: str, seq_len: int):
    prompt_tokens = tokenizer.encode(prompt)
    response_tokens = tokenizer.encode(response)
    full_tokens = prompt_tokens + response_tokens
    if len(response_tokens) < 2:
        return None

    # Keep the full response, truncate older prompt context from the left if needed.
    max_total = seq_len + 1
    if len(full_tokens) > max_total:
        keep_prompt = max(0, max_total - len(response_tokens))
        prompt_tokens = prompt_tokens[-keep_prompt:]
        full_tokens = prompt_tokens + response_tokens
        if len(full_tokens) > max_total:
            response_tokens = response_tokens[: max_total - len(prompt_tokens)]
            full_tokens = prompt_tokens + response_tokens

    if len(full_tokens) < 2:
        return None

    response_start = len(prompt_tokens)
    x = full_tokens[:-1]
    y = full_tokens[1:]
    labels = [IGNORE_INDEX] * len(y)
    for idx in range(max(0, response_start - 1), len(y)):
        labels[idx] = y[idx]
    return x, labels


def pad_batch(batch, seq_len: int, device: str):
    x = torch.zeros((len(batch), seq_len), dtype=torch.long)
    labels = torch.full((len(batch), seq_len), IGNORE_INDEX, dtype=torch.long)
    for i, (tokens, target) in enumerate(batch):
        n = min(seq_len, len(tokens))
        x[i, :n] = torch.tensor(tokens[:n], dtype=torch.long)
        labels[i, :n] = torch.tensor(target[:n], dtype=torch.long)
    return x.to(device), labels.to(device)


def make_batches(examples, tokenizer, seq_len: int):
    encoded = []
    for prompt, response in examples:
        item = encode_example(tokenizer, prompt, response, seq_len)
        if item is not None:
            encoded.append(item)
    return encoded


@torch.no_grad()
def evaluate(model, batches, batch_size: int, seq_len: int, device: str):
    model.eval()
    total_loss = 0.0
    count = 0
    for start in range(0, len(batches), batch_size):
        batch = batches[start:start + batch_size]
        x, labels = pad_batch(batch, seq_len, device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def generate_reply(model, tokenizer, prompt: str, device: str, max_new_tokens: int):
    model.eval()
    generated = tokenizer.encode(prompt)
    max_seq_len = model.pos.num_embeddings
    for _ in range(max_new_tokens):
        x = torch.tensor([generated[-max_seq_len:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_token = torch.argmax(logits[0, -1], dim=-1).item()
        generated.append(next_token)
    text = tokenizer.decode(generated)
    return text[len(prompt):].strip()


def parse_args():
    p = argparse.ArgumentParser(description="Conversational SFT for the small SRA LM using open datasets.")
    p.add_argument("--base-model", type=str, default="checkpoints/wikitext2_sra.pt")
    p.add_argument("--save", type=str, default="checkpoints/wikitext2_sra_chat.pt")
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-no-robots", type=int, default=8000)
    p.add_argument("--max-train-oasst2", type=int, default=12000)
    p.add_argument("--max-valid-no-robots", type=int, default=500)
    p.add_argument("--max-valid-oasst2", type=int, default=1000)
    p.add_argument("--local-datasets-only", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device, args.cpu)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ckpt, train_args, model, tokenizer = load_checkpoint(args.base_model, device)
    seq_len = train_args["seq_len"]

    print("Loading SFT datasets...")
    train_examples = []
    train_examples += build_no_robots_examples("train", args.max_train_no_robots, args.local_datasets_only)
    train_examples += build_oasst2_examples("train", args.max_train_oasst2, args.local_datasets_only)
    valid_examples = []
    valid_examples += build_no_robots_examples("test", args.max_valid_no_robots, args.local_datasets_only)
    valid_examples += build_oasst2_examples("validation", args.max_valid_oasst2, args.local_datasets_only)

    train_batches = make_batches(train_examples, tokenizer, seq_len)
    valid_batches = make_batches(valid_examples, tokenizer, seq_len)
    random.shuffle(train_batches)

    print(f"device={device} base={args.base_model} train_examples={len(train_batches)} valid_examples={len(valid_batches)}")

    optimizer = make_optimizer(model, args.lr)
    global_step = 0
    start_time = time.time()

    for step in range(1, args.steps + 1):
        model.train()
        loss_acc = 0.0
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            idx = ((step - 1) * args.grad_accum + micro) % len(train_batches)
            batch = [train_batches[(idx + j) % len(train_batches)] for j in range(args.batch_size)]
            x, labels = pad_batch(batch, seq_len, device)
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            ) / args.grad_accum
            loss.backward()
            loss_acc += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        global_step += 1

        if step % args.log_every == 0 or step == 1:
            val_loss = evaluate(model, valid_batches, args.batch_size, seq_len, device)
            sample_prompt = format_prompt("Tell me about Albert Einstein.")
            sample_reply = generate_reply(model, tokenizer, sample_prompt, device, max_new_tokens=80)
            print(f"step={step:4d} train_loss={loss_acc:.4f} val_loss={val_loss:.4f}")
            print(f"sample_reply={sample_reply}")

        if args.save_every > 0 and step % args.save_every == 0:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "args": {
                        **train_args,
                        "sft_base_model": args.base_model,
                        "sft_steps": step,
                    },
                },
                args.save,
            )
            print(f"saved checkpoint: {args.save}")

    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save(
        {
            "step": args.steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": {
                **train_args,
                "sft_base_model": args.base_model,
                "sft_steps": args.steps,
            },
        },
        args.save,
    )
    print(f"final saved: {args.save}")
    print(f"elapsed={time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
