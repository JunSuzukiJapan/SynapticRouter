import argparse
from types import SimpleNamespace

import torch

from sra_language_models import MoESRALanguageModel
from tokenizer_utils import build_tokenizer
from train_small_llm import choose_device, evaluate, load_corpus


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location="cpu")
    train_args = SimpleNamespace(**ckpt["args"])
    tokenizer = build_tokenizer(
        getattr(train_args, "tokenizer_type", "byte"),
        getattr(train_args, "tokenizer_name", "cl100k_base"),
    )
    model = MoESRALanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=train_args.dim,
        layers=train_args.layers,
        num_synapses=train_args.synapses,
        k=train_args.k,
        syn_hidden=train_args.syn_hidden,
        max_seq_len=train_args.seq_len,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, train_args, model, tokenizer


def sample_next_token(logits, temperature: float, top_k: int):
    if temperature <= 0:
        return logits.argmax(dim=-1).item()

    probs = torch.softmax(logits / temperature, dim=-1)
    if top_k > 0:
        values, indices = torch.topk(probs, k=min(top_k, probs.numel()))
        values = values / values.sum()
        choice = torch.multinomial(values, 1)
        return indices[choice].item()
    return torch.multinomial(probs, 1).item()


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int, temperature: float, top_k: int):
    tokens = tokenizer.encode(prompt)
    if not tokens:
        raise SystemExit("Prompt is empty after UTF-8 byte encoding.")

    generated = list(tokens)
    max_seq_len = model.pos.num_embeddings
    for _ in range(max_new_tokens):
        ctx = generated[-max_seq_len:]
        x = torch.tensor([ctx], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_token = sample_next_token(logits[0, -1], temperature=temperature, top_k=top_k)
        generated.append(next_token)

    text = tokenizer.decode(generated)
    for stop in ["\nUser:", "\nSystem:", "\nYou>"]:
        idx = text.find(stop, len(prompt))
        if idx != -1:
            text = text[:idx]
    return text


def format_chat_prompt(history, user_message: str):
    lines = ["System: You are a tiny SRA assistant trained on open text.", ""]
    for user, assistant in history:
        lines.append(f"User: {user}")
        lines.append(f"Assistant: {assistant}")
        lines.append("")
    lines.append(f"User: {user_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


def run_chat(model, tokenizer, device: str, max_new_tokens: int, temperature: float, top_k: int):
    history = []
    print("Small SRA chat. Type `exit` to quit.")
    while True:
        try:
            user_message = input("You> ").strip()
        except EOFError:
            print()
            break
        if not user_message:
            continue
        if user_message.lower() in {"exit", "quit"}:
            break

        prompt = format_chat_prompt(history, user_message)
        text = generate(
            model,
            tokenizer,
            prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        reply = text[len(prompt):].strip()
        history.append((user_message, reply))
        print(f"SRA> {reply}\n")


def run_eval(model, train_args, device: str, eval_batches_override: int):
    corpus = load_corpus(train_args)
    eval_batches = eval_batches_override if eval_batches_override > 0 else train_args.eval_batches
    val_loss, usage = evaluate(
        model,
        corpus.valid_tokens,
        train_args.eval_batch_size,
        train_args.seq_len,
        eval_batches,
        device,
    )
    usage_str = "n/a"
    if usage is not None:
        usage_str = ", ".join(f"{v:.2f}" for v in usage.tolist()[: min(8, len(usage))])
    print(f"val_loss={val_loss:.4f}")
    print(f"usage[:8]=[{usage_str}]")


def parse_args():
    p = argparse.ArgumentParser(description="Chat with or inspect a trained small SRA language model.")
    p.add_argument("--model", type=str, default="checkpoints/wikitext2_sra.pt")
    p.add_argument("--device", choices=["auto", "mps", "cpu", "cuda"], default="auto")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--prompt", type=str, default="")
    p.add_argument("--chat", action="store_true")
    p.add_argument("--eval", action="store_true")
    p.add_argument("--eval-batches", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device, args.cpu)
    ckpt, train_args, model, tokenizer = load_checkpoint(args.model, device)
    train_args.device = args.device
    train_args.cpu = args.cpu

    print(f"loaded={args.model} step={ckpt.get('step')} device={device}")

    if args.eval:
        run_eval(model, train_args, device=device, eval_batches_override=args.eval_batches)

    if args.prompt:
        print(
            generate(
                model,
                tokenizer,
                args.prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
        )

    if args.chat:
        run_chat(
            model,
            tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )


if __name__ == "__main__":
    main()
