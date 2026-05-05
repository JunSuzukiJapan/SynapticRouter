import torch
import torch.nn.functional as F
import argparse
import tiktoken
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sra_language_models import MoESRALanguageModel

LANG_TAGS = {"en": "[ENG]", "ja": "[JPN]", "fr": "[FRA]"}

TEST_SENTENCES = {
    "en": [
        "I eat apples.",
        "She reads books.",
        "They play games.",
        "The dog chases the cat."
    ],
    "ja": [
        "私 は りんご を 食べる。",
        "彼女 は 本 を 読む。",
        "彼ら は ゲーム を する。",
        "犬 は 猫 を 追いかける。"
    ],
    "fr": [
        "Je mange des pommes.",
        "Elle lit des livres.",
        "Ils jouent aux jeux.",
        "Le chien chasse le chat."
    ]
}

def analyze_routing(args):
    device = "cpu"
    tokenizer = tiktoken.get_encoding("cl100k_base")
    vocab_size = tokenizer.n_vocab + 100
    
    # Load model
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
    
    try:
        ckpt = torch.load(args.model_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"Loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Warning: Model file {args.model_path} not found. Using untrained model for testing.")

    model.eval()
    
    layer_usage_by_lang = {lang: [] for lang in TEST_SENTENCES.keys()}
    
    with torch.no_grad():
        for lang, sentences in TEST_SENTENCES.items():
            lang_usages = []
            for text in sentences:
                prompt_str = f"{LANG_TAGS[lang]} {text}"
                tokens = tokenizer.encode(prompt_str, allowed_special="all")
                x = torch.tensor([tokens], dtype=torch.long, device=device)
                
                _, router_logits = model(x, dense=False)
                
                # router_logits: list of (B, T, num_synapses)
                # Calculate mean usage across sequence length for each layer
                seq_usages = []
                for r_logits in router_logits:
                    probs = F.softmax(r_logits, dim=-1) # (1, T, num_synapses)
                    mean_prob = probs.mean(dim=1).squeeze(0) # (num_synapses,)
                    seq_usages.append(mean_prob.numpy())
                lang_usages.append(seq_usages)
                
            # Average over all test sentences for this language
            lang_usages = np.array(lang_usages) # (num_sentences, num_layers, num_synapses)
            avg_lang_usage = lang_usages.mean(axis=0) # (num_layers, num_synapses)
            layer_usage_by_lang[lang] = avg_lang_usage

    # Plotting
    num_layers = args.layers
    fig, axes = plt.subplots(num_layers, len(TEST_SENTENCES), figsize=(5 * len(TEST_SENTENCES), 4 * num_layers))
    if num_layers == 1:
        axes = np.expand_dims(axes, axis=0)
        
    for layer in range(num_layers):
        for i, lang in enumerate(TEST_SENTENCES.keys()):
            ax = axes[layer, i]
            usage = layer_usage_by_lang[lang][layer]
            sns.barplot(x=np.arange(args.synapses), y=usage, ax=ax, palette="viridis")
            ax.set_title(f"Layer {layer+1} - {lang.upper()}")
            ax.set_xlabel("Synapse ID")
            ax.set_ylabel("Activation Probability")
            ax.set_ylim(0, 1.0)
            
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved routing analysis plot to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, default="sra_translation_large.pt")
    p.add_argument("--output", type=str, default="translation_routing_analysis.png")
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--synapses", type=int, default=16)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--syn-hidden", type=int, default=1024)
    analyze_routing(p.parse_args())
