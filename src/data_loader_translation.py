import torch
from datasets import load_dataset
import tiktoken
import random

LANG_TAGS = {"en": "[ENG]", "ja": "[JPN]", "fr": "[FRA]"}
TARGET_TAGS = {"en": "[TARGET_ENG]", "ja": "[TARGET_JPN]", "fr": "[TARGET_FRA]"}

class MultilingualDataLoader:
    def __init__(self, dataset_name="iwslt2017", seq_len=128, batch_size=32, device="cpu"):
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        
        # 多言語対応のトークナイザとして cl100k_base を使用
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.vocab_size = self.tokenizer.n_vocab + 100 # 特殊トークン用のバッファ
        
        # サポートする言語ペア
        self.supported_pairs = [
            ("iwslt2017-en-ja", "en", "ja"), # iwslt2017 の en-ja サブセット
            ("iwslt2017-en-fr", "en", "fr")  # iwslt2017 の en-fr サブセット
        ]
        
        self.datasets = {}
        
    def prepare_datasets(self, split="train", streaming=True):
        """HuggingFace Datasets から指定したスプリットのデータを準備する"""
        print(f"Loading datasets for split: {split} (streaming={streaming})")
        for subset, lang1, lang2 in self.supported_pairs:
            # iwslt2017 は subset として 'iwslt2017-en-ja' などを指定
            # 言語方向が双方向で含まれるため、後でランダムに src->tgt を決める
            ds = load_dataset(self.dataset_name, subset, split=split, streaming=streaming)
            self.datasets[subset] = {
                "dataset": iter(ds) if streaming else ds,
                "streaming": streaming,
                "langs": (lang1, lang2)
            }
            
    def _encode_pair(self, src_lang, tgt_lang, src_text, tgt_text):
        """翻訳ペアをプロンプトとターゲットに変換してトークナイズ"""
        # フォーマット: [SRC_LANG] src_text [SEP] [TARGET_TGT_LANG] tgt_text [EOS]
        prompt_str = f"{LANG_TAGS.get(src_lang, f'[{src_lang.upper()}]')} {src_text} [SEP] {TARGET_TAGS.get(tgt_lang, f'[TARGET_{tgt_lang.upper()}]')} "
        target_str = f"{tgt_text} [EOS]"
        
        prompt_tokens = self.tokenizer.encode(prompt_str, allowed_special="all")
        target_tokens = self.tokenizer.encode(target_str, allowed_special="all")
        
        all_tokens = prompt_tokens + target_tokens
        if len(all_tokens) > self.seq_len + 1:
            all_tokens = all_tokens[:self.seq_len + 1]
            
        x_seq = all_tokens[:-1]
        y_seq = all_tokens[1:]
        
        # y_seq のうち、prompt部分のlossは計算しないため -100 で埋める
        prompt_len = len(prompt_tokens) - 1 # 最後のスペース等は予測に含めるかもしれないが基本はprompt部分を無視
        y_masked = [-100] * prompt_len + y_seq[prompt_len:]
        
        return x_seq, y_masked

    def get_batch(self):
        """1バッチ分のデータを生成して返す"""
        x_batch = torch.full((self.batch_size, self.seq_len), 0, dtype=torch.long)
        y_batch = torch.full((self.batch_size, self.seq_len), -100, dtype=torch.long)
        
        batch_pairs = []
        
        for i in range(self.batch_size):
            # ランダムに言語ペアのデータセットを選ぶ
            subset_name = random.choice(list(self.datasets.keys()))
            ds_info = self.datasets[subset_name]
            lang1, lang2 = ds_info["langs"]
            
            # ストリーミングから1件取得
            try:
                if ds_info["streaming"]:
                    item = next(ds_info["dataset"])
                else:
                    # Non-streaming の場合はランダムインデックス
                    idx = random.randint(0, len(ds_info["dataset"]) - 1)
                    item = ds_info["dataset"][idx]
            except StopIteration:
                # イテレータが枯渇したら再初期化（簡易実装）
                split = "train" # fallback
                ds_info["dataset"] = iter(load_dataset(self.dataset_name, subset_name, split=split, streaming=True))
                item = next(ds_info["dataset"])
                
            translation = item["translation"]
            
            # 方向をランダムに決定 (例: en->ja or ja->en)
            if random.random() < 0.5:
                src_lang, tgt_lang = lang1, lang2
            else:
                src_lang, tgt_lang = lang2, lang1
                
            src_text = translation[src_lang]
            tgt_text = translation[tgt_lang]
            
            batch_pairs.append(f"{src_lang}->{tgt_lang}")
            
            x_seq, y_masked = self._encode_pair(src_lang, tgt_lang, src_text, tgt_text)
            
            x_batch[i, :len(x_seq)] = torch.tensor(x_seq, dtype=torch.long)
            y_batch[i, :len(y_masked)] = torch.tensor(y_masked, dtype=torch.long)
            
        return x_batch.to(self.device), y_batch.to(self.device), batch_pairs

if __name__ == "__main__":
    # テスト実行用
    loader = MultilingualDataLoader(seq_len=64, batch_size=2, device="cpu")
    print("Preparing datasets (streaming)...")
    loader.prepare_datasets(split="train", streaming=True)
    
    x, y, pairs = loader.get_batch()
    print("Batch X shape:", x.shape)
    print("Batch Y shape:", y.shape)
    print("Pairs:", pairs)
    print("Sample X[0]:", x[0])
