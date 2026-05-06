"""
Seq2Seq 用データローダー
- opus100 からストリーミングで src/tgt ペアを取得
- ソース・ターゲットを別テンソルとして返す（concat なし）
- 特殊トークン: [LANG_TAG] BOS, [EOS] PAD
"""
import sys
import os
import random
from itertools import cycle
from typing import Iterator, Tuple, List, Dict

import torch
import tiktoken

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ──────────────────────────────────────────────
# 特殊トークン定義（v1/v2 と同一スキーム）
# ──────────────────────────────────────────────
def build_special_tokens(tokenizer: tiktoken.Encoding) -> Dict[str, int]:
    n = tokenizer.n_vocab
    return {
        "[ENG]": n,       # source language: English
        "[JPN]": n + 1,   # source language: Japanese
        "[FRA]": n + 2,   # source language: French
        "[SEP]": n + 3,   # separator (未使用だが互換性のため)
        "[TARGET_ENG]": n + 4,  # decoder BOS: English target
        "[TARGET_JPN]": n + 5,  # decoder BOS: Japanese target
        "[TARGET_FRA]": n + 6,  # decoder BOS: French target
        "[EOS]": n + 7,   # end-of-sequence
        "[PAD]": 0,       # padding (embedding の padding_idx と一致)
    }


# ──────────────────────────────────────────────
# データセット読み込み
# ──────────────────────────────────────────────
def load_opus100_pair(src_lang: str, tgt_lang: str, split: str = "train"):
    """
    opus100 から (src_text, tgt_text) を無限に返すジェネレータ。
    opus100 は常に "en-X" 形式の config 名を持つ。
    """
    from datasets import load_dataset

    # config 名の決定（opus100 は en-X 形式）
    if src_lang == "en":
        config = f"en-{tgt_lang}"
        src_key, tgt_key = "en", tgt_lang
    elif tgt_lang == "en":
        config = f"en-{src_lang}"
        src_key, tgt_key = src_lang, "en"
    else:
        # ja↔fr などはピボット経由になるため、ここでは en を経由して疑似生成しない
        # 直接の opus100 ペアがない場合は en-ja + en-fr から独立にサンプル
        raise ValueError(f"opus100 does not have direct {src_lang}-{tgt_lang} pair. "
                         f"Use en as pivot.")

    ds = load_dataset("opus100", config, split=split, streaming=True, trust_remote_code=False)
    while True:
        for example in ds:
            src_text = example["translation"].get(src_key, "")
            tgt_text = example["translation"].get(tgt_key, "")
            if src_text and tgt_text:
                yield src_text.strip(), tgt_text.strip()


class Seq2SeqDataLoader:
    """
    多言語 Seq2Seq データローダー。
    複数の言語方向を等確率（またはカスタム重みで）ランダムサンプリング。
    """

    # 学習対象の翻訳方向とその重み（双方向を同時に学習）
    LANG_PAIRS = [
        ("en", "fr"),  # EN → FR
        ("fr", "en"),  # FR → EN
        ("en", "ja"),  # EN → JA
        ("ja", "en"),  # JA → EN
    ]
    PAIR_WEIGHTS = [1.0, 1.0, 1.0, 1.0]  # 均等サンプリング

    def __init__(
        self,
        src_max_len: int = 80,
        tgt_max_len: int = 80,
        batch_size: int = 32,
        device: str = "cpu",
        split: str = "train",
    ):
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.batch_size = batch_size
        self.device = device

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.sp = build_special_tokens(self.tokenizer)
        self.vocab_size = self.tokenizer.n_vocab + 100

        self.PAD = self.sp["[PAD]"]
        self.EOS = self.sp["[EOS]"]

        print(f"Loading {len(self.LANG_PAIRS)} language pair streams ({split})...")
        self._iters: Dict[Tuple[str, str], Iterator] = {}
        for src_lang, tgt_lang in self.LANG_PAIRS:
            try:
                it = load_opus100_pair(src_lang, tgt_lang, split)
                self._iters[(src_lang, tgt_lang)] = it
                print(f"  ✓ {src_lang.upper()} → {tgt_lang.upper()}")
            except Exception as e:
                print(f"  ✗ {src_lang.upper()} → {tgt_lang.upper()}: {e}")

        self._available_pairs = list(self._iters.keys())
        self._weights = [
            self.PAIR_WEIGHTS[self.LANG_PAIRS.index(p)]
            for p in self._available_pairs
        ]

    # ──────────────────────────────────────────────
    _LANG_CODE = {"en": "ENG", "ja": "JPN", "fr": "FRA"}

    def _bos_for(self, tgt_lang: str) -> int:
        code = self._LANG_CODE[tgt_lang]
        return self.sp[f"[TARGET_{code}]"]

    def _src_tag(self, src_lang: str) -> int:
        code = self._LANG_CODE[src_lang]
        return self.sp[f"[{code}]"]

    def _encode_pair(
        self,
        src_lang: str,
        tgt_lang: str,
        src_text: str,
        tgt_text: str,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Returns:
            src_ids      : [SRC_TAG, t1, t2, ..., tN]               (max src_max_len)
            tgt_in_ids   : [TGT_BOS, t1, t2, ..., tM]               (max tgt_max_len)
            tgt_lab_ids  : [t1, t2, ..., tM, EOS]  or -100 for PAD  (max tgt_max_len)
        """
        src_tag = [self._src_tag(src_lang)]
        src_ids = (src_tag + self.tokenizer.encode(src_text))[: self.src_max_len]

        tgt_bos = [self._bos_for(tgt_lang)]
        tgt_tokens = self.tokenizer.encode(tgt_text)

        # decoder input: BOS + target[0..T-2]  (length ≤ tgt_max_len)
        tgt_in_ids = (tgt_bos + tgt_tokens)[: self.tgt_max_len]

        # decoder label: target[0..T-1] + EOS  (length ≤ tgt_max_len)
        tgt_lab_ids = (tgt_tokens + [self.EOS])[: self.tgt_max_len]

        # tgt_in と tgt_lab の長さを揃える
        L = min(len(tgt_in_ids), len(tgt_lab_ids))
        tgt_in_ids = tgt_in_ids[:L]
        tgt_lab_ids = tgt_lab_ids[:L]

        return src_ids, tgt_in_ids, tgt_lab_ids

    def _pad(self, seqs: List[List[int]], pad_id: int) -> List[List[int]]:
        max_len = max(len(s) for s in seqs)
        return [s + [pad_id] * (max_len - len(s)) for s in seqs]

    # ──────────────────────────────────────────────
    def get_batch(self):
        """
        Returns:
            src      : (B, S) LongTensor
            tgt_in   : (B, T) LongTensor   ← decoder input
            tgt_lab  : (B, T) LongTensor   ← labels (-100 for padding)
            pairs    : list of (src_lang, tgt_lang) for logging
        """
        batch_src, batch_tgt_in, batch_tgt_lab = [], [], []
        batch_pairs = []

        while len(batch_src) < self.batch_size:
            (src_lang, tgt_lang) = random.choices(
                self._available_pairs, weights=self._weights, k=1
            )[0]
            src_text, tgt_text = next(self._iters[(src_lang, tgt_lang)])

            # 空文や極端に長い文はスキップ
            if not src_text or not tgt_text:
                continue
            src_ids, tgt_in, tgt_lab = self._encode_pair(
                src_lang, tgt_lang, src_text, tgt_text
            )
            if len(src_ids) < 2 or len(tgt_in) < 2:
                continue

            batch_src.append(src_ids)
            batch_tgt_in.append(tgt_in)
            batch_tgt_lab.append(tgt_lab)
            batch_pairs.append((src_lang, tgt_lang))

        src_tensor = torch.tensor(
            self._pad(batch_src, self.PAD), dtype=torch.long, device=self.device
        )
        tgt_in_tensor = torch.tensor(
            self._pad(batch_tgt_in, self.PAD), dtype=torch.long, device=self.device
        )
        # label の PAD 位置は -100（CrossEntropyLoss の ignore_index）
        tgt_lab_padded = self._pad(batch_tgt_lab, -100)
        tgt_lab_tensor = torch.tensor(tgt_lab_padded, dtype=torch.long, device=self.device)

        return src_tensor, tgt_in_tensor, tgt_lab_tensor, batch_pairs
