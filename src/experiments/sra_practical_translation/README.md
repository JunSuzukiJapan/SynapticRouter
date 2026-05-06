# SRA 実務レベル多言語翻訳実験

このディレクトリには、SRAモデルをスケールアップして実務レベルの多言語翻訳精度を目指した実験スクリプト群が含まれています。

## 実験概要

- **目的**: opus100 コーパス（英・仏・日）を用いた本格的な多言語翻訳のSRA実験
- **モデル**: MoESRALanguageModel（69.4M パラメータ）
- **学習**: 15,000ステップ / MPS（Apple Silicon）/ 3.11時間

## 既存実験との違い

| | 既存実験 (`src/train_mtl_trans.py`) | 本実験 |
|---|---|---|
| データ | 合成パラレルコーパス（小規模） | opus100（実コーパス・ストリーミング） |
| ステップ数 | 300 | 15,000 |
| モデルサイズ | 小（dim=128） | 中（dim=256, 69.4M params） |
| 目的 | SVO/SOV分化・ゼロショットの検証 | 実翻訳精度の追求 |
| 対応レポート | `docs/dev/multilingual_translation_routing_analysis.md` | 本実験ログ |

## ファイル構成

| ファイル | 役割 |
|---|---|
| `train_translation_large.py` | 本格学習スクリプト（コサインLR・フェーズ管理） |
| `data_loader_translation.py` | opus100ストリーミングデータローダー |
| `analyze_translation_routing.py` | シナプス使用率ヒートマップ生成 |
| `analyze_translation_accuracy.py` | 自己回帰生成・BLEU評価 |
| `run_analysis.py` | シナプス使用率・コサイン類似度の数値分析 |

## 実行方法

プロジェクトルートから実行（`src/` が `PYTHONPATH` に通っている必要あり）：

```bash
# 学習
cd src/experiments/sra_practical_translation
python train_translation_large.py \
    --steps 15000 --batch-size 32 --seq-len 96 \
    --dim 256 --layers 4 --synapses 16 --k 4 --syn-hidden 512 \
    --save ../../../sra_translation_v2.pt

# ルーティング分析
python analyze_translation_routing.py \
    --model-path ../../../sra_translation_v2.pt \
    --output ../../../translation_routing_v2.png \
    --seq-len 96 --dim 256 --layers 4 --synapses 16 --k 4 --syn-hidden 512

# 翻訳精度評価
python analyze_translation_accuracy.py
```

## 分析結果のサマリー

- **最終損失**: 3.58（11.65 → 3.58、68%削減）
- **PPL改善**: EN→FR: 177→37（4.8倍）, EN→JA: 28→13（2.2倍）
- **習得済み能力**: 言語タグによる出力言語切り替え、定型表現（Merci: 82%、Bonjour: PPL=1.5）
- **未習得**: 動詞・目的語の翻訳忠実性（GPU・50K steps以上で改善見込み）
