# SRA Development Roadmap

このロードマップは、Synaptic Routing Architecture (SRA) を「大規模LLMそのものを作る計画」ではなく、**動的ルーティングとモジュール制御を検証する研究プロトタイプ**として前進させるための実行順序を定義する。

## 0. Current Position

SRA は、Top-k ルーティングされた小さな expert/synapse を使い、タスクごとの専門化、Hot-Swap、削除、ルーティング可視化を検証している。

現時点の強み:

- 小規模タスクで、タスク種別に応じたルーティング分化が観測されている。
- `add_synapses`, `clear_synapses`, `pop_synapses`, `allowed_synapses_mask` など、Hot-Swap/隔離制御に必要なAPIが実装されている。
- 参照実装とbatched実装の等価性テストがある。

現時点の弱み:

- 多くの結果は toy task または小規模データでの予備実験であり、汎用LLM性能を主張できる段階ではない。
- seed違い、ベースライン、アブレーション、評価スクリプトの再現性がまだ不足している。
- ドキュメントには、現在の実証範囲に対して強すぎる表現が残っている。

## 1. Make Claims Smaller, Evidence Stronger

最初の優先事項は、外向きの主張を研究スコープに合わせ、評価を再現可能にすること。

### Scope

- READMEと主要レポートの表現を「証明」から「観測」「予備検証」「小規模実験での結果」に寄せる。
- 各実験に、データ、seed、モデルサイズ、step数、比較対象、評価指標を明記する。
- `tmp/` 配下の実験用テストが `pytest` 全体実行を壊さないよう、テスト収集対象を整理する。

### Acceptance Criteria

- `pytest -q tests` が安定して通る。
- READMEから、現時点で未検証の大規模LLM性能主張が除去されている。
- 少なくとも algorithmic routing と language routing の2本について、再実行コマンドと評価指標が明記されている。

## 2. Focus on Hotswappable Modular LLM

次に、SRAの差別化を「単なるMoE」ではなく、後付け・削除・干渉制御に置く。

### Scope

- shared trunk + routed synapse の条件で、ドメイン別 expert を後から追加する最小実験を固定する。
- 追加前後で既存タスクの loss/logits がどれだけ変化するかを測る。
- `allowed_synapses_mask` による明示的隔離と、マスクなし追記型ルーターの差分を比較する。
- `clear_synapses` と `pop_synapses` の安全性をユニットテスト化する。

### Acceptance Criteria

- Hot-Swapの最小再現スクリプトが1コマンドで動く。
- 追加前後の既存タスク劣化率、追加タスク性能、削除後の残存影響が数値で出る。
- LoRA/adapter または通常fine-tuningとの比較方針が明文化されている。

## 3. Move One Experiment Toward Practicality

最後に、toy task から一段だけ実用寄りのデータへ移す。ただし、目標は商用品質ではなく、SRAの性質が実データで残るかを見ること。

### Candidate Experiments

- 小型の固定trunk言語モデルに、Code/Math/Text の domain expert を追加する。
- opus100などの翻訳データで、言語ペア別ルーティングと追加/削除の影響を見る。
- instruction tuning風の小規模データで、一般応答expertと専門応答expertを分離できるか確認する。

### Acceptance Criteria

- Dense/standard Transformer/MoE/adapter のうち最低1つと比較する。
- ルーティング分離度、既存能力劣化、新規能力獲得、削除後の残存率を同じ表で出す。
- 「性能が高い」だけでなく、「モジュール制御ができるか」を主評価にする。

## Working Rule

今後のドキュメントでは、以下の表現ルールを使う。

- 単一seedや小規模実験では「証明」ではなく「観測」「示唆」「予備結果」を使う。
- 「完全」「ゼロ」「実用レベル」は、評価条件、許容誤差、比較対象が明示されている場合だけ使う。
- 生物学的比喩は導入説明に留め、技術的結論は loss、accuracy、BLEU、routing entropy、Jaccard similarity、interference delta などの測定値で述べる。
