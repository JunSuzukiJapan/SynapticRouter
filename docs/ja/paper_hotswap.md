# Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion via Synaptic Routing Architecture

**Jun Suzuki**, Independent Researcher

## Abstract
巨大な言語モデル（LLM）は、すべての知識を単一のパラメータ空間に密に格納するモノリシック構造を持つため、特定の知識の追加・削除が困難であり、運用上の柔軟性に重大な制約を課している。本稿では、Synaptic Routing Architecture（SRA）のモジュール性を活用し、モジュールの動的な抜き差し——すなわち **Hot-Swap**——を実現する手法を提案・検証する。Hot-Swapとは、事前学習済みベースモデルに対して独立学習した特化シナプスを **一切の再学習なしで物理的に挿入（Plug-In）** し、不要になれば **外科的に抜去（Unplug）** できる、双方向のモジュール交換操作である。実験の結果、ベクトルデータベースの事前フィルタリング技術に着想を得たハードマスク機構により、挿入後もベースモデルの出力が小数点以下まで完全に一致する **Zero Forgetting** を達成した。加えて、抜去操作においてコサイン類似度におけるゼロベクトルの「ブラックホール問題」を発見・解決し、「学習→挿入→抜去→再利用」というモジュラーAIの完全なライフサイクルを実現した。

## 1. Introduction

### 1.1 モノリスモデルの運用上の限界
「Attention Is All You Need」以降、Transformerアーキテクチャは自然言語処理をはじめとする多くの領域で支配的な地位を確立した。しかし、数百億〜数千億パラメータを持つモノリシックなLLMは、その運用において以下の深刻な問題を抱えている。

1. **破滅的忘却（Catastrophic Forgetting）**: 汎用モデルに特定ドメイン（例：社内規程、専門コード）を追加学習させると、元の汎用能力が破壊・劣化する。
2. **学習コストの肥大化**: 新知識の追加ごとにモデル全体（またはLoRA等のアダプタ）の再学習・マージが必要となり、複数チームでの完全並行開発が困難である。
3. **知識削除の不可能性**: 特定の知識だけを忘れさせる「Machine Unlearning」は、パラメータが複雑に絡み合ったモノリスモデルでは至難の業であり、再学習を試みると無関係な能力まで破壊される。

### 1.2 本稿の貢献
私が提案するSynaptic Routing Architecture（SRA）[Suzuki, 2026]は、極小の独立モジュール（シナプス）と軽量なルーターから構成されるスパースなアーキテクチャである。先行研究ではルーターの自律的タスク分離能力を実証したが、本稿ではSRAのモジュール性がもたらす **運用上の革新**——モジュールの挿入と抜去を自在に行う **Hot-Swap**——に焦点を当て、以下の3つの貢献を報告する。

1.**Hot-Swap: Plug-In（挿入）**: 独立学習した特化モデルの重みテンソルを、ベースモデルの空きスロットに物理コピーするだけでデプロイが完了する手法の実装と検証。
2.**Hot-Swap: Unplug（抜去）**: 物理的な取り外し（`pop_synapses`）とゼロクリアによるパージ（`clear_synapses`）という2つの削除APIの設計と、コサイン類似度における「ゼロベクトルのブラックホール問題」の発見・解決。
3.**Zero Forgetting の実験的証明**: ベクトルDBの事前フィルタリング（Pre-filtering）に着想を得たハードマスク機構により、挿入・抜去の前後でベースモデルの出力Lossが小数点以下まで完全に一致することの実証。

## 2. Background: SRA Architecture

SRA（Synaptic Routing Architecture）は生物の脳を模倣した動的でスパースなアーキテクチャである。ここでは、Hot-Swapの理解に必要な構成要素を概説する（詳細は[Suzuki, 2026]を参照）。

### 2.1 Router
SRAの心臓部であるルーターは、Attention機構を持たない **単一の線形層（Linear Layer）** である。入力の隠れ状態 $h$ と各シナプスの特徴ベクトル（埋め込み）$e_i$ のコサイン類似度を計算し、Top-k個のシナプスを選択する。

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

ここで $\alpha$ はスケーリング係数である。

### 2.2 Tiny Synapses
各シナプスは小型のMulti-Head AttentionとMLPからなる独立した極小モジュールである。ルーターによって選択されたシナプスのみが計算を実行するため、高い計算効率を実現する。

### 2.3 共有トランク（Shared Trunk）
Hot-Swapの成立条件として極めて重要なのが、 **共有トランク方式** である。すべての特化シナプスは、同一の事前学習済みベースモデル（Embedding層、Attention層）から派生し、シナプス部分のみを独立学習する。これにより、モデル間の内部ベクトル表現の乖離（Representation Divergence）を防ぎ、物理コピーによるシナプス移植を可能にする。

## 3. Hot-Swap: Plug-In（モジュールの挿入）

Hot-Swapの第一の操作は、独立学習した特化モジュールをベースモデルに **挿入（Plug-In）** することである。本章では、この挿入操作がPyTorchのテンソル操作のみで完結する極めてシンプルなプロセスであることを示す。

### 3.1 手法

```python
# hotswap_model: 本番環境のベースモデル（空きスロットを追加済み）
# plugin_math: 数学チームが独立学習した特化LLM

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # ルーターの埋め込みベクトルをコピー
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Expert (TinySynapse) の重み (w1, w2) をコピー
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

ベースモデルのテンソルの特定インデックス（空きスロット）に対して、学習済み特化モデルのテンソルを直接代入するだけである。SRAは共有トランク方式によりベースモデルの共通知識（Attention層等）を完全に凍結した状態でシナプスのみを学習するため、この物理コピーが成立する。

### 3.2 独立並行開発の実現
この方式により、「コードチーム」と「数学チーム」が同一のベースモデルを元に、互いに一切干渉せず自分たちの特化シナプスだけを独立学習できる。学習完了後は、重みテンソルを本番環境のベースモデルの空きスロットにメモリコピーするだけでデプロイが完了する。

## 4. Zero Forgetting: ベクトルDB Pre-filteringに着想を得たハードマスク機構

### 4.1 課題：ルーターの混同
単純にテンソルを物理コピーしただけでは、ルーターが新旧のシナプスを混同し、ベースモデルの出力が変化してしまう危険がある。

### 4.2 ベクトルDBにおけるメタデータ・フィルタリング
PineconeやWeaviateなどのモダンなベクトルデータベースでは、コサイン類似度による意味検索に加えて、メタデータによるフィルタリングが重要な役割を果たす。

-**Post-filtering（事後フィルタリング）**: Top-K検索後に条件外の結果を除外する方式。K件が足りなくなる「K-NN枯渇問題」が生じやすい。
-**Pre-filtering（事前フィルタリング）**: 検索 **前** にメタデータマスクで検索空間を制限し、条件に合致する候補のみでTop-K検索を行う方式。ノイズが完全に排除される。

### 4.3 SRAの事前ハードマスク
SRAのルーターの本質は、入力ベクトルと各シナプスの埋め込みベクトルの内積を計算する **インメモリのベクトル検索エンジン（Maximum Inner Product Search: MIPS）** である。

私は、ベクトルDBのPre-filteringをルーターの順伝播に直接組み込んだ。推論時に「現在のタスクに許可されたシナプスの集合」を示すメタデータマスクをモデルに渡す。

```python
# Router層の順伝播
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtering: 許可されていないシナプスのロジットを-∞にする
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-Kルーティング
vals, idx = torch.topk(logits, k, dim=-1)
```

この `masked_fill` によるPre-filteringにより、ルーターは許可されたシナプスの中からのみ専門家を選択する。別モデルの重みがいくら共存していても、計算グラフ上では完全に無視されるため、 **ベースモデルのLossが合体前と小数点以下まで完全に一致する（数学的にゼロの干渉）** 。

### 4.4 実験結果
ベースモデル（Code/Math/Text の3ドメイン言語モデル）に対し、独立学習した特化シナプスをHot-Swapした前後で、ベースモデルのValidation Lossを比較した。結果、合体前と合体後でLossが小数点以下まで完全に一致し、Zero Forgettingが実証された。

## 5. Hot-Swap: Unplug（モジュールの抜去）

Hot-Swapの第二の操作は、不要になったモジュールをベースモデルから **抜去（Unplug）** することである。知識を「挿せる」のであれば、不要になった知識を「抜く」ことも求められる。モノリスモデルにおける Machine Unlearning は、パラメータが複雑に絡み合っているため極めて困難であるが、SRAのモジュール構造は、この問題を物理的操作で解決する。

### 5.2 アプローチ①：物理的取り外し（pop_synapses）
後からHot-Swapで追加したシナプスが不要になった場合、テンソルの末尾からスライスして物理的に切り捨てる。

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**利点**: VRAMの使用量が物理的に縮小され、モデルを追加前の状態に完全に復元できる。OSのドライバをアンインストールするように、AIの脳のパーツを物理的に取り外すことができる。

### 5.3 アプローチ②：ゼロクリアによるパージ（clear_synapses）
末尾ではなく中間インデックスのシナプスを削除したい場合、物理的な削除はインデックスのズレを引き起こし、メタデータマスクによる制御システムが崩壊する。そこで、シナプスの中身をゼロクリアして「空きスロット化」する。

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

テンソルのサイズを変えずにスロットの中身だけを無効化するため、インデックスの整合性は完璧に保たれる。空きスロットには後から新しいシナプスを上書き（Hot-Swap）して再利用できる。

## 6. The Cosine Similarity Trap: ゼロベクトルのブラックホール問題

### 6.1 問題の発見
ゼロクリアによるパージを実装したところ、 **出力が完全に崩壊する** という深刻なバグに遭遇した。

### 6.2 原因分析
SRAのルーターはコサイン類似度を用いてルーティングを行う。ゼロクリアされたシナプスの埋め込みベクトルは $\mathbf{0}$ となり、正規化しても $\mathbf{0}$ のままである。任意の入力ベクトル $h$ とゼロベクトルのコサイン類似度は $0.0$ となる。

ここで問題が発生する。コサイン類似度の値域は $[-1.0, 1.0]$ である。もし正常なシナプスのコサイン類似度が負の値（例：$-0.5$）であった場合、 **空のシナプス（$0.0$）の方が数学的にスコアが高くなり、ルーターが空のシナプスを優先的に選択してしまう** 。

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

「存在を消したはずの空きスロットにデータが吸い込まれて消滅する」という、ブラックホールのような挙動である。

### 6.3 解決策：-∞マスクによる完全封鎖
ルーターの順伝播にゼロクリアされたシナプスを検知・除外するマスク処理を追加した。

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# ゼロクリアされたシナプスを検知
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

$-\infty$ マスクにより、他のシナプスのスコアがどれだけ低くても、空のシナプスが選択されることは数学的に不可能となる。

## 7. The Complete Lifecycle of Modular AI

以上の機構により、SRAは以下のモジュラーAIの完全なライフサイクルを実現する。

```
学習（Train） → 合体（Hot-Swap） → 運用（Serve）
       ↓                                    ↓
  独立並行開発                         削除（Purge）
                                           ↓
                                   スロット再利用（Reuse）
                                           ↓
                                   新規合体（Hot-Swap）
```

1. **学習**: 複数チームがベースモデルを共有し、各自の特化シナプスを独立並行開発する。
2. **合体**: 学習済みテンソルを本番環境のベースモデルに物理コピーしてデプロイする。
3. **運用**: ハードマスクによるPre-filteringでZero Forgettingを保証しつつ推論する。
4. **削除**: 不要になったシナプスを物理的に取り外すか、ゼロクリアしてパージする。
5. **再利用**: 空きスロットに新しい特化シナプスをHot-Swapして再利用する。

## 8. Discussion

### 8.1 表現の分岐（Representation Divergence）
Hot-Swapが成立する **絶対条件** は、すべての特化シナプスが同一の事前学習済みベースモデル（共有トランク）から派生していることである。完全に独立して学習させたモデル間でシナプスを移植すると、内部ベクトル表現の乖離によりルーティングが崩壊する。

### 8.2 上位ルーター（Super Router）による代替
共有トランクの制約を緩和するため、独立したモデル全体をカプセル化し、Gumbel-Softmaxによる上位ルーター（Super Router）で束ねるアプローチも検証されている。この方式では完璧な $1.0$ vs $0.0$ のHard Routingが達成され、アーキテクチャが異なるモデル同士でも計算リソースの完全な動的切り替えが可能となる。

### 8.3 セキュリティリスク
Hot-Swap機能は、稼働中のシステム外部から動的に重みファイルを読み込むという特性上、新たなセキュリティ脅威を生む。主要なリスクとして、（1）Pickle Exploitによる任意コード実行、（2）悪意ある重みの注入（Backdoor Injection）、（3）ルーターキー偽装によるルーティング・ハイジャック、（4）スワップ・スラッシングによるDoS攻撃が挙げられる。`safetensors`フォーマットの義務化、暗号署名によるシナプス検証、レート制限の導入等による防御が推奨される。

### 8.4 現在の制約と今後の展望
本研究は `d_model=128`, `n_layers=4` 程度の小規模モデルにおける実験段階であり、10Bクラスの巨大LLMでの実証は今後の重要な課題である。また、ルーターとの同期問題——全く新しい能力を持つシナプスを追加する際にルーターキーの適応学習が必要になる可能性——についても更なる調査が求められる。

## 9. Conclusion

本稿では、SRA（Synaptic Routing Architecture）のモジュール性を活用し、LLMを「Hotswappable（動的抜き差し可能）」にする手法を提案・検証した。Hot-Swapの挿入（Plug-In）操作ではテンソルの物理コピーのみでデプロイを完了させ、抜去（Unplug）操作では物理的取り外しとゼロクリアによるパージの2手法を確立した。ベクトルデータベースのPre-filtering技術に着想を得たハードマスク機構によりZero Forgettingを数学的に保証し、抜去時に発見されたコサイン類似度におけるゼロベクトルの「ブラックホール問題」を解決することで、スロットの安全な再利用を実現した。

モデルが巨大化・ブラックボックス化の一途を辿る現代において、知能のパーツを物理的に抜き差ししてコントロールできる「Hotswappable LLM」のアプローチは、モデルのメンテナンス性・安全性・運用効率の観点から極めて有望な方向性である。

## References

- Suzuki, J. (2026).[All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

本稿で解説したHot-SwapおよびSynapse Deletionの全プロセスは、以下のGoogle Colabノートブックで実際に動かして体験できます。

- **Hot-Swap シナプス合体デモ（ベース学習〜独立学習〜合体〜Zero Forgettingの証明）**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **シナプス削除デモ（物理的取り外し〜ゼロクリア〜ブラックホール問題の解決）**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[SRAの未来：シナプスの動的ホットスワップと拡張性](./sra_future_hotswap_ja.md)** — カセット方式・シナプス運用、パーソナライズ、分散型学習に関する考察。
- **[SRAホットスワップ機能におけるセキュリティリスクと攻撃手法](./sra_security_risks_hotswap_ja.md)** — Pickle Exploit、Backdoor Injection、DoS攻撃等の脅威ベクトルと防御策。
- **[表現の分岐と階層型ルーティング](./sra_representation_divergence_ja.md)** — Shared Trunk方式とSuper Routerによる解決策。
- **[SRA階層型ルーターにおけるHard Routing手法の比較検証](./sra_hierarchical_hard_routing_ja.md)** — Soft / STE / Gumbel-Softmaxの比較実験。
