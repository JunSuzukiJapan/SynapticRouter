# Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion via Synaptic Routing Architecture

**Jun Suzuki**, Independent Researcher

## Abstract
人間は後天的に新しいスキル（例えば自転車の乗り方や新しい言語）を学習し、既存の知識を破壊することなく、脳の独立した回路として機能させることができます。しかし、現在の巨大な言語モデル（LLM）はすべての知識を単一のパラメータ空間に密に格納するモノリシック構造を持つため、このような後天的な知識の独立した追加や特定の記憶の削除が極めて困難であり、運用上の柔軟性に重大な制約を課しています。

本稿では、Synaptic Routing Architecture（SRA）の機能局在メカニズムを活用し、神経回路（モジュール）の動的な抜き差し——すなわち **Hot-Swap**——を実現する手法を提案・検証します。Hot-Swapとは、ベースモデルとは独立して学習させた特化シナプスを、**一切の再学習なしで本番モデルに外科的に移植（Plug-In）** し、不要になれば **特定の記憶だけを安全に切り離す（Unplug）** ことができる、双方向のモジュール交換操作です。実験の結果、ベクトルデータベースの事前フィルタリング技術に着想を得た神経回路のハードマスク機構により、移植後もベースモデルの出力が小数点以下まで完全に一致する **Zero Forgetting** を達成しました。加えて、抜去操作においてコサイン類似度特有のゼロベクトルの「ブラックホール問題」を発見・解決し、「学習→移植→除去→再利用」という、生物の神経可塑性を超越するモジュラーAIの完全なライフサイクルを実現しました。

## 1. Introduction

### 1.1 モノリスモデルと後天的な記憶の制御の限界
「Attention Is All You Need」以降、Transformerアーキテクチャは自然言語処理において支配的な地位を確立しました。しかし、数百億〜数千億パラメータを持つモノリシックなLLMは、後天的な知識の制御において以下の深刻な問題を抱えています。

1. **破滅的忘却（Catastrophic Forgetting）**: 汎用モデルに特定ドメイン（例：社内規程、専門コード）を追加学習させると、元の汎用能力が破壊・劣化する。
2. **学習コストの肥大化**: 新知識を追加するたびにモデル全体の再学習やマージが必要となり、複数チームでの完全並行開発が困難である。
3. **知識削除の不可能性**: 特定の知識だけを忘れさせる「Machine Unlearning」は、パラメータが複雑に絡み合ったモノリスモデルでは至難の業であり、再学習を試みると無関係な能力まで破壊される。

### 1.2 本稿の貢献
私が提案する Synaptic Routing Architecture (SRA) [Suzuki, 2026] は、経路の選択とモジュールの表現をEnd-to-Endで同時に学習することで、タスクに応じた「機能局在」が自律的に創発するアーキテクチャです。本稿ではSRAのこの生物学的なモジュール性がもたらす **運用上の革新**——神経回路の移植と除去を自在に行う **Hot-Swap**——に焦点を当て、以下の3つの貢献を報告します。

1. **Hot-Swap: Plug-In（外科的移植）**: 独立学習した特化モジュールの重みテンソルを、ベースモデルの空きスロットに物理コピーするだけでデプロイが完了する手法の実装と検証。
2. **Hot-Swap: Unplug（記憶の除去と不活性化）**: 物理的な切り離し（`pop_synapses`）と、ゼロクリアによるシナプスの不活性化パージ（`clear_synapses`）という2つの削除APIの設計と、ルーティングにおける「ゼロベクトルのブラックホール問題」の発見・解決。
3. **Zero Forgetting の実験的証明**: 特定のタスクにおいて不要な経路を強制遮断するハードマスク機構（Pre-filtering）により、移植・除去の前後でベースモデルの出力Lossが小数点以下まで完全に一致することの実証。

## 2. Background: SRA Architecture

SRAは、脳の空間的隔離と動的ルーティングを模倣した継続学習アーキテクチャです。Hot-Swapの理解に必要な構成要素を概説します（詳細は [Suzuki, 2026] を参照）。

### 2.1 Router (Dynamic Synaptic Formation)
SRAの心臓部であるルーターは、入力された情報に対して「どの神経回路を発火させるべきか」を決定する単一の線形層です。入力の隠れ状態 $h$ と各シナプスの特徴ベクトル（埋め込み）$e_i$ のコサイン類似度を計算し、Top-k個のシナプスを選択します。

### 2.2 Tiny Synapses (Functional Modules)
各シナプスは、独立した非常に小さなMulti-Head AttentionとMLPで構成されます。ルーターによって「発火」を指示されたシナプスのみが計算を実行し、それ以外のパラメータは干渉を受けません。

### 2.3 共有トランク（Shared Trunk）と機能局在
Hot-Swapの成立条件として極めて重要なのが、**共有トランク方式**です。すべての特化シナプスは、同一の事前学習済みベースモデル（視覚野や基礎言語野に相当するEmbedding/Attention層）を共有し、シナプス部分のみを独立学習します。これによりモデル間の表現の乖離（Representation Divergence）を防ぎ、物理的な回路の移植を可能にします。

## 3. Hot-Swap: Plug-In（モジュールの外科的移植）

Hot-Swapの第一の操作は、独立学習した特化モジュール（新しいスキル）をベースモデルに **移植（Plug-In）** することです。この移植操作は、PyTorchのテンソル操作のみで完結します。

### 3.1 手法

```python
# hotswap_model: 本番環境のベースモデル（脳の空きスロットを追加済み）
# plugin_math: 数学チームが独立学習した特化LLM（新しい数学回路）

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # ルーターの埋め込みベクトル（発火条件）をコピー
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Expert (TinySynapse) の重み (w1, w2) をコピー
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

ベースモデルの空きスロットに対して、学習済み特化モデルのテンソルを直接代入するだけです。SRAはベースモデルの共通知識を完全に凍結した状態でシナプスのみを学習するため、この物理コピーが成立します。

### 3.2 独立並行開発の実現
これにより、「コードチーム」と「数学チーム」が同一のベースモデルを元に、互いに一切干渉せず自分たちの特化シナプスだけを独立学習できます。完了後は、重みテンソルを本番環境のベースモデルにメモリコピーするだけでデプロイが完了します。

## 4. Zero Forgetting: 経路の強制遮断によるハードマスク機構

### 4.1 課題：ルーターの混同
単純にテンソルを移植しただけでは、ルーターが新旧のシナプスを混同し、ベースモデルの既存能力に悪影響を及ぼす（忘却する）危険があります。

### 4.2 ベクトルDBに着想を得た事前フィルタリング
SRAのルーターの本質は、インメモリのベクトル検索エンジン（Maximum Inner Product Search）に似ています。私はベクトルデータベースの **事前フィルタリング（Pre-filtering）** 技術をルーターの順伝播に組み込み、推論時に「現在のタスクにおいて不必要な神経回路への経路」を物理的に遮断する（ハードマスク）機構を設計しました。

```python
# Router層の順伝播
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtering: 許可されていないシナプスの経路を -∞ で強制遮断
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-Kルーティング
vals, idx = torch.topk(logits, k, dim=-1)
```

この `-inf` マスクにより、ルーターは許可されたシナプスの中からのみ回路を選択します。別モデルの重みがどれだけ移植されて共存していても、計算グラフ上では完全に無視されるため、**ベースモデルのLossが移植前と小数点以下まで完全に一致する（数学的にゼロの干渉）** ことが保証されます。

## 5. Hot-Swap: Unplug（特定の記憶の除去・不活性化）

Hot-Swapの第二の操作は、不要になったモジュールをベースモデルから **抜去（Unplug）** することです。モノリスモデルにおける特定の記憶の消去（Machine Unlearning）は困難ですが、SRAはこの問題を「シナプス結合の物理的切断」というアプローチで解決します。

### 5.1 アプローチ①：物理的切り離し（pop_synapses）
後から追加したシナプスが不要になった場合、テンソルの末尾からスライスして物理的に切り捨てます。

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

VRAMの使用量が縮小され、モデルはシナプス追加前の状態に完全に復元されます。これは特定の神経回路を文字通り物理的に切断する操作です。

### 5.2 アプローチ②：不活性化によるパージ（clear_synapses）
末尾ではなく中間のシナプスを削除したい場合、物理的な削除はインデックスのズレを引き起こし、脳の制御システムが崩壊します。そこで、シナプスの中身をゼロクリアして「不活性化（空きスロット化）」します。

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

テンソルサイズを変えずにスロットの中身だけを無効化するため、整合性は完璧に保たれます。

## 6. The Cosine Similarity Trap: ゼロベクトルのブラックホール問題

### 6.1 問題の発見と原因
不活性化パージを実装したところ、**出力が完全に崩壊する**という深刻な問題に遭遇しました。
SRAのルーターはコサイン類似度を用いますが、ゼロクリアされたシナプスの埋め込みベクトルは正規化しても $\mathbf{0}$ であり、コサイン類似度は $0.0$ となります。
類似度の値域は $[-1.0, 1.0]$ であるため、正常なシナプスのスコアが負（例：$-0.5$）であった場合、**不活性化したはずの空のシナプス（$0.0$）の方がスコアが高くなり、ルーターが空のシナプスを優先的に選択してしまう**現象が発生したのです。

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

これは、存在を消したはずのシナプスにデータが吸い込まれる「ブラックホール」のような挙動でした。

### 6.2 解決策：-∞マスクによる完全封鎖
この問題を解決するため、ルーターの順伝播に、ゼロクリアされたシナプスを検知して経路を完全封鎖するマスク処理を追加しました。

```python
# ゼロクリアされた（不活性化された）シナプスを検知
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

$-\infty$ マスクにより、不活性化されたシナプスが選択されることは数学的に不可能となり、スロットの安全な再利用が実現しました。

## 7. The Complete Lifecycle of Modular AI

SRAは以下のモジュラーAIの完全なライフサイクルを実現します。

```
学習（Train） → 移植（Plug-In） → 運用（Serve）
       ↓                                ↓
  独立並行開発                      除去（Unplug）
                                        ↓
                                スロット再利用（Reuse）
                                        ↓
                                新規移植（Plug-In）
```

これにより、生物の脳が一生を通じて新しいスキルを学び続けるように、モデルの無限の拡張と継続的なメンテナンスが可能になります。

## 8. Discussion

### 8.1 表現の分岐（Representation Divergence）
Hot-Swapが成立する絶対条件は、すべての特化シナプスが同一の事前学習済みベースモデルから派生していることです。完全に独立して学習させたモデル間では、内部表現の乖離によりルーティングが崩壊します。

### 8.2 上位ルーター（Super Router）による代替
共有トランクの制約を緩和するため、独立したモデル全体をカプセル化し、Gumbel-Softmaxによる上位ルーターで束ねるアプローチも検証されています。

### 8.3 セキュリティリスク
Hot-Swap機能は、稼働中のシステム外部から動的に重みファイルを読み込む特性上、（1）Pickle Exploit、（2）悪意ある重みの注入（Backdoor Injection）、（3）ルーティング・ハイジャックなどのリスクを生みます。`safetensors`フォーマットの義務化や暗号署名によるシナプス検証が必要です。

## 9. Conclusion

本稿では、SRAの機能局在メカニズムを活用し、特定の神経回路を動的に移植・除去できる「Hotswappable LLM」の手法を確立しました。テンソルの物理コピーのみで新たなスキルをPlug-Inし、物理的切断とゼロクリアによるパージで特定の記憶だけをUnplugすることに成功しました。

ハードマスク機構によりZero Forgettingを数学的に保証し、コサイン類似度における「ブラックホール問題」を解決した本アプローチは、AIモデルを「単一のブラックボックス」から「知能のパーツを物理的に抜き差しして安全にコントロールできる生命的なモジュラーシステム」へと進化させる、極めて有望な方向性です。

## References

- Suzuki, J. (2026).[Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft.md)
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

- **[SRAの未来：シナプスの動的ホットスワップと拡張性](./sra_future_hotswap_ja.md)**
- **[SRAホットスワップ機能におけるセキュリティリスクと攻撃手法](./sra_security_risks_hotswap_ja.md)**
- **[表現の分岐と階層型ルーティング](./sra_representation_divergence_ja.md)**
- **[SRA階層型ルーターにおけるHard Routing手法の比較検証](./sra_hierarchical_hard_routing_ja.md)**
