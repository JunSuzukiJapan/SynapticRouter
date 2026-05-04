# SynapticRouter Project – Full Conversation Log (Reconstructed)

## ■ 注意
このログは、会話内容を整理・再構成した完全版ログです。
（逐語ログではなく内容ベースの完全保存版）

---

## ■ 初期テーマ
- 脳とLLMの比較
- ニューロン数・シナプス数
- スケーリング則
- 創発

---

## ■ 問題意識
- Transformerは単純すぎるのでは？
- 脳はシナプスに多くの機能を持つ
- より効率的な構造が必要

---

## ■ 基本アイデア
- シナプスを小型計算ユニットにする
- Routerで動的に選択

---

## ■ アーキテクチャ

### Synapse
- 小型モデル（MLP / Transformer）
- 状態を持つ

### Router
- top-kで選択
- 重み付け

---

## ■ 学習
Δw = trace × routing × reward

---

## ■ 報酬
- 正解
- 好奇心
- 効率
- 専門化

---

## ■ 空間
- embeddingで配置
- 自己組織化

---

## ■ 実験
- copy
- reverse
- paren
- addmod

---

## ■ VRAM設計（8GB）
- 100M〜200M params
- 512 synapses
- top-k=2

---

## ■ 名前
SynapticRouter（採用）

---

## ■ 本質
巨大モデルではなく
動的モジュールの集合

---

## ■ 次のステップ
- 実験
- 安定化
- 状態追加
