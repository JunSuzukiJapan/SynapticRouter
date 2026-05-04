# SynapticRouter Project – Comprehensive Summary

## ■ Overview
This document summarizes the full design discussion of a novel AI architecture inspired by biological synapses.

---

## ■ Motivation
- Transformerはdenseで静的
- 脳は動的・状態を持つシナプスで構成
- 仮説：構造の方が重要では？

---

## ■ Core Idea
巨大な1モデルではなく
→ 小さなSynapse群 + Routerで動的構成

---

## ■ Architecture

### Synapse
- 小型モデル（MLP/Transformer）
- 専門化する

### Router
- top-k選択
- 重み付け

### Synapse Space
- embeddingで配置
- 距離 = 類似性

---

## ■ Learning

Δw = trace × routing × reward

---

## ■ Reward
- 正解
- 好奇心
- 効率
- 専門化

---

## ■ Tasks
- copy
- reverse
- parentheses
- add

---

## ■ VRAM設計（8GB）
- 100M〜200M params
- 512 synapses
- top-k=2

---

## ■ 名前
SynapticRouter

---

## ■ 本質
静的モデル → 動的モジュールシステム
