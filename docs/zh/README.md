# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) 是一种受生物大脑（突触）启发的全新动态、稀疏、模块化神经网络架构。
与庞大且静态的 Transformer 不同，SRA 将输入动态路由到合适的“突触”（微小模块），从而实现更高效的学习和结构化智能。

## 🎯 动机

近年来，随着 AI 模型变得越来越庞大，单一架构的网络面临着诸如“计算资源不断膨胀”以及“多任务学习中的灾难性遗忘”等重大挑战。
SRA 试图通过一种 **稀疏方法：“根据输入动态调用并组合必要的微小模块（突触）”** 来解决这些问题。这使得在同一网络内学习具有不同特征的多个任务而不会相互干扰，旨在同时实现可扩展性和高学习效率。

## 🧠 架构概述

SRA 主要由以下组件组成：

1. **Synapse（突触模块）**
   - 独立的微小计算单元（例如，微型 Transformer 或 MLP）。
   - 它们通过学习专门用于特定功能或模式处理。
2. **Router（路由器）**
   - 根据输入 token 从所有可用突触中动态选择最优的 `Top-k` 个突触，实现稀疏计算。
3. **Synapse Space（突触空间）**
   - 每个突触放置在一个嵌入空间中，自组织以使突触之间的距离代表“功能相似性”。
4. **Local Learning Rule（局部学习规则）**
   - 除了标准的反向传播外，它结合了基于 Hebbian、STDP 和奖励的 3 因子规则（如 `trace × routing × reward`）的局部学习规则，促进负载均衡和专业化。

## 📁 目录结构

本存储库的主要结构如下：

- `src/` : SRA 模型的核心实现及训练/评估脚本。
  - `sra_gpu_models.py` / `sra_language_models.py` : 应用 MoE（混合专家）技术的 SRA 模型实现。
  - `train_mtl_algo.py` / `train_mtl_lang.py` : 多任务学习（算法推理和语言建模）的执行脚本。
- `docs/` : 架构决策记录（ADR）以及各种实验和路由分析的报告。
- `data/` : 用于训练和验证的玩具数据集（代码、数学、文本等）。
- `tests/` : 各组件的测试代码。

## 🚀 使用方法

### 环境要求
```bash
pip install torch
```

### 基本执行
您可以使用 `copy`, `reverse`, `paren`, `addmod` 等结构化算法任务测试模型的训练和推理。

运行单一任务：
```bash
python src/sra_experiment.py --task reverse --steps 2000
```

连续运行所有任务（任务套件）：
```bash
python src/sra_experiment.py --task-suite
```

### 与其他架构的比较
提供脚本以将 SRA 与标准 `Transformer` 和 `MLP` 架构进行比较。

```bash
# 使用 copy 任务比较架构
python src/compare_architectures.py --task copy --steps 500
```

## 📊 比较

> **📝 待更新:**
> 我们目前正在对标准 Transformer 模型进行定量比较验证。
> 初步验证（少步数的 copy 任务）表明，SRA 比 Transformers 更快降低验证损失并更快达到高精度（高学习/样本效率）。
>
> 详细的基准测试结果，包括按步骤吞吐量和 VRAM 效率提升，将在未来添加于此。

## 🧪 实验与分析

- [算法推理中的多任务学习和路由分析](./routing_analysis_algorithmic.md)
  - 验证 SRA 可以同时学习多个算法任务而不受干扰，并根据任务性质自主分离和模块化专家（突触）的报告。
- [跨领域语言建模中的路由分析 (代码 / 数学 / 文本)](./routing_analysis_language.md)
  - 验证 SRA 同时学习具有不同语法和词汇的领域（代码、数学公式、自然语言），并且突触功能分化（专业化）以对每个领域执行推理的机制的报告。

## 🤝 贡献与许可

该项目目前是处于早期阶段的实验性架构。非常欢迎通过 Issues 和 PR 提供错误报告、功能讨论以及性能改进的 pull requests！

- **许可**: 本存储库基于 [MIT 许可证](../../LICENSE) 发布。有关详细信息，请参阅 [`LICENSE`](../../LICENSE) 文件。
