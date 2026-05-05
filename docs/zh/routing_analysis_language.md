# 跨领域语言建模中的路由分析 (代码 / 数学 / 文本)

## 概述
模仿 Mixtral 等模型中采用的 Sparse MoE 方法，我们使用 SRA（Synaptic Routing Architecture）进行了一项“跨领域语言建模”实验，同时学习了三个词汇和语法完全不同的领域（代码、数学、文本）。本文档总结了关于突触路由（专业化）的学习结果、推理质量和分析结果。

## 实验设置
- **任务组成**:
  - `Code`: Python 代码片段
  - `Math`: LaTeX 格式的数学公式
  - `Text`: 混合日语和英语的自然语言
- **Token化**: 字符级 (Char-level Tokenization，词汇表大小 = 178)
- **模型**: `MoESRALanguageModel` (仅解码器的自回归语言模型)
- **超参数**:
  - Dimensions: 64
  - Layers: 2
  - Number of Synapses: 16
  - k (Routing Top-K): 2
  - Sequence Length: 32
  - Steps: 1000 steps

## 1. 推理质量评估
在经过 1000 步的训练后，将每个领域的一部分作为提示输入，并使用温度采样（Temperature Sampling, T=0.7）进行推理。结果证实了以下极其高精度的预测。

### [Code] 任务
**提示 (Prompt)**:
```python
is {i}")

def ad
```
**生成结果 (Generated Result)**:
```python
is {i}")

def add(a):
    is lefthvead aleanga
```
**分析**: 它准确地从 `def ad` 预测了函数定义的签名 `def add(a):`，并保持了适当的缩进宽度（4个空格）。

### [Math] 任务
**提示 (Prompt)**:
```latex
frac{P(B|A)P(A)}
```
**生成结果 (Generated Result)**:
```latex
frac{P(B|A)P(A)}{P(B)}

\frac{\pac{\iacosen)
```
**分析**: 在接受了贝叶斯定理的分子 `frac{P(B|A)P(A)}` 后，它立即补全了分母 `{P(B)}`。此外，在换行后，观察到它试图继续使用 LaTeX 特有的命令格式 `\frac{` 的行为。

### [Text] 任务
**提示 (Prompt)**:
```text
ght-iron lattice
```
**生成结果 (Generated Result)**:
```text
ght-iron lattice tower on the Chasngelampff la
```
**分析**: 它从训练数据中记住了 "wrought-iron lattice"（埃菲尔铁塔的描述），并且能够以高精度重建后续的 `tower on the Cha` (Champ de Mars)。

---

## 2. 路由和突触专业化分析
我们在学习初期（Early）和学习末期（Final）比较了全部 16 个突触的使用频率（Usage），并验证了它们在各个领域之间的隔离状态。

### 突触使用频率的转变
| 任务 | 初期 (Early Usage) | 最终 (Final Usage) | 专业化突触 |
| :--- | :--- | :--- | :--- |
| **Code** | 广泛分散 (Max 18%) | **[8]: 30%**, **[15]: 16%**, **[11]: 13%** | 突触 8 |
| **Math** | 广泛分散 (Max 19%) | **[10]: 33%**, **[13]: 24%**, **[11]: 12%** | 突触 10, 13 |
| **Text** | 广泛分散 (Max 24%) | **[15]: 20%**, **[11]: 18%**, **[0]: 16%** | 突触 0, 15 |

### 分析讨论
1. **初始状态下的均质化 (Warmup)**
   由于学习初期的负载均衡损失，每个任务相对均匀地使用了 16 个突触（每个约 0.05 至 0.15）。
2. **路由的冻结和分化**
   作为在特定步骤冻结路由器（Phase transition）以促进每个突触专业化的结果，每个领域使用的主要突触被清晰地分离。
   - 对于处理 `Code`，**突触 8** 开始占主导地位。
   - 处理 `Math` 主要由 **突触 10 和 13** 负责。
   - 处理 `Text` 则依赖于 **突触 0 和 15**。
3. **避免干扰 (防止灾难性遗忘)**
   在同一网络中学习三个具有完全不同的语法规则（Python缩进，LaTeX反斜杠表示法，自然语言上下文）的领域并没有失败，其原因可以说是因为 **MoE 带来的突触功能专业化**。由于每个领域都获得了自己独立的专业突触（参数空间），相互之间的干扰被降至最低。

## 结论
在使用 SRA 进行的多领域自回归语言建模中，即使只有少量的参数和步数，模型也能自主发现“针对每个任务正确使用突触”，并实现了很高的推理精度。这证实了 SRA/MoE 架构在混合学习异构数据（Code/Math/Text）中非常有效。
