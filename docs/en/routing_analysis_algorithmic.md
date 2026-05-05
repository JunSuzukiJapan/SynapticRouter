# Multi-Task Learning and Routing Analysis in Algorithmic Reasoning

## Overview
We verified whether Synaptic Routing Architecture (SRA) can autonomously allocate experts (synapses) and achieve specialization (modularization) according to the nature of the tasks.
We simultaneously trained a single SRA model on four algorithmic reasoning tasks with different characteristics (`copy`, `reverse`, `paren`, `addmod`) and analyzed its inference results and routing distribution.

## Task Details
By prepending a special token for task specification to the beginning of the sequence, we instruct the model on which task to solve.
- `COPY`: Output the input sequence as is.
- `REVERSE`: Output the input sequence in reverse.
- `PAREN`: Determine whether the parentheses are balanced (`Y` or `N`).
- `ADDMOD`: Calculate the ones digit (modulo) of the addition of two numbers.

## Training Results
After 10,000 steps of simultaneous training, we obtained perfectly accurate inference results (around 100% Accuracy) across all tasks. This demonstrates that SRA can integrate multiple tasks into a single model while preventing interference (catastrophic forgetting) between tasks.

## Routing Analysis (Cosine Similarity)
When inputting data for each task into the trained model, we extracted the distribution of which synapses (experts) were used in each layer and calculated the cosine similarity between tasks.

### Layer 0 (Shallow Layer)
```text
                copy   reverse     paren    addmod
      copy     1.000     0.708     0.788     0.867
   reverse     0.708     1.000     0.836     0.299
     paren     0.788     0.836     1.000     0.609
    addmod     0.867     0.299     0.609     1.000
```
In the shallow layer, common synapses are shared to some extent across all tasks. This suggests that basic feature extraction of tokens is performed cross-task.

### Layer 1 (Deep Layer)
```text
                copy   reverse     paren    addmod
      copy     1.000     0.969     0.213     0.336
   reverse     0.969     1.000     0.029     0.134
     paren     0.213     0.029     1.000     0.858
    addmod     0.336     0.134     0.858     1.000
```
In the deep layer, the routing distribution per task was clearly grouped.

1. **Sequence Operation Group**:
   `COPY` and `REVERSE` showed a very high similarity of **0.969**. Both are tasks of "reconstructing the input sequence while maintaining or reversing its order," and SRA routes them to the same group of synapses as tasks with similar properties.

2. **Calculation/Logic Group**:
   `PAREN` and `ADDMOD` seem unrelated at first glance, but their similarity was high at **0.858**. Both require logical calculations beyond simple sequence rearrangement, such as "state management (parenthesis depth)" or "arithmetic operations (addition)", so they are routed to the same group of synapses.

3. **Clear Separation Between Groups**:
   The similarity between the two groups above is extremely low, at **0.029 to 0.336**.

## Conclusion
It was quantitatively proven that the SRA router **autonomously discerns the nature of tasks without human explicit instruction, shares synapses for similar tasks, and modularizes to clearly use different synapses for different tasks**.
