# Análise de Roteamento em Modelagem de Linguagem Interdomínio (Código / Matemática / Texto)

## Visão Geral
Conduzimos um experimento com SRA para aprender três domínios (Código, Matemática, Texto) com diferentes vocabulários simultaneamente.

## Qualidade de Inferência
Após 1000 passos, previsões precisas foram alcançadas sem interferência.

## Análise de Especialização
1. **Homogeneização Inicial**: No início, as tarefas usavam as 16 sinapses de maneira uniforme.
2. **Especialização**: O roteamento separou-se claramente.
   - `Código` usou a **Sinapse 8**.
   - `Matemática` usou as **Sinapses 10 e 13**.
   - `Texto` usou as **Sinapses 0 e 15**.

## Conclusão
O modelo descobriu autonomamente o "uso de sinapses próprias para cada tarefa", prevenindo o esquecimento catastrófico.
