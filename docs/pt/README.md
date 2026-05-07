[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

## 🎮 Demonstrações interativas (Jupyter Notebooks)

Preparamos Jupyter Notebooks onde você pode experimentar interativamente o "uso do cérebro específico de tarefas" e a "robustez" do SRA diretamente no seu navegador.Você pode executá-los em segundos no Google Colab, então experimente!

- [01 SRA Quickstart](../../notebooks/01_sra_quickstart_pt.ipynb)
- [02 Learning and Routing Demo](../../notebooks/02_learning_and_routing_demo_pt.ipynb)
- [03 Multitask Routing Demo](../../notebooks/03_multitask_routing_demo_pt.ipynb)
- [04 Decision Transformer Routing Demo](../../notebooks/04_decision_transformer_routing_demo_pt.ipynb)
- [05 Lesion Experiment Demo](../../notebooks/05_lesion_experiment_demo_pt.ipynb)


A Synaptic Routing Architecture (SRA) é uma arquitetura de rede neural dinâmica, esparsa e modular inspirada no cérebro biológico (sinapses).

## 🎯 Motivação
A SRA resolve problemas de modelos monolíticos usando uma **abordagem esparsa: "chamando e combinando dinamicamente apenas os módulos necessários (sinapses) dependendo da entrada"**. Isso permite aprender várias tarefas sem interferência.

## 💡 Ideia Básica

Modelos típicos de IA (como Transformers) tentam processar tudo usando um único "cérebro" gigante. No entanto, com essa abordagem, a carga computacional se torna muito pesada toda vez que o modelo é tornado mais inteligente ou maior. Portanto, a SRA adota um sistema onde **muitos "pequenos cérebros especialistas (que a SRA chama de 'sinapses')" são preparados, e apenas os especialistas necessários são chamados dependendo do problema em questão**.

A chave aqui é o mecanismo que decide "qual especialista chamar". A SRA tem um "roteador (guia)", que seleciona instantaneamente o especialista que parece mais capaz ao analisar os dados de entrada. À medida que cada especialista se torna mais inteligente (aprende), este roteador aprende simultaneamente "quem é o correto a escolher", crescendo para ser capaz de fazer atribuições ideais automaticamente.

## 🧠 Visão Geral da Arquitetura
1. **Sinapse:** Unidades computacionais independentes.
2. **Roteador:** Seleciona as `Top-k` melhores sinapses dinamicamente.
3. **Espaço da Sinapse:** Auto-organiza as sinapses por "similaridade funcional".
4. **Regra de Aprendizado Local:** Usa regras locais (Hebbian, STDP) para balanceamento.


---

### 🔌 6. Experimento de Hot-Swap Sináptico Dinâmico e Limites de Aprendizado do Roteador
**Arquivo:** [`06_hotswap_experiment_demo_pt.ipynb`](./06_hotswap_experiment_demo_pt.ipynb)

Demonstra o verdadeiro poder da SRA: "adição e remoção dinâmica de sinapses como plugins (Hot-Swap)".
Realizamos um experimento onde uma sinapse específica do espanhol é mesclada a um modelo de tradução francês/alemão em execução.
Neste notebook, você aprenderá a **importância crucial de compartilhar e congelar o espaço de conhecimento do modelo base (camadas de incorporação/atenção, etc.)** para estabelecer um hot-swap. Ao mesmo tempo, você enfrentará a **maior barreira da SRA (o problema do gradiente de fuga)**: o roteamento rígido padrão (Top-k) não pode aprender (diferenciar) retroativamente o roteamento das sinapses adicionadas. Essa limitação serve como um prenúncio crítico para a seção subsequente "Gumbel-Softmax (Super Router)".

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_pt.ipynb)

---

### 👑 7. Integração de Modelos via Super Router e Gumbel-Softmax
**Arquivo:** [`07_super_router_gumbel_demo_pt.ipynb`](./07_super_router_gumbel_demo_pt.ipynb)

Construímos um "Super Router" que agrupa vários modelos especializados (um modelo FR/DE e um modelo ES) e roteia dinamicamente o processamento com base na entrada.
Isso demonstra o problema de "Lazy Routing" do Soft Routing simples e mostra como o uso do Gumbel-Softmax alcança um **Hard Routing perfeito**, cortando em 100% o cálculo desnecessário do modelo.

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_pt.ipynb)

## 🧪 Experimentos e Análise
- [Análise de Aprendizado Multitarefa e Roteamento](./routing_analysis_algorithmic.md)
- [Análise de Roteamento em Modelagem de Linguagem Interdomínio](./routing_analysis_language.md)
- [Análise de Roteamento em Tradução Multilíngue (Ing / Fra / Jap) e Generalização Zero-Shot](../dev/multilingual_translation_routing_analysis.md)
  - Um relatório fascinante mostrando como o SRA atribui automaticamente diferentes módulos de tradução com base na estrutura gramatical (SVO vs SOV). Ainda mais surpreendente, ao traduzir um par de idiomas não aprendido, ele usa inconscientemente o inglês como uma "língua pivô"!
- [Separação Completa de Percepção e Política no Decision Transformer (Aprendizado por Reforço)](../dev/decision_transformer_routing_analysis.md)
  - Demos ao SRA a habilidade de jogar um jogo. Ele descobriu uma estrutura modular incrível por conta própria: usa exatamente o mesmo módulo de "visão" para perceber o ambiente em todas as tarefas, mas muda para módulos de "cérebro" completamente diferentes dependendo se precisa encontrar um tesouro ou fugir.
- [Verificação de Tradução Multilíngue Prática usando SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Um relatório demonstrando que, ao estender o SRA para uma arquitetura Encoder-Decoder e treinar por 30.000 passos em um corpus real (opus100), ele pode traduzir expressões práticas como "Merci beaucoup." e "Good morning." com BLEU=1.0. A introdução da Cross-Attention causou um salto do Decoder-only (BLEU=0) para um BLEU médio geral de 0,27, e alcançou uma precisão quase prática de BLEU=0,56 na direção FR→EN.

