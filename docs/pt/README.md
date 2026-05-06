[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

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

## 🧪 Experimentos e Análise
- [Análise de Aprendizado Multitarefa e Roteamento](./routing_analysis_algorithmic.md)
- [Análise de Roteamento em Modelagem de Linguagem Interdomínio](./routing_analysis_language.md)
- [Análise de Roteamento em Tradução Multilíngue (Ing / Fra / Jap) e Generalização Zero-Shot](../dev/multilingual_translation_routing_analysis.md)
  - Um relatório fascinante mostrando como o SRA atribui automaticamente diferentes módulos de tradução com base na estrutura gramatical (SVO vs SOV). Ainda mais surpreendente, ao traduzir um par de idiomas não aprendido, ele usa inconscientemente o inglês como uma "língua pivô"!
- [Separação Completa de Percepção e Política no Decision Transformer (Aprendizado por Reforço)](../dev/decision_transformer_routing_analysis.md)
  - Demos ao SRA a habilidade de jogar um jogo. Ele descobriu uma estrutura modular incrível por conta própria: usa exatamente o mesmo módulo de "visão" para perceber o ambiente em todas as tarefas, mas muda para módulos de "cérebro" completamente diferentes dependendo se precisa encontrar um tesouro ou fugir.
- [Verificação de Tradução Multilíngue Prática usando SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Um relatório demonstrando que, ao estender o SRA para uma arquitetura Encoder-Decoder e treinar por 30.000 passos em um corpus real (opus100), ele pode traduzir expressões práticas como "Merci beaucoup." e "Good morning." com BLEU=1.0. A introdução da Cross-Attention causou um salto do Decoder-only (BLEU=0) para um BLEU médio geral de 0,27, e alcançou uma precisão quase prática de BLEU=0,56 na direção FR→EN.

