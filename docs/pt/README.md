[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

## 🎮 Demonstrações interativas (Jupyter Notebooks)

Preparamos Jupyter Notebooks onde você pode experimentar interativamente o "uso do cérebro específico de tarefas" e a "robustez" do SRA diretamente no seu navegador.Você pode executá-los em segundos no Google Colab, então experimente!

| # | Demo | Descrição | Colab |
|---|------|-----------|-------|
| 🟢 1 | [SRA Quickstart](../../notebooks/01_sra_quickstart_pt.ipynb) | Estrutura básica SRA e visualização de roteamento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_pt.ipynb) |
| 🔵 2 | [Aprendizado e Roteamento](../../notebooks/02_learning_and_routing_demo_pt.ipynb) | Aprendizado de tarefa única e especialização de roteamento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_pt.ipynb) |
| 🔴 3 | [Roteamento Multitarefa](../../notebooks/03_multitask_routing_demo_pt.ipynb) ✨ | Aprendizado multitarefa e comutação de sinapses | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_pt.ipynb) |
| 🕹️ 4 | [Decision Transformer](../../notebooks/04_decision_transformer_routing_demo_pt.ipynb) | Separação de percepção e ação em RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_pt.ipynb) |
| 🧠 5 | [Experimento de Lesão](../../notebooks/05_lesion_experiment_demo_pt.ipynb) ✨ | Prova de modularidade funcional destruindo sinapses | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_pt.ipynb) |
| 🔌 6 | [Experimento Hot-Swap](../../notebooks/06_hotswap_experiment_demo_pt.ipynb) | Hot-swap sináptico dinâmico e limites do roteador | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_pt.ipynb) |
| 👑 7 | [Super Router (Gumbel)](../../notebooks/07_super_router_gumbel_demo_pt.ipynb) | Integração de modelos via Gumbel-Softmax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_pt.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](../../notebooks/08_sra_llm_demo_shakespeare_pt.ipynb) | Construir e treinar um Tiny LLM com SRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_pt.ipynb) |
| 📚 9 | [LLM Multidomínio](../../notebooks/09_sra_llm_demo_multidomain_pt.ipynb) | Aprendizado simultâneo multidomínio (Code/Math/Text) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_pt.ipynb) |
| 💻 10 | [Plugin Hot-Swap](../../notebooks/10_hotswap_plugins_demo_pt.ipynb) | Hot-swap zero-shot (zero esquecimento catastrófico) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_pt.ipynb) |
| 🗑️ 11 | [Exclusão de Sinapses](../../notebooks/11_synapse_deletion_demo_pt.ipynb) | Exclusão dinâmica de sinapses (pop & clear) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_pt.ipynb) |
| 🧬 12 | [Emergência de Neurônios Virtuais](../../notebooks/12_virtual_neuron_experiment_pt.ipynb) | 5 domínios x 5 tarefas revelam formação autônoma de neurônios virtuais | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/12_virtual_neuron_experiment_pt.ipynb) |
| 🧠 13 | [Troca a quente de neurônios virtuais](../../notebooks/13_virtual_neuron_hotswap_pt.ipynb) | Desaprendizagem segura na granularidade do neurônio virtual (montagem de células) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/13_virtual_neuron_hotswap_pt.ipynb) |
| 🔬 14 | [Comparação de unidades de exclusão](../../notebooks/14_compare_deletion_units_pt.ipynb) | Exclusão de unidade de sinapse vs. unidade de neurônio e emaranhamento de conhecimento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/14_compare_deletion_units_pt.ipynb) |
| 📐 15 | [Hipótese de Capacidade](../../notebooks/15_capacity_hypothesis_experiment_pt.ipynb) | Capacidade de sinapse versus limite de desaprendizado seguro | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment_pt.ipynb) |
| 💤 16 | [Prevenção de roteamento lento](../../notebooks/16_lazy_routing_prevention_experiment_pt.ipynb) | Diagnosticar e mitigar a preguiça do roteador | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/16_lazy_routing_prevention_experiment_pt.ipynb) |
| 🔁 17 | [Retorno de roteamento](../../notebooks/17_routing_fallback_experiment_pt.ipynb) | Reatribuir o tráfego quando uma sinapse ficar indisponível | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/17_routing_fallback_experiment_pt.ipynb) |
| 🧩 18 | [Sinapses Personalizadas](../../notebooks/18_custom_synapses_pt.ipynb) | Sinapses de banco de dados vetorial e calculadora não treináveis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/18_custom_synapses_pt.ipynb) |
| 🎯 19 | [Roteamento rígido Zero-Shot](../../notebooks/19_zero_shot_hard_routing_pt.ipynb) | Forçar roteamento via permitido_mask sem retreinar | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/19_zero_shot_hard_routing_pt.ipynb) |
| 🛠️ 20 | [Ajuste fino de roteamento](../../notebooks/20_routing_finetuning_pt.ipynb) | Aprendizagem de roteamento autônomo por meio de ajuste fino de pequenos conjuntos de dados | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/20_routing_finetuning_pt.ipynb) |
| 🧯 21 | [Verificação de esquecimento de ajuste fino](../../notebooks/21_finetuning_forgetting_check_pt.ipynb) | Verificação de esquecimento catastrófico após ajuste fino de roteamento | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/21_finetuning_forgetting_check_pt.ipynb) |
| 🧪 22 | [Coexistência Neuro-Simbólica](../../notebooks/22_multi_synapse_hotswap_eval_pt.ipynb) | LLM mais Vector DB mais calculadora baseada em regras em uma arquitetura | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/22_multi_synapse_hotswap_eval_pt.ipynb) |
| 🦙 23 | [Integração SRA LLM (TinyLlama)](../../notebooks/23_sra_llm_integration_pt.ipynb) | Integração nativa do roteador SRA com TinyLlama (PoC) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/23_sra_llm_integration_pt.ipynb) |
| 🏎️ 24 | [Referência de arquitetura de roteador](../../notebooks/24_router_architecture_benchmark_pt.ipynb) | Referência de roteadores de estágio único/multiestágio/último token | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/24_router_architecture_benchmark_pt.ipynb) |
| 🧰 25 | [Arquitetura da placa-mãe](../../notebooks/25_integrated_heterogeneous_routing_pt.ipynb) | Roteador de último token e fallback semântico sobre sinapses heterogêneas | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/25_integrated_heterogeneous_routing_pt.ipynb) |
| 💬 26 | [Demonstração do chatbot SRA](../../notebooks/26_chatbot_demo_pt.ipynb) | UI de chat de trabalho combinando sinapses LLM / Vector DB / Calculadora | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/26_chatbot_demo_pt.ipynb) |


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


## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways](./paper_draft.md)
- [Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion](./paper_hotswap.md)

## 🧪 Experimentos e Análise
- [Análise de Aprendizado Multitarefa e Roteamento](./routing_analysis_algorithmic.md)
- [Análise de Roteamento em Modelagem de Linguagem Interdomínio](./routing_analysis_language.md)
- [Análise de Roteamento em Tradução Multilíngue (Ing / Fra / Jap) e Generalização Zero-Shot](../dev/multilingual_translation_routing_analysis.md)
  - Um relatório fascinante mostrando como o SRA atribui automaticamente diferentes módulos de tradução com base na estrutura gramatical (SVO vs SOV). Ainda mais surpreendente, ao traduzir um par de idiomas não aprendido, ele usa inconscientemente o inglês como uma "língua pivô"!
- [Separação Completa de Percepção e Política no Decision Transformer (Aprendizado por Reforço)](../dev/decision_transformer_routing_analysis.md)
  - Demos ao SRA a habilidade de jogar um jogo. Ele descobriu uma estrutura modular incrível por conta própria: usa exatamente o mesmo módulo de "visão" para perceber o ambiente em todas as tarefas, mas muda para módulos de "cérebro" completamente diferentes dependendo se precisa encontrar um tesouro ou fugir.
- [Verificação de Tradução Multilíngue Prática usando SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Um relatório demonstrando que, ao estender o SRA para uma arquitetura Encoder-Decoder e treinar por 30.000 passos em um corpus real (opus100), ele pode traduzir expressões práticas como "Merci beaucoup." e "Good morning." com BLEU=1.0. A introdução da Cross-Attention causou um salto do Decoder-only (BLEU=0) para um BLEU médio geral de 0,27, e alcançou uma precisão quase prática de BLEU=0,56 na direção FR→EN.

---

### 📖 8. Demonstração SRA LLM (Shakespeare)
**File:** [`08_sra_llm_demo_shakespeare_pt.ipynb`](../../notebooks/08_sra_llm_demo_shakespeare_pt.ipynb)

Este é um tutorial que usa dados de Shakespeare em pequena escala para treinar SRA como um modelo generativo específico do decodificador (LLM). Após o aprendizado, um mapa de calor é utilizado para visualizar por qual sinapse passou cada token do texto gerado.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_pt.ipynb)

---

### 🌐 9. Demonstração LLM de vários domínios SRA (código, matemática, texto)
**File:** [`09_sra_llm_demo_multidomain_pt.ipynb`](../../notebooks/09_sra_llm_demo_multidomain_pt.ipynb)

Experimente a especialidade da SRA de ``aprendizagem simultânea de múltiplos domínios (Código, Matemática, Texto)'' em um LLM de pequena escala. Você pode verificar como o modelo divide (especializa) automaticamente as sinapses com base nos dados.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_pt.ipynb)

---

### 💻 10. Plugin Prático Hot-Swap (Zero-Shot)
**File:** [`10_hotswap_plugins_demo_pt.ipynb`](../../notebooks/10_hotswap_plugins_demo_pt.ipynb)

Demonstraremos um fluxo de trabalho no qual várias equipes de desenvolvimento aprendem independentemente plug-ins para "código" e "matemática" e os "mesclam fisicamente (troca a quente)" no modelo base do ambiente de produção após o fato. Está comprovado que mesmo após a fusão, as perdas de todos os domínios são exatamente as mesmas que durante a aprendizagem independente (Esquecimento Zero).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_pt.ipynb)

---

### 🗑️ 11. Exclusão sináptica dinâmica
**File:** [`11_synapse_deletion_demo_pt.ipynb`](../../notebooks/11_synapse_deletion_demo_pt.ipynb)

Demonstramos a função do SRA, “exclusão de sinapses”. Você pode experimentar tanto a ``remoção de plug-ins (pop_synapses)'', que exclui fisicamente as sinapses adicionadas posteriormente a partir do final, quanto a ``limpeza de um domínio específico (clear_synapses)'', que limpa e desativa com segurança as sinapses que não são compartilhadas.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_pt.ipynb)

