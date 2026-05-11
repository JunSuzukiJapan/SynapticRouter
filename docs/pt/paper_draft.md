# Roteamento Sináptico Bioinspirado: Superando o Esquecimento Catastrófico via Caminhos Modulares Dinâmicos

**Jun Suzuki**, Independent Researcher

## Resumo (Abstract)
O cérebro humano pode aprender e executar tarefas fundamentalmente diferentes — como andar, falar e calcular — sem interferência mútua. Isso ocorre porque os circuitos neurais (sinapses) do cérebro são roteados dinamicamente de acordo com a tarefa, mantendo uma "localização funcional (Functional Localization)" espacialmente isolada. Em contraste, quando Redes Neurais Artificiais (ANNs) aprendem múltiplas tarefas dentro de uma única rede monolítica, elas sofrem de "Esquecimento Catastrófico (Catastrophic Forgetting)", onde memórias passadas são destruídas.

Neste artigo, propomos a "Arquitetura de Roteamento Sináptico (SRA)", um modelo de aprendizado contínuo inspirado nos mecanismos biológicos de formação dinâmica de sinapses e isolamento espacial. A SRA consiste em um "Roteador (Router)" de camada única extremamente simples e múltiplos micromódulos independentes (Sinapses). Por meio de nossos experimentos, demonstramos que a SRA pode identificar autonomamente a natureza de uma tarefa a partir da entrada — sem receber um ID de tarefa externo durante a inferência — e **aprender tanto o roteamento (seleção de caminho) quanto as representações da tarefa simultaneamente de ponta a ponta (End-to-End).** Mostramos que, sem congelamento artificial de pesos ou algoritmos evolutivos complexos, uma localização funcional autônoma surge dentro do modelo, evitando completamente o esquecimento catastrófico.

## 1. Introdução (Introduction)
No campo de deep learning, o "Aprendizado Contínuo" é uma das maiores barreiras para realizar a Inteligência Artificial Geral (AGI). Redes monolíticas, como os atuais modelos massivos baseados em Transformer, inevitavelmente esquecem o conhecimento aprendido anteriormente quando ajustados (fine-tuned) em novos domínios.
A SRA é projetada como uma arquitetura que pode ligar/desligar (Plug-In/Unplug) redes independentes minúsculas (sinapses) dinamicamente através de um mecanismo de roteamento.

## 2. Trabalhos Relacionados e a Novidade da SRA
Em comparação com abordagens como a PathNet (que exige IDs de tarefa e usa algoritmos genéticos), a novidade da SRA reside em **"aprender o roteamento e as representações dos módulos simultaneamente de maneira diferenciável."** A SRA é Agnóstica à Tarefa (Task-Agnostic), usa apenas retropropagação padrão e permite o surgimento de localização funcional dinâmica por meio de ativação esparsa (Sparse Activation).

## 3. Arquitetura (Neuro-inspired Design)
A SRA imita a formação de sinapses do cérebro biológico. O roteador linear calcula a similaridade por cosseno entre os recursos de entrada e o vetor de incorporação (embedding) de cada sinapse, determinando os módulos (Top-k) que devem "disparar".

## 4, 5, 6, 7. Experimentos (Experiments)
Conduzimos experimentos em Raciocínio Algorítmico, Modelagem de Linguagem entre Domínios, Tradução Automática Multilíngue e Aprendizado por Reforço Offline (Decision Transformer). Em todos os casos, **o aprendizado simultâneo do roteador levou à segregação modular autônoma, semelhante à localização funcional do cérebro**, sem a necessidade de IDs de tarefa.

## 8. Conclusão (Conclusion)
A SRA representa uma mudança de paradigma das redes estáticas tradicionais para uma "rede modular equipada com isolamento espacial biológico e roteamento dinâmico," um passo crucial em direção à AGI escalável.

## Referências (References)
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
