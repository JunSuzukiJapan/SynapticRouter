# Routing Sinaptico Bio-ispirato: Superare l'Oblio Catastrofico tramite Percorsi Modulari Dinamici

**Jun Suzuki**, Independent Researcher

## Riassunto (Abstract)
Il cervello umano può apprendere ed eseguire compiti fondamentalmente diversi — come camminare, parlare e calcolare — senza interferenze reciproche. Questo accade perché i circuiti neurali (sinapsi) del cervello vengono indirizzati (routed) dinamicamente in base al compito, mantenendo una "localizzazione funzionale (Functional Localization)" spazialmente isolata. Al contrario, quando le Reti Neurali Artificiali (ANN) apprendono più compiti all'interno di un'unica rete monolitica, soffrono di "Oblio Catastrofico (Catastrophic Forgetting)", in cui le memorie passate vengono distrutte.

In questo articolo proponiamo l'"Architettura di Routing Sinaptico (SRA)", un modello di apprendimento continuo ispirato ai meccanismi biologici della formazione dinamica delle sinapsi e dell'isolamento spaziale. La SRA è composta da un "Router (Router)" a singolo strato estremamente semplice e da molteplici micromoduli indipendenti (Sinapsi). Attraverso i nostri esperimenti, dimostriamo che la SRA può identificare autonomamente la natura di un compito dall'input — senza ricevere alcun ID di compito esterno durante l'inferenza — e **apprendere simultaneamente sia il routing (selezione del percorso) sia le rappresentazioni del compito in modo completamente End-to-End.** Mostriamo che, senza il congelamento artificiale dei pesi o algoritmi evolutivi complessi, all'interno del modello emerge una localizzazione funzionale autonoma, che evita completamente l'oblio catastrofico.

## 1. Introduzione (Introduction)
Nel deep learning, l'"Apprendimento Continuo (Continual Learning)" è una delle barriere principali verso l'Intelligenza Artificiale Generale (AGI). Per affrontare il problema dell'oblio catastrofico delle reti monolitiche, la nostra ricerca si concentra sulla "Localizzazione Funzionale" del cervello. La SRA è progettata per accendere/spegnere dinamicamente (Plug-In/Unplug) minuscole reti indipendenti (sinapsi) tramite un meccanismo di routing.

## 2. Lavori Correlati e Novità della SRA
A differenza di approcci come PathNet che richiedono Task ID e usano algoritmi genetici, la novità della SRA risiede nell'**"apprendere simultaneamente il routing e le rappresentazioni dei moduli"**. La SRA è Task-Agnostic, utilizza solo la normale backpropagation e permette l'emergere di una localizzazione funzionale dinamica tramite attivazione sparsa (Sparse Activation).

## 3. Architettura (Neuro-inspired Design)
La SRA imita la formazione delle sinapsi nel cervello biologico. Il Router lineare calcola la similarità del coseno tra le caratteristiche di input e il vettore di incorporamento (embedding) di ciascuna Sinapsi (moduli indipendenti di Multi-Head Attention e MLP), determinando i moduli Top-k da "attivare".

## 4, 5, 6, 7. Esperimenti (Experiments)
Abbiamo condotto esperimenti su Ragionamento Algoritmico, Modellazione del Linguaggio Cross-Domain, Traduzione Automatica Multilingue e Reinforcement Learning Offline (Decision Transformer). In ogni caso, **l'apprendimento simultaneo del router ha portato a una segregazione modulare autonoma, simile alla localizzazione funzionale del cervello**, senza alcuna supervisione tramite Task ID.

## 8. Conclusione (Conclusion)
La SRA supera l'oblio catastrofico e permette il passaggio dalle reti statiche tradizionali a una "rete modulare dotata di isolamento spaziale biologico e routing dinamico", un passo cruciale verso un'AGI scalabile.

## Riferimenti (References)
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
