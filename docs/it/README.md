[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)
L'Architettura di Routing Sinaptico (SRA) è una nuova architettura di rete neurale dinamica, sparsa e modulare ispirata al cervello biologico (sinapsi).
## 🎮 Demo interattive (taccuini Jupyter)

Abbiamo preparato Jupyter Notebook in cui puoi sperimentare in modo interattivo l'"utilizzo del cervello specifico per attività" e la "robustezza" di SRA direttamente nel tuo browser.Puoi eseguirli in pochi secondi su Google Colab, quindi provali!

| # | Demo | Descrizione | Colab |
|---|------|-------------|-------|
| 🟢 1 | [SRA Quickstart](../../notebooks/01_sra_quickstart_it.ipynb) | Struttura SRA di base e visualizzazione del routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_it.ipynb) |
| 🔵 2 | [Apprendimento e Routing](../../notebooks/02_learning_and_routing_demo_it.ipynb) | Apprendimento singolo e specializzazione del routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_it.ipynb) |
| 🔴 3 | [Routing Multitask](../../notebooks/03_multitask_routing_demo_it.ipynb) ✨ | Apprendimento multitask e commutazione di sinapsi | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_it.ipynb) |
| 🕹️ 4 | [Decision Transformer](../../notebooks/04_decision_transformer_routing_demo_it.ipynb) | Separazione percezione e azione in RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_it.ipynb) |
| 🧠 5 | [Esperimento di Lesione](../../notebooks/05_lesion_experiment_demo_it.ipynb) ✨ | Prova di modularità funzionale distruggendo sinapsi | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_it.ipynb) |
| 🔌 6 | [Esperimento Hot-Swap](../../notebooks/06_hotswap_experiment_demo_it.ipynb) | Hot-swap sinaptico dinamico e limiti del router | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_it.ipynb) |
| 👑 7 | [Super Router (Gumbel)](../../notebooks/07_super_router_gumbel_demo_it.ipynb) | Integrazione di modelli via Gumbel-Softmax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_it.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](../../notebooks/08_sra_llm_demo_shakespeare_it.ipynb) | Costruire e addestrare un Tiny LLM con SRA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_it.ipynb) |
| 📚 9 | [LLM Multidominio](../../notebooks/09_sra_llm_demo_multidomain_it.ipynb) | Apprendimento simultaneo multidominio (Code/Math/Text) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_it.ipynb) |
| 💻 10 | [Plugin Hot-Swap](../../notebooks/10_hotswap_plugins_demo_it.ipynb) | Hot-swap zero-shot (zero dimenticanza catastrofica) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_it.ipynb) |
| 🗑️ 11 | [Cancellazione Sinapsi](../../notebooks/11_synapse_deletion_demo_it.ipynb) | Cancellazione dinamica di sinapsi (pop & clear) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_it.ipynb) |
| 🧬 12 | [Emersione dei neuroni virtuali](../../notebooks/12_virtual_neuron_experiment_it.ipynb) | 5 domini x 5 compiti rivelano la formazione autonoma di neuroni virtuali | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/12_virtual_neuron_experiment_it.ipynb) |
| 🧠 13 | [Hot-swap di neuroni virtuali](../../notebooks/13_virtual_neuron_hotswap_it.ipynb) | Disapprendimento sicuro alla granularità del neurone virtuale (assemblaggio cellulare). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/13_virtual_neuron_hotswap_it.ipynb) |
| 🔬 14 | [Confronto delle unità di eliminazione](../../notebooks/14_compare_deletion_units_it.ipynb) | Cancellazione di unità sinapsi vs. unità neuronali e entanglement della conoscenza | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/14_compare_deletion_units_it.ipynb) |
| 📐 15 | [Ipotesi di capacità](../../notebooks/15_capacity_hypothesis_experiment_it.ipynb) | Capacità sinapsi vs soglia di disapprendimento sicuro | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment_it.ipynb) |
| 💤 16 | [Prevenzione del routing lento](../../notebooks/16_lazy_routing_prevention_experiment_it.ipynb) | Diagnosticare e mitigare la pigrizia del router | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/16_lazy_routing_prevention_experiment_it.ipynb) |
| 🔁 17 | [Fallback di instradamento](../../notebooks/17_routing_fallback_experiment_it.ipynb) | Riassegnare il traffico quando una sinapsi diventa non disponibile | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/17_routing_fallback_experiment_it.ipynb) |
| 🧩 18 | [Sinapsi personalizzate](../../notebooks/18_custom_synapses_it.ipynb) | Sinapsi del DB vettoriale e del calcolatore non addestrabili | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/18_custom_synapses_it.ipynb) |
| 🎯 19 | [Routing difficile a tiro zero](../../notebooks/19_zero_shot_hard_routing_it.ipynb) | Forza l'instradamento tramite aware_mask senza riqualificare | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/19_zero_shot_hard_routing_it.ipynb) |
| 🛠️ 20 | [Ottimizzazione del routing](../../notebooks/20_routing_finetuning_it.ipynb) | Apprendimento autonomo del routing tramite la messa a punto di set di dati di piccole dimensioni | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/20_routing_finetuning_it.ipynb) |
| 🧯 21 | [Perfezionamento del controllo dell'oblio](../../notebooks/21_finetuning_forgetting_check_it.ipynb) | Controllo catastrofico dell'oblio dopo la messa a punto del routing | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/21_finetuning_forgetting_check_it.ipynb) |
| 🧪 22 | [Coesistenza neuro-simbolica](../../notebooks/22_multi_synapse_hotswap_eval_it.ipynb) | LLM più Vector DB più calcolatore basato su regole su un'unica architettura | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/22_multi_synapse_hotswap_eval_it.ipynb) |
| 🦙 23 | [Integrazione SRA LLM (TinyLlama)](../../notebooks/23_sra_llm_integration_it.ipynb) | Integrazione nativa del router SRA con TinyLlama (PoC) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/23_sra_llm_integration_it.ipynb) |
| 🏎️ 24 | [Benchmark sull'architettura del router](../../notebooks/24_router_architecture_benchmark_it.ipynb) | Benchmark router monostadio/multistadio/last-token | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/24_router_architecture_benchmark_it.ipynb) |
| 🧰 25 | [Architettura della scheda madre](../../notebooks/25_integrated_heterogeneous_routing_it.ipynb) | Router last-token e fallback semantico su sinapsi eterogenee | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/25_integrated_heterogeneous_routing_it.ipynb) |
| 💬 26 | [Demo del chatbot SRA](../../notebooks/26_chatbot_demo_it.ipynb) | Interfaccia utente della chat di lavoro che combina le sinapsi LLM/DB vettoriale/Calcolatrice | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/26_chatbot_demo_it.ipynb) |



## 🎯 Motivazione
SRA risolve i problemi dei modelli monolitici utilizzando un **approccio sparso: "chiamare e combinare dinamicamente solo i moduli necessari (sinapsi) a seconda dell'input"**. Questo permette di imparare più attività senza interferenze.

## 💡 Idea di Base

I modelli IA tipici (come i Transformer) cercano di elaborare tutto utilizzando un unico, gigantesco "cervello". Tuttavia, con questo approccio, il carico computazionale diventa troppo pesante ogni volta che il modello viene reso più intelligente o più grande. Pertanto, SRA adotta un sistema in cui **vengono preparati molti "piccoli cervelli esperti (che SRA chiama 'sinapsi')", e vengono chiamati solo gli esperti necessari a seconda del problema in questione**.

La chiave qui è il meccanismo che decide "quale esperto chiamare". SRA ha un "router (guida)" che seleziona istantaneamente l'esperto che sembra più capace osservando i dati di input. Man mano che ogni esperto diventa più intelligente (impara), questo router impara simultaneamente "chi è quello giusto da scegliere", crescendo per essere in grado di effettuare assegnazioni ottimali in modo automatico.

## 🧠 Panoramica dell'Architettura
1. **Sinapsi:** Unità computazionali indipendenti.
2. **Router:** Seleziona dinamicamente le `Top-k` migliori sinapsi.
3. **Spazio Sinaptico:** Auto-organizzazione per "similarità funzionale".
4. **Regola di Apprendimento Locale:** Utilizza regole locali (Hebbian, STDP) per il bilanciamento.


---

### 🔌 6. Esperimento di Hot-Swap Sinaptico Dinamico e Limiti di Apprendimento del Router
**File:** [`06_hotswap_experiment_demo_it.ipynb`](./06_hotswap_experiment_demo_it.ipynb)

Dimostra il vero potere della SRA: "aggiunta e rimozione dinamica di sinapsi come plugin (Hot-Swap)".
Eseguiamo un esperimento in cui una sinapsi specifica per lo spagnolo viene fusa in un modello di traduzione francese/tedesco in esecuzione.
In questo notebook imparerai l'**importanza cruciale della condivisione e del congelamento dello spazio di conoscenza del modello base (livelli di incorporamento/attenzione, ecc.)** per stabilire un hot-swap. Allo stesso tempo, affronterai la **maggiore barriera della SRA (il problema del gradiente evanescente)**: l'instradamento rigido standard (Top-k) non può apprendere (differenziare) retroattivamente l'instradamento delle sinapsi aggiunte. Questa limitazione funge da prefigurazione critica per la successiva sezione "Gumbel-Softmax (Super Router)".

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_it.ipynb)

---

### 👑 7. Integrazione di Modelli tramite Super Router e Gumbel-Softmax
**File:** [`07_super_router_gumbel_demo_it.ipynb`](./07_super_router_gumbel_demo_it.ipynb)

Costruiamo un "Super Router" che raggruppa più modelli specializzati (un modello FR/DE e un modello ES) e instrada dinamicamente l'elaborazione in base all'input.
Questo dimostra il problema del "Lazy Routing" del semplice Soft Routing e mostra come l'utilizzo di Gumbel-Softmax ottenga un **Hard Routing perfetto**, tagliando del 100% i calcoli inutili del modello.

[![Apri in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_it.ipynb)


## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways](./paper_draft.md)
- [Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion](./paper_hotswap.md)

## 🧪 Esperimenti e Analisi
- [Apprendimento Multitasking e Analisi del Routing](./routing_analysis_algorithmic.md)
- [Analisi del Routing nella Modellazione del Linguaggio Cross-Domain](./routing_analysis_language.md)
- [Analisi di Routing nella Traduzione Multilingue (Ing / Fra / Jap) e Generalizzazione Zero-Shot](../dev/multilingual_translation_routing_analysis.md)
  - Un rapporto affascinante che mostra come SRA assegni automaticamente diversi moduli di traduzione in base alla struttura grammaticale (SVO vs SOV). Ancora più sorprendente, quando deve tradurre una coppia di lingue non appresa, usa inconsciamente l'inglese come "lingua pivot"!
- [Separazione Completa di Percezione e Politica nel Decision Transformer (Apprendimento per Rinforzo)](../dev/decision_transformer_routing_analysis.md)
  - Abbiamo fatto giocare SRA a un gioco. Ha scoperto da solo una struttura modulare incredibile: utilizza esattamente lo stesso modulo di "visione" per percepire l'ambiente in tutte le attività, ma passa a moduli "cervello" completamente diversi a seconda che debba trovare un tesoro o fuggire.
- [Verifica della Traduzione Multilingue Pratica con SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Un rapporto che dimostra che estendendo SRA a un'architettura Encoder-Decoder e addestrandolo per 30.000 passaggi su un corpus reale (opus100), può tradurre espressioni pratiche come "Merci beaucoup." e "Good morning." con BLEU=1.0. L'introduzione della Cross-Attention ha causato un salto dal solo Decoder (BLEU=0) a un BLEU medio complessivo di 0,27 e ha raggiunto un'accuratezza quasi pratica di BLEU=0,56 nella direzione FR→EN.

---

### 📖 8. Demo SRA LLM (Shakespeare)
**File:** [`08_sra_llm_demo_shakespeare_it.ipynb`](../../notebooks/08_sra_llm_demo_shakespeare_it.ipynb)

Questo è un tutorial che utilizza dati Shakespeare su piccola scala per addestrare SRA come modello generativo specifico del decodificatore (LLM). Dopo l'apprendimento, viene utilizzata una mappa termica per visualizzare tramite quale sinapsi è passato ciascun token del testo generato.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_it.ipynb)

---

### 🌐 9. Demo LLM multidominio SRA (codice, matematica, testo)
**File:** [`09_sra_llm_demo_multidomain_it.ipynb`](../../notebooks/09_sra_llm_demo_multidomain_it.ipynb)

Sperimenta la specialità di SRA dell'"apprendimento simultaneo di più domini (codice, matematica, testo)" in un LLM su piccola scala. Puoi verificare come il modello divide (specializza) automaticamente le sinapsi in base ai dati.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_it.ipynb)

---

### 💻 10. Pratico plug-in hot-swap (Zero-Shot)
**File:** [`10_hotswap_plugins_demo_it.ipynb`](../../notebooks/10_hotswap_plugins_demo_it.ipynb)

Dimostreremo un flusso di lavoro in cui più team di sviluppo apprendono in modo indipendente i plug-in per "codice" e "matematica" e li "uniscono fisicamente (hot-swap)" nel modello base dell'ambiente di produzione dopo il fatto. È stato dimostrato che anche dopo la fusione, le perdite di tutti i domini sono esattamente le stesse dell'apprendimento indipendente (Zero Forgetting).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_it.ipynb)

---

### 🗑️ 11. Cancellazione sinaptica dinamica
**File:** [`11_synapse_deletion_demo_it.ipynb`](../../notebooks/11_synapse_deletion_demo_it.ipynb)

Dimostriamo la funzione di SRA, "eliminazione delle sinapsi". Puoi sperimentare sia la "rimozione dei plug-in (pop_synapses)", che elimina fisicamente le sinapsi aggiunte successivamente dalla fine, sia l'"eliminazione di un dominio specifico (clear_synapses)", che cancella e disabilita in modo sicuro le sinapsi che non sono condivise.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_it.ipynb)

