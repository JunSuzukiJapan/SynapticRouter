[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)
L'Architettura di Routing Sinaptico (SRA) è una nuova architettura di rete neurale dinamica, sparsa e modulare ispirata al cervello biologico (sinapsi).

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

## 🧪 Esperimenti e Analisi
- [Apprendimento Multitasking e Analisi del Routing](./routing_analysis_algorithmic.md)
- [Analisi del Routing nella Modellazione del Linguaggio Cross-Domain](./routing_analysis_language.md)
- [Analisi di Routing nella Traduzione Multilingue (Ing / Fra / Jap) e Generalizzazione Zero-Shot](../dev/multilingual_translation_routing_analysis.md)
  - Un rapporto affascinante che mostra come SRA assegni automaticamente diversi moduli di traduzione in base alla struttura grammaticale (SVO vs SOV). Ancora più sorprendente, quando deve tradurre una coppia di lingue non appresa, usa inconsciamente l'inglese come "lingua pivot"!
- [Separazione Completa di Percezione e Politica nel Decision Transformer (Apprendimento per Rinforzo)](../dev/decision_transformer_routing_analysis.md)
  - Abbiamo fatto giocare SRA a un gioco. Ha scoperto da solo una struttura modulare incredibile: utilizza esattamente lo stesso modulo di "visione" per percepire l'ambiente in tutte le attività, ma passa a moduli "cervello" completamente diversi a seconda che debba trovare un tesoro o fuggire.
- [Verifica della Traduzione Multilingue Pratica con SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Un rapporto che dimostra che estendendo SRA a un'architettura Encoder-Decoder e addestrandolo per 30.000 passaggi su un corpus reale (opus100), può tradurre espressioni pratiche come "Merci beaucoup." e "Good morning." con BLEU=1.0. L'introduzione della Cross-Attention ha causato un salto dal solo Decoder (BLEU=0) a un BLEU medio complessivo di 0,27 e ha raggiunto un'accuratezza quasi pratica di BLEU=0,56 nella direzione FR→EN.
