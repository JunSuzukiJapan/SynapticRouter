[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)
L'Architettura di Routing Sinaptico (SRA) è una nuova architettura di rete neurale dinamica, sparsa e modulare ispirata al cervello biologico (sinapsi).

## 🎯 Motivazione
SRA risolve i problemi dei modelli monolitici utilizzando un **approccio sparso: "chiamare e combinare dinamicamente solo i moduli necessari (sinapsi) a seconda dell'input"**. Questo permette di imparare più attività senza interferenze.

## 🧠 Panoramica dell'Architettura
1. **Sinapsi:** Unità computazionali indipendenti.
2. **Router:** Seleziona dinamicamente le `Top-k` migliori sinapsi.
3. **Spazio Sinaptico:** Auto-organizzazione per "similarità funzionale".
4. **Regola di Apprendimento Locale:** Utilizza regole locali (Hebbian, STDP) per il bilanciamento.

## 🧪 Esperimenti e Analisi
- [Apprendimento Multitasking e Analisi del Routing](./routing_analysis_algorithmic.md)
- [Analisi del Routing nella Modellazione del Linguaggio Cross-Domain](./routing_analysis_language.md)
