# Analisi del Routing nella Modellazione del Linguaggio Cross-Domain (Code / Math / Text)

## Panoramica
Abbiamo condotto un esperimento con SRA per apprendere tre domini (Code, Math, Text) con vocabolari diversi simultaneamente.

## Qualità dell'Inferenza
Dopo 1000 passi, sono state ottenute previsioni precise senza interferenze.

## Analisi della Specializzazione
1. **Omogeneizzazione Iniziale**: All'inizio, i compiti usavano uniformemente le 16 sinapsi.
2. **Specializzazione**: Il routing si è chiaramente separato.
   - `Code` ha usato la **Sinapsi 8**.
   - `Math` ha usato le **Sinapsi 10 e 13**.
   - `Text` ha usato le **Sinapsi 0 e 15**.

## Conclusione
Il modello ha scoperto autonomamente "l'uso delle sinapsi giuste per ogni compito", prevenendo la dimenticanza catastrofica.
