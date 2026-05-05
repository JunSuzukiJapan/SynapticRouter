# Apprendimento Multitasking e Analisi del Routing

## Panoramica
Abbiamo verificato se SRA può allocare autonomamente esperti (sinapsi) a seconda della natura delle attività (`copy`, `reverse`, `paren`, `addmod`).

## Risultati dell'Addestramento
Dopo 10.000 passi, abbiamo ottenuto il 100% di precisione, dimostrando che SRA previene la dimenticanza catastrofica.

## Analisi del Routing
1. **Gruppo di Operazione Sequenziale**: `COPY` e `REVERSE` (0.969 di similarità).
2. **Gruppo di Calcolo/Logica**: `PAREN` e `ADDMOD` (0.858 di similarità).

## Conclusione
Il router SRA **discrimina autonomamente le attività**, raggruppando le sinapsi per similarità.
