# Routing-Analyse in der domänenübergreifenden Sprachmodellierung (Code / Mathe / Text)

## Übersicht
Wir haben mit SRA ein Experiment zur "Sprachmodellierung" durchgeführt, um gleichzeitig drei Domänen (Code, Mathematik, Text) zu erlernen.

## Bewertung der Inferenzqualität
Nach 1000 Schritten wurden genaue Vorhersagen ohne Interferenzen erreicht.

## Analyse der Synapsen-Spezialisierung
1. **Homogenisierung im Anfangszustand**: Zunächst nutzte jede Aufgabe alle 16 Synapsen gleichmäßig.
2. **Spezialisierung**: Später trennte sich das Routing deutlich auf.
   - `Code` nutzte **Synapse 8**.
   - `Math` nutzte **Synapse 10 und 13**.
   - `Text` nutzte **Synapse 0 und 15**.

## Fazit
Das SRA-Modell entdeckte autonom die "richtige Synapsennutzung für jede Aufgabe" und verhinderte katastrophales Vergessen.
