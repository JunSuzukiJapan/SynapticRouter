[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) ist eine neuartige dynamische, spärliche (sparse) und modulare neuronale Netzwerkarchitektur, die vom biologischen Gehirn (Synapsen) inspiriert ist.
Anstelle eines massiven, statischen Transformers leitet SRA Eingaben dynamisch an geeignete "Synapsen" (winzige Module) weiter.

## 🎯 Motivation

Da KI-Modelle immer massiver werden, stehen monolithische Netzwerke vor Problemen wie "steigenden Rechenressourcen" und dem "katastrophalen Vergessen beim Multitasking-Lernen".
SRA löst dies mit einem **spärlichen Ansatz: "Dynamisches Aufrufen und Kombinieren nur der erforderlichen winzigen Module (Synapsen) je nach Eingabe."**

## 🧠 Architekturübersicht

1. **Synapse (Synapsenmodul)**
   - Unabhängige, winzige Berechnungseinheiten (z. B. Miniatur-Transformer).
2. **Router**
   - Wählt dynamisch nur die `Top-k` optimalen Synapsen basierend auf Eingabe-Tokens aus.
3. **Synapsenraum (Synapse Space)**
   - Synapsen ordnen sich selbst so an, dass Entfernungen "funktionale Ähnlichkeit" darstellen.
4. **Lokale Lernregel**
   - Nutzt zusätzlich zur Standard-Backpropagation lokale 3-Faktor-Regeln.

## 📁 Verzeichnisstruktur

- `src/` : Kernimplementierungen des SRA-Modells.
- `docs/` : Architekturberichte und Experimente.
- `data/` : Datensätze für Training und Validierung.
- `tests/` : Testcode.

## 🚀 Nutzung

```bash
pip install torch
python src/sra_experiment.py --task reverse --steps 2000
```

## 🧪 Experimente & Analyse

- [Multitask-Lernen und Routing-Analyse beim algorithmischen Denken](./routing_analysis_algorithmic.md)
- [Routing-Analyse in der domänenübergreifenden Sprachmodellierung](./routing_analysis_language.md)

## 🤝 Beitrag & Lizenz

Lizenz: [MIT License](../../LICENSE).
