[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) ist eine neuartige dynamische, spärliche (sparse) und modulare neuronale Netzwerkarchitektur, die vom biologischen Gehirn (Synapsen) inspiriert ist.
Anstelle eines massiven, statischen Transformers leitet SRA Eingaben dynamisch an geeignete "Synapsen" (winzige Module) weiter.

## 🎯 Motivation

Da KI-Modelle immer massiver werden, stehen monolithische Netzwerke vor Problemen wie "steigenden Rechenressourcen" und dem "katastrophalen Vergessen beim Multitasking-Lernen".
SRA löst dies mit einem **spärlichen Ansatz: "Dynamisches Aufrufen und Kombinieren nur der erforderlichen winzigen Module (Synapsen) je nach Eingabe."**

## 💡 Grundidee

Typische KI-Modelle (wie Transformer) versuchen, alles mit einem einzigen, riesigen „Gehirn“ zu verarbeiten. Bei diesem Ansatz wird die Rechenlast jedoch jedes Mal viel zu schwer, wenn das Modell intelligenter oder größer gemacht wird. Daher verwendet SRA ein System, bei dem **viele „kleine Expertengehirne (die SRA als ‚Synapsen‘ bezeichnet)“ vorbereitet werden und je nach vorliegendem Problem nur die benötigten Experten aufgerufen werden**.

Der Schlüssel hier ist der Mechanismus, der entscheidet, „welcher Experte aufgerufen werden soll“. SRA verfügt über einen „Router (Leitfaden)“, der sofort den fähigsten Experten auswählt, indem er die Eingabedaten betrachtet. Während jeder Experte intelligenter wird (lernt), lernt dieser Router gleichzeitig, „wer der Richtige ist“, und wächst heran, um optimale Zuweisungen automatisch vornehmen zu können.

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
