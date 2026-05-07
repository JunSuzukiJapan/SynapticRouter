[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

## 🎮 Interaktive Demos (Jupyter Notebooks)

Wir haben Jupyter-Notizbücher vorbereitet, in denen Sie die „aufgabenspezifische Gehirnnutzung“ und „Robustheit“ von SRA direkt in Ihrem Browser interaktiv erleben können.Sie können sie in Sekundenschnelle auf Google Colab ausführen, also probieren Sie sie bitte aus!

- [01 SRA Quickstart](../../notebooks/01_sra_quickstart_de.ipynb)
- [02 Learning and Routing Demo](../../notebooks/02_learning_and_routing_demo_de.ipynb)
- [03 Multitask Routing Demo](../../notebooks/03_multitask_routing_demo_de.ipynb)
- [04 Decision Transformer Routing Demo](../../notebooks/04_decision_transformer_routing_demo_de.ipynb)
- [05 Lesion Experiment Demo](../../notebooks/05_lesion_experiment_demo_de.ipynb)


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
- [Routing-Analyse in mehrsprachiger Übersetzung (Eng / Fra / Jap) & Zero-Shot-Generalisierung](../dev/multilingual_translation_routing_analysis.md)
  - Ein faszinierender Bericht, der zeigt, wie SRA je nach grammatikalischer Struktur (SVO vs. SOV) automatisch verschiedene Übersetzungsmodule zuweist. Noch überraschender ist, dass es bei der Übersetzung eines nicht gelernten Sprachpaares unbewusst Englisch als "Pivot-Sprache" verwendet!
- [Vollständige Trennung von Wahrnehmung und Strategie im Decision Transformer (Reinforcement Learning)](../dev/decision_transformer_routing_analysis.md)
  - Wir haben SRA ein Spiel spielen lassen. Es entdeckte selbstständig eine erstaunliche modulare Struktur: Es verwendet exakt dasselbe "Sicht"-Modul zur Wahrnehmung der Umgebung über alle Aufgaben hinweg, wechselt aber zu völlig unterschiedlichen "Gehirn"-Modulen, je nachdem, ob es einen Schatz finden oder fliehen muss.
- [Verifizierung der praktischen mehrsprachigen Übersetzung mit SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Ein Bericht, der zeigt, dass SRA durch die Erweiterung auf eine Encoder-Decoder-Architektur und das Training für 30.000 Schritte auf einem realen Korpus (opus100) praktische Ausdrücke wie "Merci beaucoup." und "Good morning." mit BLEU=1.0 übersetzen kann. Die Einführung von Cross-Attention führte zu einem Sprung von Decoder-only (BLEU=0) zu einem durchschnittlichen BLEU von 0,27 und erreichte eine nahezu praktische Genauigkeit von BLEU=0,56 in der Richtung FR→EN.


## 🤝 Beitrag & Lizenz

Lizenz: [MIT License](../../LICENSE).
