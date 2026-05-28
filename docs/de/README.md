[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

## 🎮 Interaktive Demos (Jupyter Notebooks)

Wir haben Jupyter-Notizbücher vorbereitet, in denen Sie die „aufgabenspezifische Gehirnnutzung“ und „Robustheit“ von SRA direkt in Ihrem Browser interaktiv erleben können.Sie können sie in Sekundenschnelle auf Google Colab ausführen, also probieren Sie sie bitte aus!

| # | Demo | Beschreibung | Colab |
|---|------|--------------|-------|
| 🟢 1 | [SRA Quickstart](../../notebooks/01_sra_quickstart_de.ipynb) | Grundlegende SRA-Struktur und Routing-Visualisierung | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/01_sra_quickstart_de.ipynb) |
| 🔵 2 | [Lernen und Routing](../../notebooks/02_learning_and_routing_demo_de.ipynb) | Einzelaufgaben-Lernen und Routing-Spezialisierung | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/02_learning_and_routing_demo_de.ipynb) |
| 🔴 3 | [Multitask-Routing](../../notebooks/03_multitask_routing_demo_de.ipynb) ✨ | Multitask-Lernen und Synapsen-Umschaltung | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/03_multitask_routing_demo_de.ipynb) |
| 🕹️ 4 | [Decision Transformer](../../notebooks/04_decision_transformer_routing_demo_de.ipynb) | Trennung von Wahrnehmung und Aktion in RL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/04_decision_transformer_routing_demo_de.ipynb) |
| 🧠 5 | [Läsions-Experiment](../../notebooks/05_lesion_experiment_demo_de.ipynb) ✨ | Nachweis funktionaler Modularität durch Synapsen-Zerstörung | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/05_lesion_experiment_demo_de.ipynb) |
| 🔌 6 | [Hot-Swap-Experiment](../../notebooks/06_hotswap_experiment_demo_de.ipynb) | Dynamischer Synapsen-Hot-Swap und Router-Lernlimits | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_de.ipynb) |
| 👑 7 | [Super Router (Gumbel)](../../notebooks/07_super_router_gumbel_demo_de.ipynb) | Modellintegration via Gumbel-Softmax | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_de.ipynb) |
| 📖 8 | [SRA LLM (Shakespeare)](../../notebooks/08_sra_llm_demo_shakespeare_de.ipynb) | Tiny LLM mit SRA aufbauen und trainieren | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_de.ipynb) |
| 📚 9 | [Multidomänen-LLM](../../notebooks/09_sra_llm_demo_multidomain_de.ipynb) | Gleichzeitiges Multi-Domänen-Lernen (Code/Math/Text) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_de.ipynb) |
| 💻 10 | [Plugin Hot-Swap](../../notebooks/10_hotswap_plugins_demo_de.ipynb) | Zero-Shot-Hot-Swap (kein katastrophales Vergessen) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_de.ipynb) |
| 🗑️ 11 | [Synapsen-Löschung](../../notebooks/11_synapse_deletion_demo_de.ipynb) | Dynamische Synapsen-Löschung (pop & clear) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_de.ipynb) |
| 🧬 12 | [Entstehung virtueller Neuronen](../../notebooks/12_virtual_neuron_experiment_de.ipynb) | 5 Domänen x 5 Aufgaben offenbaren die autonome Bildung virtueller Neuronen | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/12_virtual_neuron_experiment_de.ipynb) |
| 🧠 13 | [Hot-Swap für virtuelle Neuronen](../../notebooks/13_virtual_neuron_hotswap_de.ipynb) | Sicheres Verlernen auf der Granularität des virtuellen Neurons (Zellanordnung). | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/13_virtual_neuron_hotswap_de.ipynb) |
| 🔬 14 | [Vergleich der Löscheinheiten](../../notebooks/14_compare_deletion_units_de.ipynb) | Löschung von Synapseneinheiten vs. Neuroneneinheiten und Wissensverschränkung | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/14_compare_deletion_units_de.ipynb) |
| 📐 15 | [Kapazitätshypothese](../../notebooks/15_capacity_hypothesis_experiment_de.ipynb) | Synapsenkapazität vs. sichere Verlernschwelle | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/15_capacity_hypothesis_experiment_de.ipynb) |
| 💤 16 | [Verhinderung von Lazy Routing](../../notebooks/16_lazy_routing_prevention_experiment_de.ipynb) | Diagnostizieren und mildern Sie die Faulheit des Routers | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/16_lazy_routing_prevention_experiment_de.ipynb) |
| 🔁 17 | [Routing-Fallback](../../notebooks/17_routing_fallback_experiment_de.ipynb) | Weisen Sie den Datenverkehr neu zu, wenn eine Synapse nicht mehr verfügbar ist | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/17_routing_fallback_experiment_de.ipynb) |
| 🧩 18 | [Benutzerdefinierte Synapsen](../../notebooks/18_custom_synapses_de.ipynb) | Nicht trainierbare Vektor-DB- und Rechnersynapsen | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/18_custom_synapses_de.ipynb) |
| 🎯 19 | [Zero-Shot-Hard-Routing](../../notebooks/19_zero_shot_hard_routing_de.ipynb) | Erzwingen Sie das Routing über „allowed_mask“ ohne erneutes Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/19_zero_shot_hard_routing_de.ipynb) |
| 🛠️ 20 | [Feinabstimmung des Routings](../../notebooks/20_routing_finetuning_de.ipynb) | Autonomes Routing-Lernen durch Feinabstimmung kleiner Datensätze | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/20_routing_finetuning_de.ipynb) |
| 🧯 21 | [Feinabstimmung der Vergessensprüfung](../../notebooks/21_finetuning_forgetting_check_de.ipynb) | Katastrophale Vergessensprüfung nach Routing-Feinabstimmung | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/21_finetuning_forgetting_check_de.ipynb) |
| 🧪 22 | [Neurosymbolische Koexistenz](../../notebooks/22_multi_synapse_hotswap_eval_de.ipynb) | LLM plus Vector DB plus regelbasierter Rechner auf einer Architektur | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/22_multi_synapse_hotswap_eval_de.ipynb) |
| 🦙 23 | [SRA LLM-Integration (TinyLlama)](../../notebooks/nb23_sra_llm_integration_de.ipynb) | Native SRA-Router-Integration mit TinyLlama (PoC) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/nb23_sra_llm_integration_de.ipynb) |
| 🏎️ 24 | [Router-Architektur-Benchmark](../../notebooks/24_router_architecture_benchmark_de.ipynb) | Benchmarking von einstufigen/mehrstufigen/Last-Token-Routern | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/24_router_architecture_benchmark_de.ipynb) |
| 🧰 25 | [Motherboard-Architektur](../../notebooks/25_integrated_heterogeneous_routing_de.ipynb) | Last-Token-Router und semantischer Fallback über heterogene Synapsen | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/25_integrated_heterogeneous_routing_de.ipynb) |
| 💬 26 | [SRA Chatbot-Demo](../../notebooks/26_chatbot_demo_de.ipynb) | Funktionierende Chat-Benutzeroberfläche, die LLM-/Vektor-DB-/Rechner-Synapsen kombiniert | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/26_chatbot_demo_de.ipynb) |


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



---

### 🔌 6. Dynamisches Synapsen-Hot-Swap-Experiment und Router-Lernlimits
**Datei:** [`06_hotswap_experiment_demo_de.ipynb`](./06_hotswap_experiment_demo_de.ipynb)

Demonstriert die wahre Kraft von SRA: "dynamisches Hinzufügen und Entfernen von Synapsen als Plugins (Hot-Swap)".
Wir führen ein Experiment durch, bei dem eine spanischspezifische Synapse in ein laufendes Französisch/Deutsch-Übersetzungsmodell integriert wird.
In diesem Notizbuch lernen Sie die **entscheidende Bedeutung des Teilens und Einfrierens des Wissensraums des Basismodells (Embedding-/Attention-Schichten usw.)** kennen, um einen Hot-Swap herzustellen. Gleichzeitig werden Sie mit der **größten Barriere von SRA (dem Problem des verschwindenden Gradienten)** konfrontiert: Standard-Hard-Routing (Top-k) kann das Routing hinzugefügter Synapsen nicht nachträglich lernen (differenzieren). Diese Einschränkung dient als wichtige Vorahnung für den folgenden Abschnitt "Gumbel-Softmax (Super Router)".

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_de.ipynb)

---

### 👑 7. Modellintegration über Super Router und Gumbel-Softmax
**Datei:** [`07_super_router_gumbel_demo_de.ipynb`](./07_super_router_gumbel_demo_de.ipynb)

Wir bauen einen "Super Router", der mehrere spezialisierte Modelle (ein FR/DE-Modell und ein ES-Modell) bündelt und die Verarbeitung basierend auf der Eingabe dynamisch weiterleitet.
Dies demonstriert das "Lazy Routing"-Problem des einfachen Soft Routings und zeigt, wie die Verwendung von Gumbel-Softmax ein **perfektes Hard Routing** erreicht und unnötige Modellberechnungen um 100 % reduziert.

[![In Colab öffnen](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_de.ipynb)

---

### 📖 8. SRA LLM Demo (Shakespeare)
**File:** [`08_sra_llm_demo_shakespeare_de.ipynb`](../../notebooks/08_sra_llm_demo_shakespeare_de.ipynb)

Dies ist ein Tutorial, das kleine Shakespeare-Daten verwendet, um SRA als decoderspezifisches generatives Modell (LLM) zu trainieren. Nach dem Lernen wird eine Heatmap verwendet, um zu visualisieren, welche Synapsen die einzelnen Token des generierten Textes passieren.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_de.ipynb)

---

### 🌐 9. SRA Multi-Domain LLM Demo (Code, Mathematik, Text)
**File:** [`09_sra_llm_demo_multidomain_de.ipynb`](../../notebooks/09_sra_llm_demo_multidomain_de.ipynb)

Erleben Sie SRAs Spezialität des „gleichzeitigen Lernens mehrerer Domänen (Code, Mathematik, Text)“ in einem kleinen LLM. Sie können überprüfen, wie das Modell Synapsen basierend auf Daten automatisch unterteilt (spezialisiert).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_de.ipynb)

---

### 💻 10. Praktischer Plugin-Hot-Swap (Zero-Shot)
**File:** [`10_hotswap_plugins_demo_de.ipynb`](../../notebooks/10_hotswap_plugins_demo_de.ipynb)

Wir werden einen Arbeitsablauf demonstrieren, in dem mehrere Entwicklungsteams unabhängig voneinander Plug-ins für „Code“ und „Mathematik“ erlernen und diese nachträglich „physisch zusammenführen (Hot-Swap)“ in das Basismodell der Produktionsumgebung integrieren. Es ist erwiesen, dass auch nach der Zusammenführung die Verluste aller Domänen genau die gleichen sind wie beim selbstständigen Lernen (Zero Forgetting).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_de.ipynb)

---

### 🗑️ 11. Dynamische synaptische Löschung
**File:** [`11_synapse_deletion_demo_de.ipynb`](../../notebooks/11_synapse_deletion_demo_de.ipynb)

Wir demonstrieren die Funktion von SRA, „Synapsenlöschung“. Sie können sowohl die „Entfernung von Plug-Ins (pop_synapses)“ erleben, die später hinzugefügte Synapsen physisch am Ende löscht, als auch die „Bereinigung einer bestimmten Domäne (clear_synapses)“, die nicht gemeinsam genutzte Synapsen sicher löscht und deaktiviert.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_de.ipynb)




## 📄 Research Papers

- [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways](./paper_draft.md)
- [Hotswappable LLM: Zero-Shot Module Composition and Surgical Knowledge Deletion](./paper_hotswap.md)

## 🤝 Beitrag & Lizenz

Lizenz: [MIT License](../../LICENSE).
