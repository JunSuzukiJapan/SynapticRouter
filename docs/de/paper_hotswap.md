# Hotswappable LLM: Zero-Shot Modulkomposition und Chirurgisches Löschen von Wissen durch die Synaptic Routing Architektur

**Jun Suzuki**, Independent Researcher

## Zusammenfassung (Abstract)
Menschen können postnatal neue Fähigkeiten erlernen (z. B. Fahrradfahren oder eine neue Sprache) und diese als unabhängige Schaltkreise im Gehirn funktionieren lassen, ohne bestehendes Wissen zu zerstören. Aktuelle Massive Language Models (LLMs) besitzen jedoch eine monolithische Struktur, die das gesamte Wissen in einem einzigen Parameterraum speichert. Dies macht das unabhängige Hinzufügen von Wissen oder das Löschen spezifischer Erinnerungen extrem schwierig.

In diesem Paper nutzen wir den Mechanismus der funktionellen Lokalisierung der Synaptic Routing Architecture (SRA), um eine Methode für das dynamische Einfügen und Entfernen neuronaler Schaltkreise (Module) vorzuschlagen und zu validieren: **Hot-Swap**. Hot-Swap ist eine bidirektionale Modulaustauschoperation. Spezialisierte Synapsen, die unabhängig vom Basismodell trainiert wurden, können **ohne jegliches Nachtraining chirurgisch in das Produktionsmodell transplantiert werden (Plug-In)**. Wenn sie nicht mehr benötigt werden, können **bestimmte Erinnerungen sicher getrennt werden (Unplug)**. Die Ergebnisse zeigen, dass wir mit einem Hard-Masking-Mechanismus, inspiriert von Vektordatenbanken, **Zero Forgetting** erreicht haben. Zudem haben wir das „Schwarze-Loch-Problem“ von Nullvektoren bei der Kosinus-Ähnlichkeit entdeckt und gelöst.

## 1. Einleitung (Introduction)
Monolithische LLMs leiden unter katastrophalem Vergessen, aufgeblähten Trainingskosten und der Unmöglichkeit des maschinellen Verlernens (Machine Unlearning).
SRA ermöglicht die operative Innovation des **Hot-Swap**: Plug-In (Transplantation), Unplug (Entfernen/Inaktivieren) und die mathematische Garantie des Zero Forgetting.

## 2. Hintergrund: SRA Architektur
SRA imitiert die räumliche Isolation des Gehirns. Eine zwingende Voraussetzung für das Funktionieren von Hot-Swap ist der **Shared-Trunk-Ansatz**. Alle spezialisierten Synapsen teilen sich das gleiche vortrainierte Basismodell, um eine Repräsentationsdivergenz (Representation Divergence) zwischen den Modellen zu verhindern.

## 3. Hot-Swap: Plug-In (Chirurgische Transplantation von Modulen)
Wir transplantieren ein spezialisiertes Modul einfach, indem wir die PyTorch-Tensoren (Gewichte) in die leeren Slots des Basismodells kopieren. Da das Basismodell eingefroren ist, funktioniert diese physische Kopie perfekt.

## 4. Zero Forgetting: Hard-Mask-Mechanismus
Um zu verhindern, dass der Router alte und neue Synapsen verwechselt, verwenden wir **Pre-Filtering**, inspiriert von Vektordatenbanken. Wir blockieren bei der Inferenz physisch (Hard-Mask) unnötige Pfade mit `-inf`, was mathematisch null Interferenz (Zero Forgetting) garantiert.

## 5. Hot-Swap: Unplug (Entfernen/Inaktivieren bestimmter Erinnerungen)
Es gibt zwei Ansätze, um Module vom Basismodell zu trennen (Unplug):
1. **Physische Trennung (`pop_synapses`)**: Durch Abschneiden am Ende des Tensors.
2. **Löschung durch Inaktivierung (`clear_synapses`)**: Den Inhalt der Synapse auf Null setzen (zero-clear), ohne die Größe des Tensors zu ändern, um die Index-Konsistenz aufrechtzuerhalten.

## 6. Die Kosinus-Ähnlichkeits-Falle: Das Nullvektor-Schwarze-Loch-Problem
Das Nullsetzen einer Synapse führt zu einem Nullvektor mit einem Ähnlichkeits-Score von $0.0$. Wenn normale Scores negativ sind, wählt der Router die leere Synapse (Schwarzes Loch). Ich habe dieses Problem gelöst, indem ich eine $-\infty$-Maske hinzugefügt habe, die auf null gesetzte Synapsen erkennt und ihre Pfade vollständig blockiert.

## 7, 8, 9. Fazit
Der komplette Lebenszyklus (Lernen → Transplantieren → Betreiben → Entfernen → Wiederverwenden) von SRA Hot-Swap ermöglicht es KI-Modellen, sich von monolithischen Black-Boxen zu vitalen modularen Systemen zu entwickeln, in denen Teile der Intelligenz zur sicheren Kontrolle physisch eingesetzt und entfernt werden können.

## Referenzen (References)
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
