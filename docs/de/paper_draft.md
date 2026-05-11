# Neuro-inspiriertes Synaptisches Routing: Überwindung des Katastrophalen Vergessens durch Dynamische Modulare Pfade

**Jun Suzuki**, Independent Researcher

## Zusammenfassung (Abstract)
Das menschliche Gehirn kann grundlegend verschiedene Aufgaben – wie Gehen, Sprechen und Rechnen – lernen und ausführen, ohne dass diese sich gegenseitig stören. Dies liegt daran, dass die neuronalen Schaltkreise (Synapsen) des Gehirns je nach Aufgabe dynamisch geroutet werden, wodurch eine räumlich isolierte „funktionelle Lokalisierung (Functional Localization)“ aufrechterhalten wird. Im Gegensatz dazu leiden Künstliche Neuronale Netze (ANNs), wenn sie mehrere Aufgaben in einem einzigen monolithischen Netzwerk lernen, unter „Katastrophalem Vergessen (Catastrophic Forgetting)“, bei dem vergangene Erinnerungen zerstört werden.

In diesem Paper schlagen wir die „Synaptic Routing Architecture (SRA)“ vor, ein Modell für kontinuierliches Lernen, das von den biologischen Mechanismen der dynamischen Synapsenbildung und der räumlichen Isolation inspiriert ist. SRA besteht aus einem extrem einfachen „Router (Router)“ mit nur einer Schicht und mehreren unabhängigen, winzigen Modulen (Synapsen). Durch unsere Experimente zeigen wir, dass SRA die Natur einer Aufgabe anhand der Eingabe autonom erkennen kann – ohne dass während der Inferenz eine externe Task-ID angegeben werden muss – und **sowohl das Routing (die Pfadauswahl) als auch die Repräsentationen der Aufgabe vollständig End-to-End und gleichzeitig lernen kann.** Wir zeigen, dass ohne künstliches Einfrieren von Gewichten oder komplexe evolutionäre Algorithmen eine autonome funktionelle Lokalisierung innerhalb des Modells entsteht, die katastrophales Vergessen vollständig vermeidet.

## 1. Einleitung (Introduction)
Im Bereich Deep Learning ist das „Kontinuierliche Lernen (Continual Learning)“ eine der größten Hürden auf dem Weg zur künstlichen allgemeinen Intelligenz (AGI). Monolithische Netzwerke, wie die aktuellen massiven Transformer-Modelle, vergessen zwangsläufig zuvor erlerntes Wissen, wenn sie auf Daten aus neuen Domänen feinabgestimmt werden.

Um dieses Problem zu lösen, konzentriert sich unsere Forschung auf die „funktionelle Lokalisierung“ des Gehirns. Genau wie die Sprach- und Motorikzentren des Gehirns physisch unterschiedliche Schaltkreise nutzen, um Störungen zu vermeiden, ist SRA als Architektur konzipiert, die winzige unabhängige Netzwerke (Synapsen) über einen dynamischen Routing-Mechanismus dynamisch ein- und ausschalten (Plug-In/Unplug) kann.

## 2. Verwandte Arbeiten und die Neuheit von SRA
Bestehende Ansätze zur Verhinderung von katastrophalem Vergessen umfassen Regularisierungsmethoden wie EWC (Elastic Weight Consolidation). Diese Methoden sind jedoch durch die Kapazität des Modells begrenzt. Ein eher „struktureller und modularer“ Ansatz ist **PathNet (2017)** von Google DeepMind, der genetische Algorithmen verwendet und Gewichte nach dem Lernen einfriert.

### Der überwältigende Vorteil von SRA (Gleichzeitiges Lernen)
Im Vergleich zu herkömmlichen Ansätzen wie PathNet liegt die fundamentale Neuheit von SRA in der Fähigkeit, **„die Entdeckung von Pfaden (Routing) und die Modulrepräsentationen auf differenzierbare, End-to-End-Weise gleichzeitig zu lernen.“**
1. **Autonomes Routing (Task-Agnostic):** PathNet benötigt eine Task-ID. Der einzelne lineare Router von SRA bestimmt die Domäne autonom basierend auf der Kosinus-Ähnlichkeit der Eingabemerkmale.
2. **Kein Einfrieren von Gewichten oder evolutionäre Algorithmen:** SRA nutzt ausschließlich Standard-Backpropagation.
3. **Entstehung Dynamischer Funktioneller Lokalisierung:** Räumliche Isolation entsteht natürlich durch spärliche Aktivierung (Sparse Activation).

## 3. Architektur (Neuro-inspired Design)
SRA ist eine dynamische und spärliche Architektur, die die Synapsenbildung des biologischen Gehirns nachahmt.

### 3.1 Der Router (Dynamic Synaptic Formation)
Der Router ist eine einfache lineare Schicht, die die Kosinus-Ähnlichkeit zwischen Eingabemerkmalen und dem „Embedding-Vektor“ jeder Synapse berechnet, um die Top-k Synapsen auszuwählen, die „feuern“ sollen.

### 3.2 Winzige Synapsen (Functional Modules)
Jede Synapse besteht aus einem unabhängigen, extrem kleinen Multi-Head Attention und MLP. Nur vom Router ausgewählte Synapsen führen Berechnungen durch; andere Parameter bleiben unbeeinflusst.

## 4, 5, 6, 7. Experimente
Wir führten Experimente zu algorithmischem Schlussfolgern, domänenübergreifender Sprachmodellierung, mehrsprachiger maschineller Übersetzung und Offline-Reinforcement-Learning (Decision Transformer) durch. In jedem Fall führte **das gleichzeitige Lernen des Routers zu einer autonomen modularen Trennung, ähnlich der funktionellen Lokalisierung des Gehirns**, ganz ohne die Überwachung durch Task-IDs.

## 8. Fazit (Conclusion)
In diesem Paper haben wir durch SRA das Potenzial für einen Paradigmenwechsel aufgezeigt: von „traditionellen statischen neuronalen Netzen“, die alle Parameter über alle Aufgaben hinweg teilen, zu einem „modularen Netzwerk, das mit biologischer räumlicher Isolation und dynamischem Routing ausgestattet ist.“ Es ist ein entscheidender Schritt in Richtung einer skalierbaren Künstlichen Allgemeinen Intelligenz (AGI).

## Referenzen (References)
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
