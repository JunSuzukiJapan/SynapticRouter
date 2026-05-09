# Hotswappable LLM: Zero-Shot-Modulkomposition und chirurgische Wissenslöschung via Synaptic Routing Architecture

**Jun Suzuki**, Unabhängiger Forscher

## Abstract
Große Sprachmodelle (LLMs) speichern ihr gesamtes Wissen dicht in einem einzigen monolithischen Parameterraum, was das Hinzufügen oder Entfernen spezifischen Wissens äußerst schwierig macht und der betrieblichen Flexibilität schwerwiegende Einschränkungen auferlegt. In diesem Artikel nutze ich die Modularität der Synaptic Routing Architecture (SRA), um **Hot-Swap** vorzuschlagen und zu validieren — eine bidirektionale Modulaustauchoperation für neuronale Netze. Hot-Swap ermöglicht **Plug-In**: das physische Einfügen unabhängig trainierter spezialisierter Synapsen in ein vortrainiertes Basismodell **ohne jegliches Nachtraining**, und **Unplug**: das chirurgische Entfernen nicht mehr benötigten Wissens. Experimente zeigen, dass ein von Vektordatenbank-Pre-Filtering-Techniken inspirierter Hartmaskierungsmechanismus **Zero Forgetting** erreicht, bei dem der Output-Loss des Basismodells vor und nach der Einfügung bis auf die Dezimalstelle genau übereinstimmt. Zusätzlich entdecke und löse ich das „Schwarzes-Loch-Problem" der Nullvektoren bei der Kosinusähnlichkeit, das bei der Unplug-Operation auftritt, und etabliere damit den vollständigen Lebenszyklus modularer KI: Trainieren → Plug-In → Unplug → Wiederverwenden.

## 1. Introduction

### 1.1 Betriebliche Limitierungen monolithischer Modelle
Seit der Einführung von „Attention Is All You Need" hat die Transformer-Architektur in vielen Bereichen, einschließlich der natürlichen Sprachverarbeitung, eine dominierende Position eingenommen. Allerdings stehen monolithische LLMs mit Hunderten von Milliarden Parametern vor den folgenden kritischen betrieblichen Herausforderungen:

1. **Katastrophales Vergessen (Catastrophic Forgetting)**: Das Feintuning eines Allzweckmodells auf eine spezifische Domäne (z.B. interne Vorschriften, spezialisierter Code) zerstört oder verschlechtert seine ursprünglichen allgemeinen Fähigkeiten.
2. **Eskalierende Trainingskosten**: Jede Wissensergänzung erfordert das Nachtraining des gesamten Modells (oder von Adaptern wie LoRA), was eine vollständig parallele Entwicklung durch mehrere Teams unpraktikabel macht.
3. **Unmöglichkeit der Wissenslöschung**: „Machine Unlearning" — das selektive Vergessen spezifischen Wissens — ist in monolithischen Modellen, in denen Parameter tief verflochten sind, extrem schwierig, und Nachtrainingsversuche zerstören oft unrelated Fähigkeiten.

### 1.2 Beiträge
Ich habe zuvor die Synaptic Routing Architecture (SRA) [Suzuki, 2026] vorgeschlagen, eine sparse Architektur, die aus winzigen unabhängigen Modulen (Synapsen) und einem leichtgewichtigen Router besteht. Während frühere Arbeiten die autonome Aufgabentrennungsfähigkeit des Routers demonstrierten, konzentriert sich dieser Artikel auf die durch SRAs Modularität ermöglichten **betrieblichen Innovationen** — den bidirektionalen **Hot-Swap** von Modulen (Einfügen und Entfernen) — und berichtet drei Beiträge:

1. **Hot-Swap: Plug-In (Einfügen)**: Implementierung und Validierung einer Methode, bei der die Bereitstellung unabhängig trainierter spezialisierter Modelle nur eine physische Tensorkopie in die leeren Slots des Basismodells erfordert.
2. **Hot-Swap: Unplug (Entfernen)**: Design zweier Entfernungs-APIs — physisches Abtrennen (`pop_synapses`) und Null-Löschung (`clear_synapses`) — sowie die Entdeckung und Lösung des „Schwarzes-Loch-Problems" der Nullvektoren bei der Kosinusähnlichkeit.
3. **Experimenteller Beweis des Zero Forgetting**: Nachweis, dass ein von Vektordatenbank-Pre-Filtering inspirierter Hartmaskierungsmechanismus garantiert, dass der Output-Loss des Basismodells vor und nach Einfügen und Entfernen bis auf die Dezimalstelle identisch bleibt.

## 2. Background: SRA Architecture

SRA (Synaptic Routing Architecture) ist eine dynamische, sparse Architektur, inspiriert vom biologischen Gehirn. Dieser Abschnitt skizziert die für das Verständnis von Hot-Swap wesentlichen Komponenten (Details siehe [Suzuki, 2026]).

### 2.1 Router
Der Router, das Herzstück von SRA, ist eine **einzelne lineare Schicht** ohne jeden Attention-Mechanismus. Er berechnet die Kosinusähnlichkeit zwischen dem verborgenen Eingabezustand $h$ und dem Merkmalsvektor (Embedding) $e_i$ jeder Synapse und wählt die Top-k Synapsen aus.

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

wobei $\alpha$ ein Skalierungsfaktor ist.

### 2.2 Tiny Synapses
Jede Synapse ist ein unabhängiges, winziges Modul, bestehend aus einer kleinen Multi-Head-Attention-Schicht und einem MLP. Nur vom Router ausgewählte Synapsen führen Berechnungen durch, was hohe Recheneffizienz erreicht.

### 2.3 Geteilter Stamm (Shared Trunk)
Eine kritische Voraussetzung für Hot-Swap ist der **Shared-Trunk**-Ansatz. Alle spezialisierten Synapsen werden vom selben vortrainierten Basismodell (Embedding-Schichten, Attention-Schichten) abgeleitet, wobei nur die Synapsenkomponenten unabhängig trainiert werden. Dies verhindert die Divergenz interner Vektorrepräsentationen (Representation Divergence) zwischen Modellen und ermöglicht die Synaptentransplantation durch physische Kopie.

## 3. Hot-Swap: Plug-In (Moduleinfügung)

Die erste Operation des Hot-Swap ist **Plug-In** — das Einfügen unabhängig trainierter spezialisierter Module in das Basismodell. Dieser Abschnitt demonstriert, dass der Einfügeprozess vollständig aus PyTorch-Tensoroperationen besteht und extrem einfach ist.

### 3.1 Methode

```python
# hotswap_model: Produktions-Basismodell (mit hinzugefügten leeren Slots)
# plugin_math: Von Mathematik-Team unabhängig trainiertes spezialisiertes LLM

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # Embedding-Vektoren des Routers kopieren
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Expert (TinySynapse) Gewichte (w1, w2) kopieren
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

Die trainierten spezialisierten Modell-Tensoren werden einfach direkt bestimmten Indizes (leeren Slots) der Basismodell-Tensoren zugewiesen. Da SRA über den Shared-Trunk-Ansatz nur Synapsen trainiert, während das gemeinsame Wissen des Basismodells (Attention-Schichten etc.) vollständig eingefroren bleibt, ist diese physische Kopieroperation gültig.

### 3.2 Ermöglichung unabhängiger paralleler Entwicklung
Dieser Ansatz ermöglicht es einem „Code-Team" und einem „Mathematik-Team", ihre spezialisierten Synapsen basierend auf demselben Basismodell mit null gegenseitiger Interferenz unabhängig zu trainieren. Nach dem Training wird die Bereitstellung einfach durch Speicherkopie der Gewichtstensoren in die leeren Slots des Produktions-Basismodells abgeschlossen.

## 4. Zero Forgetting: Hartmaskierung inspiriert vom Vektordatenbank-Pre-Filtering

### 4.1 Herausforderung: Router-Verwirrung
Die einfache physische Tensorkopie birgt das Risiko, dass der Router alte und neue Synapsen verwechselt und möglicherweise die Ausgabe des Basismodells verändert.

### 4.2 Metadaten-Filterung in Vektordatenbanken
Moderne Vektordatenbanken wie Pinecone und Weaviate setzen neben der kosinusähnlichkeitsbasierten semantischen Suche auch Metadaten-Filterung ein.

- **Post-Filtering**: Schließt nicht übereinstimmende Ergebnisse nach der Top-K-Suche aus. Anfällig für „K-NN-Erschöpfung", bei der nach der Filterung unzureichende Ergebnisse verbleiben.
- **Pre-Filtering**: Beschränkt den Suchraum über Metadaten-Masken **vor** der Suche und führt Top-K nur unter qualifizierenden Kandidaten durch. Rauschen wird vollständig eliminiert.

### 4.3 SRAs Pre-Execution-Hartmaske
Der SRA-Router ist im Wesentlichen eine **In-Memory-Vektorsuchmaschine (Maximum Inner Product Search: MIPS)**, die Skalarprodukte zwischen Eingabevektoren und Synapsen-Embedding-Vektoren berechnet.

Ich habe das Vektordatenbank-Pre-Filtering direkt in den Forward-Pass des Routers integriert. Zur Inferenzzeit wird dem Modell eine Metadaten-Maske bereitgestellt, die den „Satz der für die aktuelle Aufgabe zugelassenen Synapsen" spezifiziert.

```python
# Router Forward-Pass
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-Filtering: Logits nicht autorisierter Synapsen auf -Unendlich setzen
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Top-K-Routing
vals, idx = torch.topk(logits, k, dim=-1)
```

Dieses `masked_fill`-Pre-Filtering stellt sicher, dass der Router Experten nur aus zugelassenen Synapsen auswählt. Unabhängig davon, wie viele Gewichte anderer Modelle koexistieren, werden sie im Berechnungsgraphen vollständig ignoriert, was garantiert, dass **der Loss des Basismodells vor und nach der Komposition bis auf die Dezimalstelle genau übereinstimmt (mathematisch null Interferenz)**.

### 4.4 Experimentelle Ergebnisse
Ich verglich den Validation-Loss des Basismodells (Code/Math/Text 3-Domänen-Sprachmodell) vor und nach dem Plug-In unabhängig trainierter spezialisierter Synapsen. Der Loss stimmte bis auf die Dezimalstelle genau überein, was Zero Forgetting empirisch demonstriert.

## 5. Hot-Swap: Unplug (Modulentfernung)

Die zweite Operation des Hot-Swap ist **Unplug** — das Entfernen nicht mehr benötigter Module aus dem Basismodell. Wenn Wissen „eingesteckt" werden kann, ist die Fähigkeit, es „auszustecken", ebenso wesentlich. Machine Unlearning in monolithischen Modellen ist aufgrund der komplexen Verflechtung der Parameter extrem schwierig, aber SRAs modulare Struktur löst dieses Problem durch physische Operationen.

### 5.2 Ansatz 1: Physische Entfernung (pop_synapses)
Wenn per Hot-Swap hinzugefügte Synapsen nicht mehr benötigt werden, werden sie physisch vom Ende des Tensors abgeschnitten.

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**Vorteil**: Der VRAM-Verbrauch wird physisch reduziert, und das Modell kann vollständig in seinen Zustand vor der Erweiterung zurückversetzt werden — wie bei der Deinstallation eines OS-Treibers kann ein physischer Teil des KI-Gehirns entfernt werden.

### 5.3 Ansatz 2: Null-Löschung (clear_synapses)
Beim Entfernen von Synapsen an Zwischenindizes statt am Ende würde physisches Löschen alle nachfolgenden Synapsenindizes verschieben und das Metadaten-Masken-Kontrollsystem zerstören. Stattdessen wird der Synapseninhalt auf null gesetzt, um einen „leeren Slot" zu erzeugen.

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

Durch die Invalidierung nur des Slot-Inhalts ohne Änderung der Tensorgröße wird die Index-Integrität perfekt bewahrt. Leere Slots können später durch Überschreiben mit neuen Synapsen via Hot-Swap wiederverwendet werden.

## 6. The Cosine Similarity Trap: Das Schwarzes-Loch-Problem des Nullvektors

### 6.1 Entdeckung
Bei der Implementierung der Null-Löschung für die Unplug-Operation trat ein kritischer Bug auf, bei dem **die Ausgabe vollständig zusammenbrach**.

### 6.2 Ursachenanalyse
Der SRA-Router führt Routing mittels Kosinusähnlichkeit durch. Der Embedding-Vektor einer null-gelöschten Synapse wird zu $\mathbf{0}$, der auch nach Normalisierung $\mathbf{0}$ bleibt. Die Kosinusähnlichkeit zwischen jedem Eingabevektor $h$ und dem Nullvektor beträgt $0.0$.

Das Problem entsteht, weil die Kosinusähnlichkeit im Bereich $[-1.0, 1.0]$ liegt. Wenn die Kosinusähnlichkeit einer gültigen Synapse negativ ist (z.B. $-0.5$), **erhält die leere Synapse ($0.0$) einen mathematisch höheren Score, wodurch der Router die leere Synapse bevorzugt auswählt**.

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

Daten werden „in den nicht existierenden leeren Slot hineingesaugt und verschwinden" — ein schwarzes-Loch-ähnliches Verhalten.

### 6.3 Lösung: Vollständige Blockade via -∞-Maskierung
Ich habe eine Maskenverarbeitung hinzugefügt, um null-gelöschte Synapsen im Forward-Pass des Routers zu erkennen und auszuschließen.

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Null-gelöschte Synapsen erkennen
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

Die $-\infty$-Maske macht es mathematisch unmöglich, leere Synapsen auszuwählen, unabhängig davon, wie niedrig die Scores anderer Synapsen sein mögen.

## 7. The Complete Lifecycle of Modular AI

Die oben beschriebenen Mechanismen ermöglichen SRA die Realisierung des vollständigen Lebenszyklus modularer KI:

```
Trainieren → Hot-Swap (Komponieren) → Bereitstellen
   ↓                                      ↓
Unabhängige                           Löschen (Purge)
parallele Entwicklung                      ↓
                                    Slot-Wiederverwendung
                                           ↓
                                    Neuer Hot-Swap
```

1. **Trainieren**: Mehrere Teams teilen sich ein Basismodell und entwickeln unabhängig ihre spezialisierten Synapsen parallel.
2. **Komponieren**: Trainierte Tensoren werden physisch in das Produktions-Basismodell für die Bereitstellung kopiert.
3. **Bereitstellen**: Inferenz läuft mit durch Hartmasken-Pre-Filtering garantiertem Zero Forgetting.
4. **Löschen**: Nicht benötigte Synapsen werden physisch entfernt oder null-gelöscht.
5. **Wiederverwenden**: Leere Slots werden durch Hot-Swap neuer spezialisierter Synapsen wiederverwendet.

## 8. Discussion

### 8.1 Repräsentationsdivergenz
Die **absolute Voraussetzung** für Hot-Swap ist, dass alle spezialisierten Synapsen vom selben vortrainierten Basismodell (Shared Trunk) abgeleitet sind. Die Transplantation von Synapsen zwischen vollständig unabhängig trainierten Modellen verursacht Routing-Zusammenbruch aufgrund der Divergenz interner Vektorrepräsentationen.

### 8.2 Super Router als Alternative
Um die Shared-Trunk-Beschränkung zu lockern, wurde ein Ansatz validiert, bei dem unabhängige Gesamtmodelle gekapselt und durch einen Super Router mit Gumbel-Softmax orchestriert werden. Dieser Ansatz erreicht perfektes $1.0$ vs $0.0$ Hard Routing und ermöglicht die vollständige dynamische Umschaltung von Rechenressourcen selbst zwischen Modellen mit unterschiedlichen Architekturen.

### 8.3 Sicherheitsrisiken
Die Hot-Swap-Fähigkeit führt durch ihre Eigenschaft des dynamischen Ladens von Gewichtsdateien von außerhalb eines laufenden Systems neue Sicherheitsbedrohungsvektoren ein. Hauptrisiken umfassen: (1) beliebige Codeausführung über Pickle-Exploits, (2) bösartige Gewichtsinjektion (Backdoor Injection), (3) Routing-Hijacking durch Router-Schlüssel-Fälschung und (4) DoS-Angriffe durch Swap-Thrashing. Gegenmaßnahmen wie das verpflichtende `safetensors`-Format, kryptografische Synapsensignierung und Rate-Limiting werden empfohlen.

### 8.4 Aktuelle Einschränkungen und zukünftige Arbeit
Diese Forschung befindet sich im Experimentalstadium mit kleinskaligen Modellen ($d_\text{model}=128$, $n_\text{layers}=4$). Die Validierung an LLMs der 10B-Klasse bleibt eine wichtige zukünftige Herausforderung. Darüber hinaus erfordert das Router-Synchronisationsproblem — die potenzielle Notwendigkeit eines adaptiven Lernens der Router-Schlüssel beim Hinzufügen von Synapsen mit völlig neuen Fähigkeiten — weitere Untersuchungen.

## 9. Conclusion

In diesem Artikel habe ich Methoden vorgeschlagen und validiert, um LLMs durch Nutzung der Modularität von SRA (Synaptic Routing Architecture) „hotswappable" (dynamisch steckbar) zu machen. Die Plug-In-Operation des Hot-Swap schließt die Bereitstellung allein durch physische Tensorkopie ab, während die Unplug-Operation zwei Entfernungsansätze etabliert: physisches Abtrennen und Null-Löschung. Durch einen von Vektordatenbank-Pre-Filtering inspirierten Hartmaskierungsmechanismus wird Zero Forgetting mathematisch garantiert. Durch die Entdeckung und Lösung des „Schwarzes-Loch-Problems" der Nullvektoren bei der Kosinusähnlichkeit bei der Unplug-Operation wird eine sichere Slot-Wiederverwendung erreicht.

In einer Ära, in der Modelle weiterhin größer und undurchsichtiger werden, stellt der „Hotswappable LLM"-Ansatz — der die physische Plug-and-Unplug-Steuerung von Intelligenzkomponenten ermöglicht — eine äußerst vielversprechende Richtung für Modellwartbarkeit, Sicherheit und betriebliche Effizienz dar.

## References

- Suzuki, J. (2026). [All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

Die vollständigen Hot-Swap- und Synapsenlöschungsprozesse, die in diesem Artikel beschrieben werden, können in den folgenden Google Colab Notebooks interaktiv erlebt werden.

- **Hot-Swap Synapsenkompositions-Demo (Basistraining → Unabhängiges Training → Komposition → Zero Forgetting Beweis)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **Synapsenlöschungs-Demo (Physische Entfernung → Null-Löschung → Lösung des Schwarzen-Loch-Problems)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[Die Zukunft von SRA: Dynamischer Hot-Swap und Erweiterbarkeit](./sra_future_hotswap_ja.md)** — Diskussion über kassettenbasierten Synapsenbetrieb, Personalisierung und verteiltes Lernen.
- **[Sicherheitsrisiken beim SRA Hot-Swap](./sra_security_risks_hotswap_ja.md)** — Bedrohungsvektoren einschließlich Pickle Exploit, Backdoor Injection, DoS-Angriffe und Gegenmaßnahmen.
- **[Repräsentationsdivergenz und hierarchisches Routing](./sra_representation_divergence_ja.md)** — Shared-Trunk-Ansatz und Super-Router-Lösungen.
- **[Hard-Routing-Vergleich für den hierarchischen SRA-Router](./sra_hierarchical_hard_routing_ja.md)** — Vergleichsexperimente von Soft / STE / Gumbel-Softmax.
