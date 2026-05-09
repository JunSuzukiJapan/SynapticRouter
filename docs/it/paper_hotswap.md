# Hotswappable LLM: Composizione di Moduli Zero-Shot e Cancellazione Chirurgica della Conoscenza tramite Synaptic Routing Architecture

**Jun Suzuki**, Ricercatore Indipendente

## Abstract
I grandi modelli linguistici (LLM) memorizzano tutta la conoscenza in modo denso all'interno di un singolo spazio di parametri monolitico, rendendo estremamente difficile aggiungere o rimuovere conoscenza specifica e imponendo severe limitazioni alla flessibilità operativa. In questo articolo, sfrutto la modularità della Synaptic Routing Architecture (SRA) per proporre e validare l'**Hot-Swap** — un'operazione bidirezionale di scambio di moduli per le reti neurali. L'Hot-Swap consente il **Plug-In**: inserire fisicamente sinapsi specializzate addestrate indipendentemente in un modello base pre-addestrato **senza alcun riaddestramento**, e l'**Unplug**: rimuovere chirurgicamente la conoscenza non più necessaria. Gli esperimenti dimostrano che un meccanismo di mascheramento rigido ispirato alle tecniche di pre-filtraggio dei database vettoriali raggiunge il **Zero Forgetting**, dove la loss di output del modello base corrisponde esattamente fino al punto decimale prima e dopo l'inserimento. Inoltre, scopro e risolvo il "Problema del Buco Nero" dei vettori zero nella similarità del coseno incontrato durante l'operazione Unplug, stabilendo il ciclo di vita completo dell'IA modulare: Addestrare → Plug-In → Unplug → Riutilizzare.

## 1. Introduction

### 1.1 Limitazioni operative dei modelli monolitici
Dall'introduzione di "Attention Is All You Need", l'architettura Transformer ha stabilito una posizione dominante in molti domini, inclusa l'elaborazione del linguaggio naturale. Tuttavia, i LLM monolitici con centinaia di miliardi di parametri affrontano le seguenti sfide operative critiche:

1. **Oblio catastrofico (Catastrophic Forgetting)**: Il fine-tuning di un modello generico su un dominio specifico (es. regolamenti interni, codice specializzato) distrugge o degrada le sue capacità generali originali.
2. **Escalation dei costi di addestramento**: Ogni aggiunta di conoscenza richiede il riaddestramento dell'intero modello (o di adattatori come LoRA), rendendo impraticabile lo sviluppo completamente parallelo da parte di più team.
3. **Impossibilità di cancellazione della conoscenza**: Il "Machine Unlearning" — dimenticare selettivamente conoscenza specifica — è estremamente difficile nei modelli monolitici dove i parametri sono profondamente intrecciati, e i tentativi di riaddestramento spesso distruggono capacità non correlate.

### 1.2 Contributi
Ho precedentemente proposto la Synaptic Routing Architecture (SRA) [Suzuki, 2026], un'architettura sparsa composta da moduli indipendenti minuscoli (sinapsi) e un router leggero. Mentre i lavori precedenti hanno dimostrato la capacità di separazione autonoma dei compiti del router, questo articolo si concentra sulle **innovazioni operative** abilitate dalla modularità di SRA — l'**Hot-Swap** bidirezionale di moduli (inserimento e rimozione) — riportando tre contributi:

1. **Hot-Swap: Plug-In (Inserimento)**: Implementazione e validazione di un metodo dove il deployment di modelli specializzati addestrati indipendentemente richiede solo una copia fisica dei tensori negli slot vuoti del modello base.
2. **Hot-Swap: Unplug (Rimozione)**: Progettazione di due API di rimozione — distacco fisico (`pop_synapses`) e purga tramite azzeramento (`clear_synapses`) — e la scoperta e risoluzione del "Problema del Buco Nero" dei vettori zero nella similarità del coseno.
3. **Prova sperimentale del Zero Forgetting**: Dimostrazione che un meccanismo di mascheramento rigido ispirato al pre-filtraggio dei database vettoriali garantisce che la loss di output del modello base rimanga identica fino al punto decimale prima e dopo sia l'inserimento che la rimozione.

## 2. Background: SRA Architecture

SRA (Synaptic Routing Architecture) è un'architettura dinamica e sparsa ispirata al cervello biologico. Questa sezione delinea i componenti essenziali per comprendere l'Hot-Swap (vedere [Suzuki, 2026] per i dettagli).

### 2.1 Router
Il router, cuore di SRA, è un **singolo strato lineare** senza alcun meccanismo di Attention. Calcola la similarità del coseno tra lo stato nascosto di input $h$ e il vettore di caratteristiche (embedding) $e_i$ di ogni sinapsi, selezionando le Top-k sinapsi.

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

dove $\alpha$ è un fattore di scala.

### 2.2 Tiny Synapses
Ogni sinapsi è un modulo minuscolo indipendente composto da un piccolo strato Multi-Head Attention e un MLP. Solo le sinapsi selezionate dal router eseguono calcoli, raggiungendo un'elevata efficienza computazionale.

### 2.3 Tronco condiviso (Shared Trunk)
Un prerequisito critico per l'Hot-Swap è l'approccio del **Tronco Condiviso**. Tutte le sinapsi specializzate sono derivate dallo stesso modello base pre-addestrato (strati di Embedding, strati di Attention), addestrando indipendentemente solo i componenti sinaptici. Ciò previene la divergenza nelle rappresentazioni vettoriali interne (Representation Divergence) tra i modelli e consente il trapianto di sinapsi tramite copia fisica.

## 3. Hot-Swap: Plug-In (Inserimento del modulo)

La prima operazione dell'Hot-Swap è il **Plug-In** — inserire moduli specializzati addestrati indipendentemente nel modello base. Questa sezione dimostra che il processo di inserimento consiste interamente in operazioni tensoriali PyTorch ed è estremamente semplice.

### 3.1 Metodo

```python
# hotswap_model: Modello base di produzione (con slot vuoti aggiunti)
# plugin_math: LLM specializzato in matematica addestrato indipendentemente dal team di matematica

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # Copiare i vettori di embedding del router
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Copiare i pesi Expert (TinySynapse) (w1, w2)
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

I tensori del modello specializzato addestrato vengono semplicemente assegnati direttamente a indici specifici (slot vuoti) dei tensori del modello base. Poiché SRA addestra solo le sinapsi mantenendo la conoscenza condivisa del modello base (strati di Attention, ecc.) completamente congelata tramite l'approccio del Tronco Condiviso, questa operazione di copia fisica è valida.

### 3.2 Abilitazione dello sviluppo parallelo indipendente
Questo approccio consente a un "team Codice" e un "team Matematica" di addestrare indipendentemente le proprie sinapsi specializzate basandosi sullo stesso modello base con zero interferenza reciproca. Dopo l'addestramento, il deployment viene completato semplicemente copiando in memoria i tensori dei pesi negli slot vuoti del modello base di produzione.

## 4. Zero Forgetting: Mascheramento rigido ispirato dal Pre-filtraggio dei database vettoriali

### 4.1 Sfida: Confusione del router
La semplice copia fisica dei tensori rischia che il router confonda vecchie e nuove sinapsi, alterando potenzialmente l'output del modello base.

### 4.2 Filtraggio per metadati nei database vettoriali
I database vettoriali moderni come Pinecone e Weaviate impiegano il filtraggio per metadati insieme alla ricerca semantica basata sulla similarità del coseno.

- **Post-filtraggio**: Esclude i risultati non corrispondenti dopo la ricerca Top-K. Soggetto all'"esaurimento K-NN" dove restano risultati insufficienti dopo il filtraggio.
- **Pre-filtraggio**: Restringe lo spazio di ricerca tramite maschere di metadati **prima** della ricerca, eseguendo il Top-K solo tra i candidati qualificati. Il rumore viene completamente eliminato.

### 4.3 Maschera rigida di pre-esecuzione di SRA
Il router SRA è essenzialmente un **motore di ricerca vettoriale in memoria (Maximum Inner Product Search: MIPS)** che calcola prodotti scalari tra vettori di input e vettori di embedding delle sinapsi.

Ho incorporato il Pre-filtraggio dei database vettoriali direttamente nella forward pass del router. Al momento dell'inferenza, viene fornita al modello una maschera di metadati che specifica l'"insieme di sinapsi autorizzate per il compito corrente".

```python
# Forward pass del router
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pre-filtraggio: Impostare a -infinito i logit delle sinapsi non autorizzate
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Routing Top-K
vals, idx = torch.topk(logits, k, dim=-1)
```

Questo Pre-filtraggio tramite `masked_fill` garantisce che il router selezioni esperti solo tra le sinapsi autorizzate. Indipendentemente da quanti pesi di altri modelli coesistano, vengono completamente ignorati nel grafo computazionale, garantendo che **la loss del modello base corrisponda esattamente fino al punto decimale prima e dopo la composizione (interferenza matematicamente zero)**.

### 4.4 Risultati sperimentali
Ho confrontato la Validation Loss del modello base (modello linguistico a 3 domini Code/Math/Text) prima e dopo il Plug-In di sinapsi specializzate addestrate indipendentemente. La loss corrispondeva esattamente fino al punto decimale, dimostrando empiricamente il Zero Forgetting.

## 5. Hot-Swap: Unplug (Rimozione del modulo)

La seconda operazione dell'Hot-Swap è l'**Unplug** — rimuovere dal modello base i moduli che non sono più necessari. Se la conoscenza può essere "inserita", la capacità di "rimuoverla" è ugualmente essenziale. Il Machine Unlearning nei modelli monolitici è estremamente difficile a causa dell'intreccio complesso dei parametri, ma la struttura modulare di SRA risolve questo problema tramite operazioni fisiche.

### 5.2 Approccio 1: Rimozione fisica (pop_synapses)
Quando le sinapsi aggiunte tramite Hot-Swap non sono più necessarie, vengono fisicamente tagliate dalla fine del tensore.

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**Vantaggio**: L'utilizzo di VRAM viene fisicamente ridotto, e il modello può essere completamente ripristinato al suo stato pre-aggiunta — come disinstallare un driver del SO, una parte fisica del cervello dell'IA può essere rimossa.

### 5.3 Approccio 2: Purga tramite azzeramento (clear_synapses)
Quando si rimuovono sinapsi a indici intermedi anziché alla fine, la cancellazione fisica sposterebbe tutti gli indici delle sinapsi successive, compromettendo il sistema di controllo tramite maschera di metadati. Invece, il contenuto delle sinapsi viene azzerato per creare uno "slot vuoto".

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

Invalidando solo il contenuto degli slot senza modificare la dimensione del tensore, l'integrità degli indici viene perfettamente preservata. Gli slot vuoti possono essere successivamente riutilizzati sovrascrivendoli con nuove sinapsi tramite Hot-Swap.

## 6. The Cosine Similarity Trap: Il problema del buco nero del vettore zero

### 6.1 Scoperta
Nell'implementare la purga tramite azzeramento per l'operazione Unplug, ho incontrato un bug critico in cui **l'output è completamente collassato**.

### 6.2 Analisi della causa radice
Il router SRA esegue il routing usando la similarità del coseno. Il vettore di embedding di una sinapsi azzerata diventa $\mathbf{0}$, che rimane $\mathbf{0}$ anche dopo la normalizzazione. La similarità del coseno tra qualsiasi vettore di input $h$ e il vettore zero è $0.0$.

Il problema sorge perché la similarità del coseno ha un range di $[-1.0, 1.0]$. Se la similarità del coseno di una sinapsi valida è negativa (es. $-0.5$), **la sinapsi vuota ($0.0$) ottiene un punteggio matematicamente più alto, causando la selezione preferenziale della sinapsi vuota da parte del router**.

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

I dati vengono "risucchiati e scompaiono in" quello che dovrebbe essere uno slot vuoto inesistente — un comportamento simile a un buco nero.

### 6.3 Soluzione: Blocco completo tramite mascheramento -∞
Ho aggiunto un'elaborazione di maschera per rilevare ed escludere le sinapsi azzerate nella forward pass del router.

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Rilevare le sinapsi azzerate
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

La maschera $-\infty$ rende matematicamente impossibile la selezione delle sinapsi vuote, indipendentemente da quanto bassi possano essere i punteggi delle altre sinapsi.

## 7. The Complete Lifecycle of Modular AI

I meccanismi sopra descritti consentono a SRA di realizzare il ciclo di vita completo dell'IA modulare:

```
Addestrare → Hot-Swap (Comporre) → Servire
   ↓                                   ↓
Sviluppo                          Eliminare (Purge)
parallelo indipendente                 ↓
                              Riutilizzo degli slot
                                       ↓
                                Nuovo Hot-Swap
```

1. **Addestrare**: Più team condividono un modello base e sviluppano indipendentemente le proprie sinapsi specializzate in parallelo.
2. **Comporre**: I tensori addestrati vengono fisicamente copiati nel modello base di produzione per il deployment.
3. **Servire**: L'inferenza viene eseguita con Zero Forgetting garantito dal pre-filtraggio con maschera rigida.
4. **Eliminare**: Le sinapsi non necessarie vengono fisicamente rimosse o azzerate per la purga.
5. **Riutilizzare**: Gli slot vuoti vengono riutilizzati inserendo nuove sinapsi specializzate tramite Hot-Swap.

## 8. Discussion

### 8.1 Divergenza di rappresentazione (Representation Divergence)
Il **prerequisito assoluto** per l'Hot-Swap è che tutte le sinapsi specializzate siano derivate dallo stesso modello base pre-addestrato (Tronco Condiviso). Il trapianto di sinapsi tra modelli addestrati in modo completamente indipendente causa il collasso del routing a causa della divergenza nelle rappresentazioni vettoriali interne.

### 8.2 Super Router come alternativa
Per allentare il vincolo del Tronco Condiviso, è stato validato un approccio in cui modelli interi indipendenti vengono incapsulati e orchestrati da un Super Router che utilizza Gumbel-Softmax. Questo approccio raggiunge un routing rigido perfetto $1.0$ vs $0.0$, consentendo la commutazione dinamica completa delle risorse computazionali anche tra modelli con architetture diverse.

### 8.3 Rischi di sicurezza
La capacità Hot-Swap introduce nuovi vettori di minacce alla sicurezza a causa della sua proprietà di caricare dinamicamente file di pesi dall'esterno di un sistema in esecuzione. I rischi principali includono: (1) esecuzione arbitraria di codice tramite exploit Pickle, (2) iniezione malevola di pesi (Backdoor Injection), (3) hijacking del routing tramite falsificazione delle chiavi del router, e (4) attacchi DoS tramite swap thrashing. Sono raccomandate mitigazioni come il formato obbligatorio `safetensors`, la firma crittografica delle sinapsi e la limitazione della frequenza.

### 8.4 Limitazioni attuali e lavori futuri
Questa ricerca è in fase sperimentale con modelli a piccola scala ($d_\text{model}=128$, $n_\text{layers}=4$). La validazione su LLM di classe 10B rimane un'importante sfida futura. Inoltre, il problema della sincronizzazione del router — la potenziale necessità di un apprendimento adattivo delle chiavi del router quando si aggiungono sinapsi con capacità completamente nuove — richiede ulteriori indagini.

## 9. Conclusion

In questo articolo, ho proposto e validato metodi per rendere i LLM "Hotswappable" (dinamicamente collegabili) sfruttando la modularità di SRA (Synaptic Routing Architecture). L'operazione Plug-In dell'Hot-Swap completa il deployment attraverso la sola copia fisica dei tensori, mentre l'operazione Unplug stabilisce due approcci di rimozione: distacco fisico e purga tramite azzeramento. Attraverso un meccanismo di mascheramento rigido ispirato dal Pre-filtraggio dei database vettoriali, il Zero Forgetting è garantito matematicamente. Scoprendo e risolvendo il "Problema del Buco Nero" dei vettori zero nella similarità del coseno incontrato durante l'Unplug, viene raggiunto il riutilizzo sicuro degli slot.

In un'era in cui i modelli continuano a crescere e diventare più opachi, l'approccio "Hotswappable LLM" — che consente il controllo fisico di collegamento e scollegamento dei componenti dell'intelligenza — rappresenta una direzione estremamente promettente per la manutenibilità, la sicurezza e l'efficienza operativa dei modelli.

## References

- Suzuki, J. (2026). [All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

I processi completi di Hot-Swap e cancellazione delle sinapsi descritti in questo articolo possono essere sperimentati interattivamente nei seguenti notebook Google Colab.

- **Demo di composizione sinaptica Hot-Swap (Addestramento base → Addestramento indipendente → Composizione → Prova del Zero Forgetting)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **Demo di cancellazione delle sinapsi (Rimozione fisica → Azzeramento → Risoluzione del problema del buco nero)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[Il futuro di SRA: Hot-Swap dinamico ed estensibilità](./sra_future_hotswap_ja.md)** — Discussione sull'operazione di sinapsi in modalità cassetta, personalizzazione e apprendimento distribuito.
- **[Rischi di sicurezza nell'Hot-Swap di SRA](./sra_security_risks_hotswap_ja.md)** — Vettori di minaccia inclusi Pickle Exploit, Backdoor Injection, attacchi DoS e strategie di mitigazione.
- **[Divergenza di rappresentazione e routing gerarchico](./sra_representation_divergence_ja.md)** — Approccio del Tronco Condiviso e soluzioni Super Router.
- **[Confronto del routing rigido per il router gerarchico SRA](./sra_hierarchical_hard_routing_ja.md)** — Esperimenti comparativi di Soft / STE / Gumbel-Softmax.
