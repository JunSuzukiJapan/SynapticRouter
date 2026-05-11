# LLM Sostituibile a Caldo (Hotswappable LLM): Composizione di Moduli Zero-Shot e Rimozione Chirurgica della Conoscenza tramite Architettura di Routing Sinaptico

**Jun Suzuki**, Independent Researcher

## Riassunto (Abstract)
Gli esseri umani possono apprendere nuove abilità (es. andare in bicicletta) in modo postnatale e farle funzionare come circuiti indipendenti nel cervello senza distruggere le conoscenze esistenti. Tuttavia, gli attuali Grandi Modelli Linguistici (LLM) possiedono una struttura monolitica che memorizza densamente tutte le conoscenze, rendendo l'aggiunta indipendente di conoscenze o l'eliminazione di ricordi specifici estremamente difficile.

In questo articolo, sfruttiamo il meccanismo di localizzazione funzionale della Synaptic Routing Architecture (SRA) per proporre l'**Hot-Swap**, un metodo per l'inserimento e la rimozione dinamica di circuiti neurali. Sinapsi specializzate possono essere **trapiantate chirurgicamente (Plug-In) nel modello di produzione senza alcun riaddestramento** e, quando non servono più, **memorie specifiche possono essere disconnesse in sicurezza (Unplug)**. Utilizzando un meccanismo di hard-masking ispirato ai database vettoriali, abbiamo ottenuto lo **Zero Forgetting**. Abbiamo anche risolto il "problema del buco nero" dei vettori nulli nella similarità del coseno, realizzando il ciclo di vita completo dell'IA modulare.

## 1. Introduzione (Introduction)
Gli LLM monolitici soffrono di oblio catastrofico e impossibilità di Machine Unlearning. La SRA risolve questi problemi con l'innovazione operativa dell'Hot-Swap: Plug-In (Trapianto) e Unplug (Rimozione), garantendo matematicamente Zero Interferenze.

## 2. L'Architettura SRA
La SRA imita l'isolamento spaziale del cervello. Una condizione assoluta per far funzionare l'Hot-Swap è l'approccio **Shared Trunk (Tronco Condiviso)**. Tutte le sinapsi specializzate condividono lo stesso modello di base pre-addestrato per prevenire la Divergenza di Rappresentazione.

## 3. Hot-Swap: Plug-In (Trapianto di Moduli)
Trapiantiamo il modulo specializzato semplicemente copiando fisicamente i tensori PyTorch negli slot vuoti del modello base. Poiché il modello base è congelato, la copia fisica funziona perfettamente.

## 4. Zero Forgetting: Meccanismo Hard-Mask
Per evitare che il router confonda le sinapsi vecchie con quelle nuove, usiamo il **Pre-filtering**, bloccando fisicamente i percorsi non necessari con `-inf` durante l'inferenza, garantendo matematicamente lo Zero Forgetting.

## 5. Hot-Swap: Unplug (Rimozione / Inattivazione)
I moduli non necessari vengono rimossi dal modello base tramite:
1. **Disconnessione Fisica (`pop_synapses`)**: Tagliando dalla fine del tensore.
2. **Epurazione per Inattivazione (`clear_synapses`)**: Azzerando il contenuto della sinapsi senza modificare le dimensioni del tensore per mantenere la consistenza degli indici.

## 6. Il Problema del Buco Nero del Vettore Nullo
L'azzeramento di una sinapsi crea un vettore nullo con similarità $0.0$. Se i punteggi normali sono negativi, il router sceglierà la sinapsi vuota (il buco nero). L'ho risolto aggiungendo una maschera $-\infty$ per rilevare e bloccare i percorsi delle sinapsi azzerate.

## 7, 8, 9. Conclusione
Il ciclo di vita completo (Apprendere → Trapiantare → Operare → Rimuovere → Riutilizzare) della SRA consente ai modelli di evolvere da scatole nere monolitiche a sistemi modulari vitali per il controllo sicuro.

## Riferimenti (References)
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
