# Hotswappable LLM : Composition de Modules Zero-Shot et Suppression Chirurgicale de Connaissances via l'Architecture de Routage Synaptique

**Jun Suzuki**, Chercheur Indépendant

## Abstract
Les grands modèles de langage (LLM) stockent toutes les connaissances de manière dense dans un espace de paramètres monolithique unique, rendant extrêmement difficile l'ajout ou la suppression de connaissances spécifiques et imposant de sévères contraintes sur la flexibilité opérationnelle. Dans cet article, j'exploite la modularité de la Synaptic Routing Architecture (SRA) pour proposer et valider le **Hot-Swap** — une opération d'échange bidirectionnelle de modules pour les réseaux de neurones. Le Hot-Swap permet le **Plug-In** : insérer physiquement des synapses spécialisées entraînées indépendamment dans un modèle de base pré-entraîné **sans aucun réentraînement**, et l'**Unplug** : supprimer chirurgicalement les connaissances devenues inutiles. Les expériences démontrent qu'un mécanisme de masquage dur inspiré des techniques de pré-filtrage des bases de données vectorielles atteint le **Zero Forgetting**, où la perte de sortie du modèle de base correspond exactement à la décimale près avant et après l'insertion. De plus, je découvre et résous le « Problème du Trou Noir » des vecteurs nuls dans la similarité cosinus rencontré lors de l'opération Unplug, établissant le cycle de vie complet de l'IA modulaire : Entraîner → Plug-In → Unplug → Réutiliser.

## 1. Introduction

### 1.1 Limitations opérationnelles des modèles monolithiques
Depuis l'introduction de « Attention Is All You Need », l'architecture Transformer a établi une position dominante dans de nombreux domaines, y compris le traitement du langage naturel. Cependant, les LLM monolithiques avec des centaines de milliards de paramètres font face aux défis opérationnels critiques suivants :

1. **Oubli catastrophique (Catastrophic Forgetting)** : L'ajustement fin d'un modèle généraliste sur un domaine spécifique (par ex. réglementations internes, code spécialisé) détruit ou dégrade ses capacités générales d'origine.
2. **Escalade des coûts d'entraînement** : Chaque ajout de connaissances nécessite le réentraînement du modèle entier (ou d'adaptateurs tels que LoRA), rendant le développement entièrement parallèle par plusieurs équipes impraticable.
3. **Impossibilité de suppression de connaissances** : Le « Machine Unlearning » — oublier sélectivement des connaissances spécifiques — est extrêmement difficile dans les modèles monolithiques où les paramètres sont profondément entrelacés, et les tentatives de réentraînement détruisent souvent des capacités sans rapport.

### 1.2 Contributions
J'ai précédemment proposé la Synaptic Routing Architecture (SRA) [Suzuki, 2026], une architecture sparse composée de modules indépendants minuscules (synapses) et d'un routeur léger. Tandis que les travaux antérieurs ont démontré la capacité de séparation autonome des tâches du routeur, cet article se concentre sur les **innovations opérationnelles** permises par la modularité de SRA — le **Hot-Swap** bidirectionnel de modules (insertion et suppression) — rapportant trois contributions :

1. **Hot-Swap : Plug-In (Insertion)** : Implémentation et validation d'une méthode où le déploiement de modèles spécialisés entraînés indépendamment ne nécessite qu'une copie physique de tenseurs dans les emplacements vides du modèle de base.
2. **Hot-Swap : Unplug (Suppression)** : Conception de deux API de suppression — détachement physique (`pop_synapses`) et purge par remise à zéro (`clear_synapses`) — et la découverte et résolution du « Problème du Trou Noir » des vecteurs nuls dans la similarité cosinus.
3. **Preuve expérimentale du Zero Forgetting** : Démonstration qu'un mécanisme de masquage dur inspiré du pré-filtrage de bases de données vectorielles garantit que la perte de sortie du modèle de base reste identique à la décimale près avant et après l'insertion et la suppression.

## 2. Background: SRA Architecture

SRA (Synaptic Routing Architecture) est une architecture dynamique et sparse inspirée du cerveau biologique. Cette section décrit les composants essentiels à la compréhension du Hot-Swap (voir [Suzuki, 2026] pour les détails).

### 2.1 Router
Le routeur, cœur de SRA, est une **unique couche linéaire** sans aucun mécanisme d'Attention. Il calcule la similarité cosinus entre l'état caché d'entrée $h$ et le vecteur de caractéristiques (embedding) $e_i$ de chaque synapse, sélectionnant les Top-k synapses.

$$\text{logits}_i = \frac{h \cdot e_i}{\|h\| \cdot \|e_i\|} \cdot \alpha$$

où $\alpha$ est un facteur d'échelle.

### 2.2 Tiny Synapses
Chaque synapse est un module minuscule indépendant composé d'une petite couche Multi-Head Attention et d'un MLP. Seules les synapses sélectionnées par le routeur exécutent des calculs, atteignant une efficacité computationnelle élevée.

### 2.3 Tronc partagé (Shared Trunk)
Un prérequis critique pour le Hot-Swap est l'approche du **Tronc Partagé**. Toutes les synapses spécialisées sont dérivées du même modèle de base pré-entraîné (couches d'Embedding, couches d'Attention), entraînant indépendamment uniquement les composants synaptiques. Ceci empêche la divergence des représentations vectorielles internes (Representation Divergence) entre les modèles et permet la transplantation de synapses par copie physique.

## 3. Hot-Swap : Plug-In (Insertion de module)

La première opération du Hot-Swap est le **Plug-In** — insérer des modules spécialisés entraînés indépendamment dans le modèle de base. Cette section démontre que le processus d'insertion consiste entièrement en opérations de tenseurs PyTorch et est extrêmement simple.

### 3.1 Méthode

```python
# hotswap_model: Modèle de base de production (avec des emplacements vides ajoutés)
# plugin_math: LLM spécialisé en mathématiques entraîné indépendamment par l'équipe math

with torch.no_grad():
    for l in range(layers):
        target_block = hotswap_model.blocks[l]
        src_block = plugin_math.blocks[l]
        
        # Copier les vecteurs d'embedding du routeur
        target_block.router.synapse_emb.data[4:8] = src_block.router.synapse_emb.data
        
        # Copier les poids Expert (TinySynapse) (w1, w2)
        target_block.w1.data[4:8] = src_block.w1.data
        target_block.w2.data[4:8] = src_block.w2.data
```

Les tenseurs du modèle spécialisé entraîné sont simplement assignés directement à des indices spécifiques (emplacements vides) des tenseurs du modèle de base. Parce que SRA entraîne uniquement les synapses tout en gardant les connaissances partagées du modèle de base (couches d'Attention, etc.) complètement gelées via l'approche du Tronc Partagé, cette opération de copie physique est valide.

### 3.2 Permettre le développement parallèle indépendant
Cette approche permet à une « équipe Code » et une « équipe Math » d'entraîner indépendamment leurs synapses spécialisées sur la base du même modèle de base avec zéro interférence mutuelle. Après l'entraînement, le déploiement est complété simplement en copiant en mémoire les tenseurs de poids dans les emplacements vides du modèle de base de production.

## 4. Zero Forgetting : Masquage dur inspiré du pré-filtrage de bases de données vectorielles

### 4.1 Défi : Confusion du routeur
La simple copie physique de tenseurs risque que le routeur confonde anciennes et nouvelles synapses, altérant potentiellement la sortie du modèle de base.

### 4.2 Filtrage par métadonnées dans les bases de données vectorielles
Les bases de données vectorielles modernes telles que Pinecone et Weaviate utilisent le filtrage par métadonnées en parallèle de la recherche sémantique basée sur la similarité cosinus.

- **Post-filtrage** : Exclut les résultats non correspondants après la recherche Top-K. Sujet à l'« épuisement K-NN » où des résultats insuffisants subsistent après filtrage.
- **Pré-filtrage** : Restreint l'espace de recherche via des masques de métadonnées **avant** la recherche, effectuant le Top-K uniquement parmi les candidats qualifiés. Le bruit est complètement éliminé.

### 4.3 Masque dur de pré-exécution de SRA
Le routeur SRA est essentiellement un **moteur de recherche vectorielle en mémoire (Maximum Inner Product Search: MIPS)** calculant les produits scalaires entre les vecteurs d'entrée et les vecteurs d'embedding des synapses.

J'ai incorporé le pré-filtrage des bases de données vectorielles directement dans la passe avant du routeur. Au moment de l'inférence, un masque de métadonnées spécifiant l'« ensemble des synapses autorisées pour la tâche courante » est fourni au modèle.

```python
# Passe avant du routeur
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Pré-filtrage: Mettre à -infini les logits des synapses non autorisées
logits = logits.masked_fill(~allowed_mask, float('-inf'))

# Routage Top-K
vals, idx = torch.topk(logits, k, dim=-1)
```

Ce pré-filtrage par `masked_fill` garantit que le routeur sélectionne les experts uniquement parmi les synapses autorisées. Quel que soit le nombre de poids d'autres modèles coexistant, ils sont complètement ignorés dans le graphe de calcul, garantissant que **la perte du modèle de base correspond exactement à la décimale près avant et après la composition (interférence mathématiquement nulle)**.

### 4.4 Résultats expérimentaux
J'ai comparé la Validation Loss du modèle de base (modèle de langage 3 domaines Code/Math/Text) avant et après le Plug-In de synapses spécialisées entraînées indépendamment. La perte correspondait exactement à la décimale près, démontrant empiriquement le Zero Forgetting.

## 5. Hot-Swap : Unplug (Suppression de module)

La deuxième opération du Hot-Swap est l'**Unplug** — supprimer du modèle de base les modules qui ne sont plus nécessaires. Si les connaissances peuvent être « branchées », la capacité de les « débrancher » est tout aussi essentielle. Le Machine Unlearning dans les modèles monolithiques est extrêmement difficile en raison de l'enchevêtrement complexe des paramètres, mais la structure modulaire de SRA résout ce problème par des opérations physiques.

### 5.2 Approche 1 : Suppression physique (pop_synapses)
Lorsque les synapses ajoutées par Hot-Swap ne sont plus nécessaires, elles sont physiquement découpées depuis la fin du tenseur.

```python
def pop_synapses(self, num_drop: int):
    self.synapse_emb = nn.Parameter(self.synapse_emb.data[:-num_drop])
```

**Avantage** : L'utilisation VRAM est physiquement réduite, et le modèle peut être entièrement restauré à son état pré-ajout — comme la désinstallation d'un pilote OS, une partie physique du cerveau de l'IA peut être retirée.

### 5.3 Approche 2 : Purge par remise à zéro (clear_synapses)
Lors du débranchement de synapses à des indices intermédiaires plutôt qu'à la fin, la suppression physique décalerait tous les indices de synapses suivants, brisant le système de contrôle par masque de métadonnées. Au lieu de cela, le contenu des synapses est remis à zéro pour créer un « emplacement vide ».

```python
def clear_synapses(self, indices_to_clear: list[int]):
    for idx in indices_to_clear:
        self.synapse_emb.data[idx].zero_()
        self.w1.data[idx].zero_()
        self.w2.data[idx].zero_()
```

En invalidant uniquement le contenu des emplacements sans modifier la taille du tenseur, l'intégrité des indices est parfaitement préservée. Les emplacements vides peuvent être ultérieurement réutilisés en y inscrivant de nouvelles synapses via Hot-Swap.

## 6. The Cosine Similarity Trap : Le problème du trou noir du vecteur nul

### 6.1 Découverte
Lors de l'implémentation de la purge par remise à zéro pour l'opération Unplug, j'ai rencontré un bug critique où **la sortie s'est complètement effondrée**.

### 6.2 Analyse de la cause racine
Le routeur SRA effectue le routage en utilisant la similarité cosinus. Le vecteur d'embedding d'une synapse remise à zéro devient $\mathbf{0}$, qui reste $\mathbf{0}$ même après normalisation. La similarité cosinus entre tout vecteur d'entrée $h$ et le vecteur nul est $0.0$.

Le problème survient parce que la similarité cosinus varie sur $[-1.0, 1.0]$. Si la similarité cosinus d'une synapse valide est négative (par ex. $-0.5$), **la synapse vide ($0.0$) obtient un score mathématiquement plus élevé, amenant le routeur à sélectionner préférentiellement la synapse vide**.

$$\text{similarity}(h, \mathbf{0}) = 0.0 > -0.5 = \text{similarity}(h, e_{\text{valid}})$$

Les données sont « aspirées et disparaissent dans » ce qui devrait être un emplacement vide inexistant — un comportement semblable à un trou noir.

### 6.3 Solution : Blocage complet via masquage -∞
J'ai ajouté un traitement de masque pour détecter et exclure les synapses remises à zéro dans la passe avant du routeur.

```python
logits = torch.einsum("btd,nd->btn", h_norm, emb_norm) * self.scale

# Détecter les synapses remises à zéro
is_cleared = (full_emb == 0).all(dim=-1)
if is_cleared.any():
    logits = logits.masked_fill(is_cleared.view(1, 1, -1), float('-inf'))
```

Le masque $-\infty$ rend mathématiquement impossible la sélection de synapses vides, quel que soit le niveau des scores des autres synapses.

## 7. The Complete Lifecycle of Modular AI

Les mécanismes décrits ci-dessus permettent à SRA de réaliser le cycle de vie complet de l'IA modulaire :

```
Entraîner → Hot-Swap (Composer) → Servir
   ↓                                 ↓
Développement                    Purger (Supprimer)
parallèle indépendant               ↓
                              Réutilisation d'emplacement
                                     ↓
                              Nouveau Hot-Swap
```

1. **Entraîner** : Plusieurs équipes partagent un modèle de base et développent indépendamment leurs synapses spécialisées en parallèle.
2. **Composer** : Les tenseurs entraînés sont physiquement copiés dans le modèle de base de production pour le déploiement.
3. **Servir** : L'inférence fonctionne avec le Zero Forgetting garanti par le pré-filtrage par masque dur.
4. **Supprimer** : Les synapses inutiles sont physiquement retirées ou remises à zéro pour la purge.
5. **Réutiliser** : Les emplacements vides sont réutilisés en y insérant de nouvelles synapses spécialisées par Hot-Swap.

## 8. Discussion

### 8.1 Divergence de représentation (Representation Divergence)
Le **prérequis absolu** pour le Hot-Swap est que toutes les synapses spécialisées soient dérivées du même modèle de base pré-entraîné (Tronc Partagé). La transplantation de synapses entre des modèles entraînés de manière complètement indépendante provoque l'effondrement du routage en raison de la divergence des représentations vectorielles internes.

### 8.2 Super Routeur comme alternative
Pour assouplir la contrainte du Tronc Partagé, une approche a été validée où des modèles entiers indépendants sont encapsulés et orchestrés par un Super Routeur utilisant Gumbel-Softmax. Cette approche atteint un routage dur parfait $1.0$ vs $0.0$, permettant une commutation dynamique complète des ressources de calcul même entre des modèles d'architectures différentes.

### 8.3 Risques de sécurité
La capacité Hot-Swap introduit de nouveaux vecteurs de menaces de sécurité en raison de sa propriété de chargement dynamique de fichiers de poids depuis l'extérieur d'un système en cours d'exécution. Les risques principaux incluent : (1) l'exécution arbitraire de code via des exploits Pickle, (2) l'injection de poids malveillants (Backdoor Injection), (3) le détournement de routage via la falsification de clés de routeur, et (4) les attaques DoS via le swap thrashing. Des mesures d'atténuation telles que le format obligatoire `safetensors`, la signature cryptographique des synapses et la limitation du débit sont recommandées.

### 8.4 Limitations actuelles et travaux futurs
Cette recherche est au stade expérimental avec des modèles à petite échelle ($d_\text{model}=128$, $n_\text{layers}=4$). La validation sur des LLM de classe 10B reste un défi futur important. De plus, le problème de synchronisation du routeur — la nécessité potentielle d'un apprentissage d'adaptation des clés du routeur lors de l'ajout de synapses avec des capacités entièrement nouvelles — nécessite une investigation plus approfondie.

## 9. Conclusion

Dans cet article, j'ai proposé et validé des méthodes pour rendre les LLM « Hotswappable » (dynamiquement enfichables) en exploitant la modularité de SRA (Synaptic Routing Architecture). L'opération Plug-In du Hot-Swap complète le déploiement par la seule copie physique de tenseurs, tandis que l'opération Unplug établit deux approches de suppression : le détachement physique et la purge par remise à zéro. Grâce à un mécanisme de masquage dur inspiré du pré-filtrage de bases de données vectorielles, le Zero Forgetting est garanti mathématiquement. En découvrant et résolvant le « Problème du Trou Noir » des vecteurs nuls dans la similarité cosinus rencontré lors de l'Unplug, la réutilisation sûre des emplacements est atteinte.

À une époque où les modèles continuent de croître et de devenir plus opaques, l'approche « Hotswappable LLM » — permettant le contrôle physique par branchement et débranchement des composants d'intelligence — représente une direction extrêmement prometteuse pour la maintenabilité, la sécurité et l'efficacité opérationnelle des modèles.

## References

- Suzuki, J. (2026). [All You Need Is Router: Dynamic Sparse Modularity in Neural Networks. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS*.
- Shazeer, N. et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. *ICLR*.
- Jiang, A.Q. et al. (2024). Mixtral of Experts. *arXiv:2401.04088*.
- Jang, E. et al. (2017). Categorical Reparameterization with Gumbel-Softmax. *ICLR*.

## Appendix: Interactive Demos

Les processus complets de Hot-Swap et de suppression de synapses décrits dans cet article peuvent être expérimentés de manière interactive dans les notebooks Google Colab suivants.

- **Démo de composition de synapses Hot-Swap (Entraînement de base → Entraînement indépendant → Composition → Preuve du Zero Forgetting)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo.ipynb)
- **Démo de suppression de synapses (Suppression physique → Remise à zéro → Résolution du problème du trou noir)**<br>
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo.ipynb)

## Appendix: Related Technical Documents

- **[L'avenir de SRA : Hot-Swap dynamique et extensibilité](./sra_future_hotswap_ja.md)** — Discussion sur l'opération de synapses en mode cassette, la personnalisation et l'apprentissage distribué.
- **[Risques de sécurité dans le Hot-Swap SRA](./sra_security_risks_hotswap_ja.md)** — Vecteurs de menaces incluant Pickle Exploit, Backdoor Injection, attaques DoS et stratégies d'atténuation.
- **[Divergence de représentation et routage hiérarchique](./sra_representation_divergence_ja.md)** — Approche du Tronc Partagé et solutions Super Routeur.
- **[Comparaison du routage dur pour le routeur hiérarchique SRA](./sra_hierarchical_hard_routing_ja.md)** — Expériences comparatives de Soft / STE / Gumbel-Softmax.
