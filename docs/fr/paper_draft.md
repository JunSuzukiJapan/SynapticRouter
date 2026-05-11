# Routage Synaptique Bio-inspiré : Surmonter l'Oubli Catastrophique via des Voies Modulaires Dynamiques

**Jun Suzuki**, Independent Researcher

## Résumé (Abstract)
Le cerveau humain peut apprendre et exécuter des tâches fondamentalement différentes (comme marcher, parler et calculer) sans interférence mutuelle. Cela est dû au fait que les circuits neuronaux (synapses) du cerveau sont routés dynamiquement en fonction de la tâche, maintenant une "localisation fonctionnelle" spatialement isolée. En revanche, lorsque les Réseaux de Neurones Artificiels (ANNs) apprennent de multiples tâches au sein d'un seul réseau monolithique, ils souffrent d'"Oubli Catastrophique", où les mémoires passées sont détruites.

Dans cet article, nous proposons l'"Architecture de Routage Synaptique (SRA)", un modèle d'apprentissage continu inspiré des mécanismes biologiques de formation dynamique des synapses et d'isolation spatiale. La SRA se compose d'un "Routeur (Router)" monocouche extrêmement simple et de multiples micro-modules indépendants (Synapses). À travers nos expériences, nous démontrons que la SRA peut identifier de manière autonome la nature d'une tâche à partir de l'entrée — sans qu'on lui fournisse un ID de tâche externe pendant l'inférence — et **apprendre simultanément le routage (sélection des voies) et les représentations des tâches de manière totalement de bout en bout (End-to-End).** Nous montrons que, sans gel artificiel des poids ni algorithmes évolutifs complexes, une localisation fonctionnelle autonome émerge au sein du modèle, évitant complètement l'oubli catastrophique.

## 1. Introduction (Introduction)
Dans le domaine du deep learning, l'"Apprentissage Continu" est l'une des plus grandes barrières pour réaliser l'Intelligence Artificielle Générale (AGI). Les réseaux monolithiques, comme les modèles Transformer massifs actuels, oublient inévitablement les connaissances apprises précédemment lorsqu'ils sont affinés sur de nouvelles données.

Pour résoudre ce problème, notre recherche se concentre sur la "Localisation Fonctionnelle" du cerveau. Tout comme les zones du langage et motrices du cerveau utilisent des circuits physiquement distincts, la SRA est conçue comme une architecture capable d'allumer/éteindre (brancher/débrancher) dynamiquement de minuscules réseaux indépendants (synapses) via un mécanisme de routage dynamique.

## 2. Travaux Connexes et Nouveauté de la SRA
Une approche plus "structurelle et modulaire" s'apparentant à la SRA est **PathNet (2017)** par Google DeepMind, qui utilise des algorithmes génétiques pour découvrir des "chemins" et gèle les poids.

### L'Avantage Écrasant de la SRA (Apprentissage Simultané)
La nouveauté fondamentale de la SRA réside dans sa capacité à **"apprendre simultanément la découverte des voies (routage) et les représentations des modules de manière différentiable, de bout en bout".**
1. **Routage Autonome (Task-Agnostic):** Le routeur linéaire unique de la SRA détermine de manière autonome le domaine basé sur la similarité cosinus des caractéristiques d'entrée.
2. **Élimination du Gel des Poids et des Algorithmes Évolutifs:** La SRA utilise uniquement la Rétropropagation standard.
3. **Émergence de la Localisation Fonctionnelle Dynamique:** L'isolation spatiale émerge naturellement via une activation clairsemée (sparse).

## 3. Architecture (Neuro-inspired Design)
La SRA imite la formation des synapses du cerveau biologique. Le Routeur calcule la similarité cosinus entre les caractéristiques d'entrée et le vecteur d'incorporation (embedding) de chaque Synapse (des micro-modules de Multi-Head Attention et MLP).

## 4, 5, 6, 7. Expériences
Nous avons mené des expériences en Raisonnement Algorithmique, Modélisation de Langage Inter-domaines, Traduction Automatique Multilingue, et Apprentissage par Renforcement hors ligne (Decision Transformer). Dans chaque cas, **le routage simultané a conduit à la ségrégation modulaire autonome, similaire à la localisation fonctionnelle du cerveau**, sans aucune supervision d'ID de tâche.

## 8. Conclusion (Conclusion)
À travers la SRA, nous avons démontré le potentiel d'un changement de paradigme, passant des réseaux neuronaux statiques traditionnels à un "réseau modulaire doté d'une isolation spatiale biologique et d'un routage dynamique." C'est une étape cruciale vers une AGI évolutive.

## Références (References)
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
