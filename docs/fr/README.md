[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) est une nouvelle architecture de réseau de neurones dynamique, clairsemée et modulaire inspirée par le cerveau biologique (synapses).
Au lieu d'un Transformer massif et statique, SRA achemine dynamiquement les entrées vers des "synapses" (petits modules) appropriées pour un apprentissage plus efficace et une intelligence structurelle.

## 🎯 Motivation

Face à la croissance des modèles d'IA, les réseaux monolithiques rencontrent des problèmes tels que "l'augmentation des ressources de calcul" et "l'oubli catastrophique lors de l'apprentissage multi-tâches".
SRA résout ces problèmes par une **approche clairsemée : "appeler et combiner dynamiquement uniquement les modules nécessaires (synapses) en fonction de l'entrée."** Cela permet d'apprendre plusieurs tâches dans le même réseau sans interférence.

## 🧠 Vue d'ensemble de l'Architecture

1. **Synapse (Module Synaptique)**
   - Petites unités de calcul indépendantes (ex. Transformers miniatures ou MLPs).
2. **Routeur**
   - Sélectionne dynamiquement les `Top-k` meilleures synapses selon l'entrée.
3. **Espace Synaptique (Synapse Space)**
   - Les synapses s'organisent selon leur "similarité fonctionnelle".
4. **Règle d'Apprentissage Locale**
   - Combine la rétropropagation standard avec des règles locales à 3 facteurs.

## 📁 Structure du Répertoire

- `src/` : Implémentations principales du modèle SRA et scripts.
- `docs/` : Rapports et décisions d'architecture.
- `data/` : Jeux de données (Code, Mathématiques, Texte).
- `tests/` : Tests des différents composants.

## 🚀 Utilisation

```bash
pip install torch
python src/sra_experiment.py --task reverse --steps 2000
```

## 🧪 Expériences et Analyse

- [Apprentissage Multi-tâches et Analyse de Routage](./routing_analysis_algorithmic.md)
- [Analyse de Routage en Modélisation de Langage Multi-domaine](./routing_analysis_language.md)

## 🤝 Contribution et Licence

Licence : [MIT License](../../LICENSE).
