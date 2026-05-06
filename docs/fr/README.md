[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

Synaptic Routing Architecture (SRA) est une nouvelle architecture de réseau de neurones dynamique, clairsemée et modulaire inspirée par le cerveau biologique (synapses).
Au lieu d'un Transformer massif et statique, SRA achemine dynamiquement les entrées vers des "synapses" (petits modules) appropriées pour un apprentissage plus efficace et une intelligence structurelle.

## 🎯 Motivation

Face à la croissance des modèles d'IA, les réseaux monolithiques rencontrent des problèmes tels que "l'augmentation des ressources de calcul" et "l'oubli catastrophique lors de l'apprentissage multi-tâches".
SRA résout ces problèmes par une **approche clairsemée : "appeler et combiner dynamiquement uniquement les modules nécessaires (synapses) en fonction de l'entrée."** Cela permet d'apprendre plusieurs tâches dans le même réseau sans interférence.

## 💡 Idée de Base

Les modèles d'IA classiques (comme les Transformers) tentent de tout traiter à l'aide d'un seul "cerveau" géant. Cependant, avec cette approche, la charge de calcul devient beaucoup trop lourde chaque fois que le modèle est rendu plus intelligent ou plus grand. Par conséquent, SRA adopte un système où **de nombreux "petits cerveaux experts (que SRA appelle 'synapses')" sont préparés, et seuls les experts nécessaires sont appelés en fonction du problème en cours**.

La clé ici est le mécanisme qui décide "quel expert appeler". SRA possède un "routeur (guide)", qui sélectionne instantanément l'expert qui semble le plus capable en examinant les données d'entrée. Au fur et à mesure que chaque expert devient plus intelligent (apprend), ce routeur apprend simultanément "qui est le bon à choisir", évoluant pour être capable de faire des affectations optimales automatiquement.

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
- [Analyse de Routage en Traduction Multilingue (Ang / Fra / Jap) et Généralisation Zero-Shot](../dev/multilingual_translation_routing_analysis.md)
  - Un rapport fascinant montrant comment SRA attribue automatiquement différents modules de traduction en fonction de la structure grammaticale (SVO vs SOV). Plus surprenant encore, lorsqu'on lui demande de traduire une paire de langues non apprise, il utilise inconsciemment l'anglais comme "langue pivot" !
- [Séparation Complète de la Perception et de la Politique dans Decision Transformer (Apprentissage par Renforcement)](../dev/decision_transformer_routing_analysis.md)
  - Nous avons fait jouer SRA à un jeu. Il a découvert de lui-même une structure modulaire incroyable : il utilise exactement le même module de "vision" pour percevoir l'environnement dans toutes les tâches, mais bascule vers des modules de "cerveau" complètement différents selon qu'il doit trouver un trésor ou s'enfuir.
- [Vérification de la Traduction Multilingue Pratique avec SRA Encoder-Decoder](../dev/sra_seq2seq_translation_analysis.md)
  - Un rapport démontrant qu'en étendant SRA à une architecture Encoder-Decoder et en l'entraînant pendant 30 000 étapes sur un corpus réel (opus100), il peut traduire des expressions pratiques comme "Merci beaucoup." et "Good morning." avec un BLEU=1.0. L'introduction de Cross-Attention a permis de passer de Decoder-only (BLEU=0) à un BLEU moyen global de 0,27, et a atteint une précision presque pratique de BLEU=0,56 dans la direction FR→EN.


## 🤝 Contribution et Licence

Licence : [MIT License](../../LICENSE).
