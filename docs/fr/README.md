[English](../../README.md) | [日本語](../../README_JP.md)

# Synaptic Routing Architecture (SRA)

## 🎮 Démos interactives (Carnets Jupyter)

Nous avons préparé des blocs-notes Jupyter dans lesquels vous pouvez découvrir de manière interactive « l'utilisation du cerveau spécifique à une tâche » et la « robustesse » de SRA directement dans votre navigateur.Vous pouvez les exécuter en quelques secondes sur Google Colab, alors essayez-les !

- [01 SRA Quickstart](../../notebooks/01_sra_quickstart_fr.ipynb)
- [02 Learning and Routing Demo](../../notebooks/02_learning_and_routing_demo_fr.ipynb)
- [03 Multitask Routing Demo](../../notebooks/03_multitask_routing_demo_fr.ipynb)
- [04 Decision Transformer Routing Demo](../../notebooks/04_decision_transformer_routing_demo_fr.ipynb)
- [05 Lesion Experiment Demo](../../notebooks/05_lesion_experiment_demo_fr.ipynb)


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



---

### 🔌 6. Expérience de remplacement dynamique des synapses et limites d'apprentissage du routeur
**Fichier:** [`06_hotswap_experiment_demo_fr.ipynb`](./06_hotswap_experiment_demo_fr.ipynb)

Démontre le véritable pouvoir de SRA : « l'ajout et le retrait dynamiques de synapses comme plugins (Hot-Swap) ».
Nous effectuons une expérience où une synapse spécifique à l'espagnol est fusionnée dans un modèle de traduction français/allemand en cours d'exécution.
Dans ce notebook, vous apprendrez l'**importance cruciale du partage et du gel de l'espace de connaissances du modèle de base (couches d'intégration/d'attention, etc.)** pour établir un remplacement à chaud. En même temps, vous ferez face à la **plus grande barrière de SRA (le problème du gradient évanescent)** : le routage dur standard (Top-k) ne peut pas apprendre (différencier) rétroactivement le routage des synapses ajoutées. Cette limitation sert de préfiguration critique pour la section suivante « Gumbel-Softmax (Super Router) ».

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/06_hotswap_experiment_demo_fr.ipynb)

---

### 👑 7. Intégration de Modèles via Super Routeur et Gumbel-Softmax
**Fichier:** [`07_super_router_gumbel_demo_fr.ipynb`](./07_super_router_gumbel_demo_fr.ipynb)

Nous construisons un "Super Routeur" qui regroupe plusieurs modèles spécialisés (un modèle FR/DE et un modèle ES) et achemine dynamiquement le traitement en fonction de l'entrée.
Cela démontre le problème du "Routage Paresseux" du routage doux simple (Soft Routing) et montre comment l'utilisation de Gumbel-Softmax permet d'obtenir un **Routage Dur parfait**, réduisant ainsi de 100% les calculs inutiles du modèle.

[![Ouvrir dans Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/07_super_router_gumbel_demo_fr.ipynb)

---

### 📖 8. Démo SRA LLM (Shakespeare)
**File:** [`08_sra_llm_demo_shakespeare_fr.ipynb`](../../notebooks/08_sra_llm_demo_shakespeare_fr.ipynb)

Il s'agit d'un didacticiel qui utilise des données Shakespeare à petite échelle pour former SRA en tant que modèle génératif spécifique au décodeur (LLM). Après l'apprentissage, une carte thermique est utilisée pour visualiser par quelle synapse chaque jeton du texte généré est transmis.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/08_sra_llm_demo_shakespeare_fr.ipynb)

---

### 🌐 9. Démo LLM multi-domaines SRA (Code, Mathématiques, Texte)
**File:** [`09_sra_llm_demo_multidomain_fr.ipynb`](../../notebooks/09_sra_llm_demo_multidomain_fr.ipynb)

Découvrez la spécialité de SRA « apprentissage simultané de plusieurs domaines (code, mathématiques, texte) » dans un LLM à petite échelle. Vous pouvez vérifier comment le modèle divise (spécialise) automatiquement les synapses en fonction des données.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/09_sra_llm_demo_multidomain_fr.ipynb)

---

### 💻 10. Plugin pratique Hot-Swap (Zero-Shot)
**File:** [`10_hotswap_plugins_demo_fr.ipynb`](../../notebooks/10_hotswap_plugins_demo_fr.ipynb)

Nous démontrerons un flux de travail dans lequel plusieurs équipes de développement apprennent indépendamment des plug-ins pour le « code » et les « mathématiques » et les « fusionnent physiquement (échange à chaud) » dans le modèle de base de l'environnement de production après coup. Il a été prouvé que même après la fusion, les pertes de tous les domaines sont exactement les mêmes que lors d'un apprentissage indépendant (Zéro Oubli).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/10_hotswap_plugins_demo_fr.ipynb)

---

### 🗑️ 11. Suppression synaptique dynamique
**File:** [`11_synapse_deletion_demo_fr.ipynb`](../../notebooks/11_synapse_deletion_demo_fr.ipynb)

Nous démontrons la fonction de la SRA, « suppression des synapses ». Vous pouvez expérimenter à la fois la « suppression des plug-ins (pop_synapses) », qui supprime physiquement les synapses ajoutées plus tard depuis la fin, et la « purge d'un domaine spécifique (clear_synapses) », qui efface et désactive en toute sécurité les synapses qui ne sont pas partagées.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunSuzukiJapan/SynapticRouter/blob/main/notebooks/11_synapse_deletion_demo_fr.ipynb)



## 🤝 Contribution et Licence

Licence : [MIT License](../../LICENSE).
