# Analyse de Routage en Modélisation de Langage Multi-domaine (Code / Math / Texte)

## Vue d'ensemble
Nous avons mené une expérience de "modélisation de langage" avec SRA pour apprendre simultanément trois domaines (Code, Mathématiques, Texte) avec des vocabulaires différents.

## Évaluation de la Qualité d'Inférence
Après 1000 étapes, des prédictions précises ont été obtenues sans interférence.

## Analyse de la Spécialisation des Synapses
1. **Homogénéisation (Warmup)** : Au début, les tâches utilisaient les 16 synapses équitablement.
2. **Spécialisation** : Le routage s'est clairement séparé.
   - `Code` utilisait la **Synapse 8**.
   - `Math` utilisait les **Synapses 10 et 13**.
   - `Texte` utilisait les **Synapses 0 et 15**.

## Conclusion
Le modèle a découvert de manière autonome "l'utilisation de synapses propres à chaque tâche", empêchant l'oubli catastrophique.
