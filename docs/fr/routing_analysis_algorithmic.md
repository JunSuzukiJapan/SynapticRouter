# Apprentissage Multi-tâches et Analyse de Routage

## Vue d'ensemble
Nous avons vérifié si SRA peut allouer de manière autonome des experts (synapses) selon la nature des tâches (`copy`, `reverse`, `paren`, `addmod`).

## Résultats d'Entraînement
Après 10 000 étapes, nous avons obtenu des résultats parfaitement précis (~100% de précision), démontrant que SRA prévient l'oubli catastrophique.

## Analyse de Routage (Similarité Cosinus)
1. **Groupe d'Opération de Séquence** : `COPY` et `REVERSE` (0.969 de similarité).
2. **Groupe Calcul/Logique** : `PAREN` et `ADDMOD` (0.858 de similarité).

## Conclusion
Le routeur SRA **discerne de manière autonome la nature des tâches**, partageant des synapses pour des tâches similaires et utilisant des synapses différentes pour des tâches distinctes.
