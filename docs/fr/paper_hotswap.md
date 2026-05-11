# LLM Remplaçable à Chaud (Hotswappable LLM) : Composition de Modules Zero-Shot et Suppression Chirurgicale de Connaissances via l'Architecture de Routage Synaptique

**Jun Suzuki**, Independent Researcher

## Résumé (Abstract)
Les humains peuvent apprendre de nouvelles compétences (ex. faire du vélo) sans détruire les connaissances existantes, grâce à des circuits neuronaux indépendants. Cependant, les Grands Modèles de Langage (LLMs) ont une structure monolithique qui rend l'ajout indépendant ou la suppression de mémoires spécifiques extrêmement difficile.

Dans cet article, nous utilisons le mécanisme de localisation fonctionnelle de l'Architecture de Routage Synaptique (SRA) pour proposer une méthode d'insertion et de retrait dynamique de circuits neuronaux — le **Hot-Swap**. C'est une opération où des synapses spécialisées peuvent être **transplantées chirurgicalement (Plug-In) dans le modèle sans réentraînement**, et **des mémoires spécifiques peuvent être déconnectées en toute sécurité (Unplug)**. Nous avons atteint le **Zero Forgetting (Zéro Oubli)** grâce à un mécanisme de masquage matériel (hard-mask) inspiré du pré-filtrage des bases de données vectorielles. Nous avons également résolu le "problème du trou noir" des vecteurs nuls, réalisant un cycle de vie complet de l'IA modulaire.

## 1. Introduction (Introduction)
Les LLMs monolithiques souffrent d'Oubli Catastrophique, de coûts d'entraînement gonflés et d'une impossibilité de suppression de connaissances (Machine Unlearning). 
La SRA permet l'innovation opérationnelle du **Hot-Swap** : Plug-In (Transplantation), Unplug (Suppression), et le Zéro Oubli garanti mathématiquement.

## 2. Contexte : Architecture SRA
La SRA imite l'isolation spatiale du cerveau avec un Routeur et des Synapses Minuscules. Une condition absolue pour le Hot-Swap est le **Tronc Partagé (Shared Trunk)**, où toutes les synapses partagent le même modèle de base pré-entraîné pour éviter la Divergence de Représentation.

## 3. Hot-Swap : Plug-In (Transplantation de Modules)
Nous transplantons le module spécialisé simplement via des opérations de copie de tenseurs PyTorch dans les emplacements vides du modèle de base.

## 4. Zéro Oubli : Mécanisme de Masquage Matériel (Hard-Mask)
Pour éviter que le routeur ne confonde les synapses, nous utilisons le **Pré-filtrage (Pre-filtering)**. Nous masquons physiquement les voies inutiles avec `-inf` lors de l'inférence.

## 5. Hot-Swap : Unplug (Suppression / Inactivation)
La suppression de connaissances est réalisée soit par **Déconnexion Physique** (`pop_synapses`), soit par **Purge par Inactivation** (`clear_synapses` - mise à zéro du tenseur sans changer sa taille).

## 6. Le Piège de la Similarité Cosinus : Le Problème du Trou Noir
La mise à zéro d'une synapse crée un vecteur nul avec un score de similarité de $0.0$. Si les scores normaux sont négatifs, le routeur choisira la synapse vide (le trou noir). J'ai résolu cela en bloquant complètement les synapses purgées avec un masque $-\infty$.

## 7, 8, 9. Conclusion
Le cycle de vie complet (Apprendre → Transplanter → Opérer → Supprimer → Réutiliser) permet aux modèles d'évoluer de boîtes noires monolithiques vers des systèmes modulaires vitaux où l'intelligence peut être physiquement insérée et retirée en toute sécurité.

## Références
- Suzuki, J. (2026). [Neuro-inspired Synaptic Routing: Overcoming Catastrophic Forgetting via Dynamic Modular Pathways. *Technical Report*.](https://github.com/JunSuzukiJapan/SynapticRouter/blob/main/docs/paper_draft_en.md)
