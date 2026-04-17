# Documentation du Modèle de Détection de Pollution de l'Eau (GRL)

Ce document décrit l'architecture, le prétraitement et le pipeline d'entraînement/évaluation du modèle d'IA développé pour détecter des anomalies physiques et chimiques à la surface des rivières, tout en ignorant les différences environnementales propres à chaque caméra.

## 1. Contexte et Problématique
Le projet vise à détecter 4 grandes classes de pollution de l'eau :
- **0 : Propre** (Eau naturelle)
- **1 : Coloration** (Anomalie chimique souvent orangée)
- **2 : Limon** (Sédimentation ou boue marron)
- **3 : Mousse** (Surfactants, rejets blancs)

Le principal défi (le **Domaine**) réside dans le fait que chaque caméra surveille une rivière différente (ex: l'Aire, la Ziplo, l'Avril). Historiquement, les modèles *overfittent* (surapprennent) sur le décor (l'herbe, le pont, l'éclairage) au lieu de regarder l'eau. Si le modèle est entraîné uniquement sur la rivière A, il échouera silencieusement sur la rivière B.

## 2. Prétraitement (Channel Splitting & Ratios)
Pour forcer le modèle à ignorer la "couleur naturelle" de l'eau (qui varie selon la météo et la rivière) et à se concentrer sur les anomalies : **on n'utilise plus d'images RGB classiques**.

Nous appliquons une classe PyTorch `PollutionFilters` (définie dans `test_pollution_filters.py`) qui extrait 3 canaux mathématiques distincts, concaténés en un seul tenseur `[3, H, W]` :
* **Canal 1 (Détection du Limon)** : Canal Rouge (R) + histogramming local adaptatif (CLAHE) pour faire ressortir les contrastes de boue fine.
* **Canal 2 (Coloration Chimique)** : Calcul d'un Pseudo-NDTI (Normalized Difference Turbidity Index) : `(G - R) / (G + R + epsilon)`. Ce ratio exalte immédiatement les anomalies orangées ou toxiques.
* **Canal 3 (Mousse/Irisation)** : Combinaison de la Luminance pure (seuil sur les blancs intenses) et de filtres de Sobel (contours brutaux typiques de l'écume et de la mousse).

## 3. Architecture du Réseau (Gradient Reversal Layer)
Pour lutter contre l'overfitting du décor, nous employons une technique d'**Unsupervised Domain Adaptation** basée sur un GRL (Gradient Reversal Layer).

Le réseau (`WaterPollutionGRL`) se compose de 3 modules :
1. **Feature Extractor** : Un backbone `EfficientNet-V2` (version small) pré-entraîné, tronqué avant sa couche de classification (`features` + `avgpool`). Il extrait un vecteur de dimension 1280 fortement optimisé.
2. **Class Predictor (Tête de Classement)** : Une tête Dense qui tente de prédire la classe de pollution réelle.
3. **Domain Predictor (Tête de Domaine)** : Une tête Dense qui tente de prédire **de quelle rivière provient l'image**.
   * *Magie du GRL* : Entre le Feature Extractor et cette tête de Domaine se trouve une couche mathématique d'inversion. Lors de la rétropropagation (backpropagation), cette couche **multiplie les gradients par -Alpha**.
   * Au lieu d'aider le Feature Extractor à détecter la rivière, cela force le Feature Extractor à "désapprendre" tout ce qui distingue les rivières entre elles. À la fin, l'extracteur devient aveugle à l'environnement et se concentre purement sur la texture (la pollution).

## 4. Organisation du Répertoire Produit
Toutes vos expérimentations sont désormais sauvegardées intelligemment de manière hiérarchique :
* `models/` : Contient vos poids PyTorch.
* `evaluation_results/` : Contient les graphiques d'analyse post-entraînement.

**Arborescence de Sauvegarde :**
```text
[Type_de_Modele] / [Data_Preprocess] / [Dataset_Entrainé] / Fichiers...
Exemple : multiclass / no_mask / train_aire_ziplo / best_grl_model.pth
```
- Le suffixe `train_...` vous certifie sur quelles données (rivières) les poids ont convergé (utile pour prouver la non-contamination lors d'une validation externe).

## 5. Scripts du Pipeline
Tous ces outils ont été déplacés dans le répertoire isolé `/pipeline/` :

1. **`test_pollution_filters.py`** : Visualiseur de debug. Lancer ce script ouvre une interface graphique permettant de voir concrètement l'effet des filtres mathématiques (C1, C2, C3) sur une image brute.
2. **`preprocess_pipeline.py`** : Générateur de dataset. Prend le dataset initial et découpe informatiquement les carrés 224x224 (avec ou sans l'application d'un polygone de masque noir).
3. **`train_grl.py`** : Lance l'entraînement. 
   * Prenez soin d'utiliser `--train_rivers` (ex: `Ziplo`) et `--val_rivers` pour définir le comportement d'évaluation (Split ciblé vs split 80/20 aléatoire).
4. **`evaluate_grl.py`** : Moteur d'évaluation scientifique. Accepte les modèles `.pth` construits.
   * Il génère une **Matrice de Confusion** classique.
   * Il calcule **l'Input Saliency Map**, quantifiant en % absolu l'importance mathématique requise par le réseau sur les Canaux 1, 2, et 3 sur le Dataset fourni.
   * Il produit des **Heatmaps Grad-CAM**, projetant la zone de focus spatial du réseau (*où* le réseau a regardé) superposée sur l'image d'origine.
