#set page(paper: "a4", margin: (x: 2cm, y: 2.5cm))
#set text(font: "New Computer Modern", size: 11pt, lang: "fr")
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1.")

#show heading: it => block(
  spacing: 1.5em,
  [
    #if it.level == 1 {
      text(weight: "bold", size: 1.2em, fill: rgb("#005bbb"))[#it]
    } else {
      text(weight: "bold", size: 1.1em, fill: rgb("#333333"))[#it]
    }
  ]
)

#align(center)[
  #text(size: 22pt, weight: "bold", fill: rgb("#002366"))[Travail de master TM]
  
  #v(0.5em)
  #text(size: 14pt)[*Rapport sur GRL et filtre RGB*]
  #v(2em)
]

= Constitution du *Ground Truth* (> 2000 images)
J'ai labelisé environ 2000 images car celle venant des fichier excel comprenaient des erreurs de labelisation. j'ai créer un petit script en streamlit pour annoter facilement les images.

Certaines images étaient unitilisables donc je les ai supprimées

quatre classes : *Propre, Coloration Chimique, Limon, et Mousse*. On laisse de coté l'irisation pour le moment

= RGB et Filtres

J'ai discuté avec un ami qui a fini sont master, et lui dans sont cas, le fite de filtrer les image en trois R G B lui a permis d'avoir un trés bon préprocess des images.

Le problème d'utiliser les image brutes RGB c'est que le réseaux de neurones vont overfitter sur les couleurs naturelle de l'eau ou sur la background


== Les Filtres qui ont échoué
Au début, j'ai tenté d'utiliser l'espace de couleur standard `HSV` (Teinte, Saturation) couplé à des filtres de netteté basiques CLAHE. mais visuellement ça ne donnait rien. Une simple correction de teinte ne permettait pas de distinguer formellement des sédiments subtils (Limon) des simples reflets du ciel gris. 

== La Solution : Mathématiques Spatiales (Channel Splitting)

Pour forcer le modèle à chercher la pollution réelle en ignorant la couleur "naturelle" de l'eau, l'approche retenue consiste à remplacer les 3 canaux de couleur classiques (R, G, B) par 3 cartes physiques:

+ *Canal 1 : Limon et Sédiments*
  - _Méthode_ : Isolation du canal Rouge (R). Dans le spectre visible, l'eau claire absorbe très rapidement les ondes rouges, tandis que les sédiments et la terre en suspension les reflètent énomrément.
  - _Amélioration (CLAHE)_ : Application de l'algorithme "Contrast Limited Adaptive Histogram Equalization" pour maximiser le contraste de la turbidité.

+ *Canal 2 : Coloration Chimique (Ratio Satellitaire NDTI)*
  - _Méthode_ : Calcul de la différence normalisée via la formule $(G - R) / (G + R + epsilon)$.
  - _Propriété d'annulation_ : Les rejets chimiques modifient les longueurs d'onde Vertes et Rouges. En divisant par la somme des deux, ce ratio se normalise entre $[-1, 1]$ et annule intégralement l'impact du niveau d'éclairage ambiant (nuages, ombres des arbres).

+ *Canal 3 : Mousse et Écume (Hautes Fréquences Géométriques)*
  - _Méthode_ : La mousse possède une luminance extrême (blancheur)
  - _Traitements croisés_ : On applique un seuillage pour surpprimer les pixels avec une intensité faible et on ajoute un filtre de sobel (filtre de contours) pour accentué les bords et supprimé les ombres/reflets qui sont flous 

= Gradient reversal layer
Le problème de nos image est que le background permet au modèle d'apprendre la type de polution (surapprentissage).
Pour y parvenir, j'ai implémenté le  *Gradient Reversal Layer (GRL)*.
J'ai ajouté au backbone du *EfficientNet-V2-S* deux têtes simultanées :
- Une tête tente de deviner la classe de pollution.
- *Une tête tente de deviner la rivière.*
Lors de la correction des poids par l'optimiseur, la couche mathématique *GRL* intercepte le signal de la rivière et le multiplie par une valeur négative $(-lambda)$. Ainsi, au lieu d'aider le Feature Extractor à mémoriser le lieu, le GRL l'oblige mathématiquement à *désapprendre* et à oublier toutes les spécificités du background

= Configuration du Modèle et Hyperparamètres
Configuration ML déployée :
- *Modèle (Feature Extractor)* : `EfficientNet-V2-S` (State-of-the-Art, vecteur dense de 1280 dimensions).
- *Fonctions d'Activation* : Utilisation de `ReLU` dans les têtes de classification, avec du `Dropout` à 50% pour prévenir le surapprentissage.
- *Normalisation* : Standardisation(`mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`). Permet de normaliser la quantité de lumière de chaque canal RGB, dans la cas du chanel splitting and ratio, j'ai modifié l'entrée du en un triplet mathématique (Limon, NDTI, Sobel) donc on met tout à 0.5.
- *Data Augmentation* : Transformations géométriques appliquées en vol à chaque époque (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15°)`, et `ColorJitter`).
- *Stratégie de Croisée* : Validation croisée`StratifiedKFold`. 

= Résultats Comparatifs : Test Zero-Shot sur l'Aire
Les 4 architectures ci-dessous ont été entraînées *exclusivement sur l'Arve et la Ziplo*, et testées purement sur la rivière *Aire* .

== 1. Modèle Binaire (Propre vs Pollué) - *SANS Masque*
- *Accuracy Globale* : 67%
- *Saliency (Focus du réseau)* : Canal 1 (Limon) 29.38% | *Canal 2 (Ratio NDTI) 53.05%* | Canal 3 (Mousse/Iris) 17.57%
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #table(
      columns: 4,
      fill: (_, row) => if row == 0 { luma(230) } else { white },
      [*Classe*], [*Précision*], [*Rappel*], [*F1-Score*],
      [Propre (0)], [0.66], [0.71], [0.68],
      [Pollué (1)], [0.69], [0.64], [0.66],
      [Moyenne], [0.67], [0.67], [0.67]
    )
  ],
  image("evaluation_results/binary/no_mask/train_arve_ziplo/confusion_matrix.png", width: 90%)
)

== 2. Modèle Binaire - *AVEC Masque (Filtrage des Berges)*
- *Accuracy Globale* : 65%
- *Saliency* : Canal 1 29.61% | *Canal 2 53.36%* | Canal 3 17.03%
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #table(
      columns: 4,
      fill: (_, row) => if row == 0 { luma(230) } else { white },
      [*Classe*], [*Précision*], [*Rappel*], [*F1-Score*],
      [Propre (0)], [0.67], [0.83], [0.74],
      [Pollué (1)], [0.56], [0.35], [0.43],
      [Moyenne], [0.62], [0.59], [0.59]
    )
  ],
  image("evaluation_results/binary/with_mask/train_arve_ziplo/confusion_matrix.png", width: 90%)
)

== 3. Modèle Multiclasse - *SANS Masque*
- *Accuracy Globale* : 34% 
- *Saliency* : Canal 1 29.21% | *Canal 2 53.29%* | Canal 3 17.50%
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #table(
      columns: 4,
      fill: (_, row) => if row == 0 { luma(230) } else { white },
      [*Classe*], [*Précision*], [*Rappel*], [*F1-Score*],
      [Propre (0)], [1.00], [0.09], [0.17],
      [Coloration (1)], [0.00], [0.00], [0.00],
      [Limon (2)], [0.31], [1.00], [0.47]
    )
  ],
  image("evaluation_results/multiclass/no_mask/train_arve_ziplo/confusion_matrix.png", width: 90%)
)

== 4. Modèle Multiclasse - *AVEC Masque*
- *Accuracy Globale* : 56% .
- *Saliency* : Canal 1 29.45% | *Canal 2 53.92%* | Canal 3 16.63%
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #table(
      columns: 4,
      fill: (_, row) => if row == 0 { luma(230) } else { white },
      [*Classe*], [*Précision*], [*Rappel*], [*F1-Score*],
      [Propre (0)], [1.00], [0.45], [0.62],
      [Coloration (1)], [0.25], [0.50], [0.33],
      [Limon (2)], [0.38], [0.83], [0.52]
    )
  ],
  image("evaluation_results/multiclass/with_mask/train_arve_ziplo/confusion_matrix.png", width: 90%)
)

#v(1em)
#align(center)[
  #text(weight: "bold", size: 12pt)[Preuve de Focus Intégré (Grad-CAM)] \
  #image("evaluation_results/multiclass/with_mask/train_arve_ziplo/grad_cam_samples.png", width: 95%)
]

#v(2em)
#line(length: 100%, stroke: 0.5pt + luma(200))


= Rapport 2
Implémentation de DANN Domain-Adversarial Neural Network. 

== Résultats Comparatifs : Classifieur de Base (Baseline GRL) testé sur Avril
Nous avons mené des expériences exhaustives en entraînant exclusivement sur *Ziplo et Aire*, pour tester la généralisation sur *Avril*.

=== 1. Modèle de Base - SANS Masque / SANS GRL
- *Accuracy Globale* : 67%
- *Saliency* : Canal 1 32.2% | *Canal 2 48.6%* | Canal 3 19.1%
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #table(
      columns: 4,
      fill: (_, row) => if row == 0 { luma(230) } else { white },
      [*Classe*], [*Précision*], [*Rappel*], [*F1-Score*],
      [Propre], [0.57], [0.65], [0.61],
      [Pollué], [0.75], [0.68], [0.71],
      [Moyenne], [0.66], [0.66], [0.66]
    )
  ],
  image("grl/no_mask/no_grl/train_ziplo_aire/eval_avril/confusion_matrix.png", width: 90%)
)

=== 2. Modèle de Base - AVEC Masque / SANS GRL
- *Accuracy Globale* : 61%
- *Saliency* : Canal 1 32.5% | *Canal 2 49.1%* | Canal 3 18.5%
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #table(
      columns: 4,
      fill: (_, row) => if row == 0 { luma(230) } else { white },
      [*Classe*], [*Précision*], [*Rappel*], [*F1-Score*],
      [Propre], [0.50], [0.81], [0.62],
      [Pollué], [0.79], [0.47], [0.59],
      [Moyenne], [0.65], [0.64], [0.61]
    )
  ],
  image("grl/with_mask/no_grl/train_ziplo_aire/eval_avril/confusion_matrix.png", width: 90%)
)

=== 3. Modèle de Base - SANS Masque / AVEC GRL
- *Accuracy Globale* : 68%
- *Saliency* : Canal 1 32.2% | *Canal 2 48.3%* | Canal 3 19.6%
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #table(
      columns: 4,
      fill: (_, row) => if row == 0 { luma(230) } else { white },
      [*Classe*], [*Précision*], [*Rappel*], [*F1-Score*],
      [Propre], [0.57], [0.81], [0.67],
      [Pollué], [0.83], [0.60], [0.70],
      [Moyenne], [0.70], [0.70], [0.68]
    )
  ],
  image("grl/no_mask/with_grl/train_ziplo_aire/eval_avril/confusion_matrix.png", width: 90%)
)

=== 4. Modèle de Base - AVEC Masque / AVEC GRL
- *Accuracy Globale* : 67%
- *Saliency* : Canal 1 34.1% | *Canal 2 48.3%* | Canal 3 17.6%
#grid(
  columns: (1fr, 1fr),
  gutter: 10pt,
  [
    #table(
      columns: 4,
      fill: (_, row) => if row == 0 { luma(230) } else { white },
      [*Classe*], [*Précision*], [*Rappel*], [*F1-Score*],
      [Propre], [0.56], [0.77], [0.65],
      [Pollué], [0.80], [0.60], [0.69],
      [Moyenne], [0.68], [0.68], [0.67]
    )
  ],
  image("grl/with_mask/with_grl/train_ziplo_aire/eval_avril/confusion_matrix.png", width: 90%)
)

#pagebreak()

#v(1em)
#align(center)[
  #text(weight: "bold", size: 12pt)[Exemple Grad-CAM (SANS Masque / SANS GRL)] \
  #image("grl/no_mask/no_grl/train_ziplo_aire/eval_avril/grad_cam_samples.png", width: 95%)
]
#v(2em)
#line(length: 100%, stroke: 0.5pt + luma(200))

#v(1em)
#align(center)[
  #text(weight: "bold", size: 12pt)[Exemple Grad-CAM (AVEC Masque / SANS GRL)] \
  #image("grl/with_mask/no_grl/train_ziplo_aire/eval_avril/grad_cam_samples.png", width: 95%)
]
#v(2em)
#line(length: 100%, stroke: 0.5pt + luma(200))

#pagebreak()

#v(1em)
#align(center)[
  #text(weight: "bold", size: 12pt)[Exemple Grad-CAM (SANS Masque / AVEC GRL)] \
  #image("grl/no_mask/with_grl/train_ziplo_aire/eval_avril/grad_cam_samples.png", width: 95%)
]
#v(2em)
#line(length: 100%, stroke: 0.5pt + luma(200))

#v(1em)
#align(center)[
  #text(weight: "bold", size: 12pt)[Exemple Grad-CAM (AVEC Masque / AVEC GRL)] \
  #image("grl/with_mask/with_grl/train_ziplo_aire/eval_avril/grad_cam_samples.png", width: 95%)
]
#v(2em)
#line(length: 100%, stroke: 0.5pt + luma(200))



= Architecture du Modèle et Pipeline d'Apprentissage



== 1. Constitution des Triplets (Stratégie de _Hard Mining_)
Dataloader: fourni les 3 images à la place d'une seul
- *Ancre (Anchor)* : Une image $A$ sélectionnée dans le batch (ex. Eau Propre, Rivière $X$).
- *Positif (Positive)* : Une image $P$ de la *même classe* que $A$, mais extraite obligatoirement d'un *autre domaine* (Rivière $Y$). Cela force le modèle à chercher les caractéristiques de l'eau plutôt que celles des berges.
- *Négatif (Negative)* : Une image $N$ de la *classe opposée*, idéalement du *même domaine* que $A$, pour que la seule différence mathématique exploitable par le réseau soit la présence de pollution.

== Extracteur de Caractéristiques (_Feature Extractor_)
- *Modèle de base* : `EfficientNet-V2-S`.
- *Modification* : La tête de classification d'origine est retirée. Seules les couches convolutionnelles (`.features`) et le pooling global (`.avgpool`) sont conservés.
- *Sortie* : Un vecteur de caractéristiques brutes de $1280$ dimensions.

== Projection dans l'Espace Latent (_Embedder_)
Le vecteur brut de $1280$ dimensions est compressé et organisé spatialement via un Perceptron Multicouche (MLP) :
- *Architecture* : `Linear(1280, 512)` $->$ `BatchNorm1d` $->$ `ReLU` $->$ `Dropout(0.3)` $->$ `Linear(512, 128)`.
- *Normalisation L2* : Les vecteurs finaux de $128$ dimensions subissent une normalisation euclidienne (`p=2`). Cette étape projette tous les embeddings sur une hypersphère de rayon 1, ce qui est crucial pour stabiliser la _Triplet Loss_ et éviter une dispersion infinie des points.

== Censure du Domaine (_Gradient Reversal Layer_)
Pour garantir que l'Espace Latent est indépendant de la rivière d'origine (_Domain-Invariant_), une tête antagoniste est branchée *directement sur l'embedding final* :
- *Le GRL* : Intercepte le gradient lors de la rétropropagation et le multiplie par une valeur $-alpha$.
- *Dynamique de $alpha$* : La force de l'inversion passe progressivement de $0$ à $1$ au fil des époques selon la formule $2 \/ (1 + exp(-10 \cdot p)) - 1$, évitant de détruire les poids du réseau dès les premiers batchs.
- *Classifieur de domaine* : Un MLP (`Linear` $->$ `BatchNorm` $->$ `ReLU` $->$ `Dropout` $->$ `Linear`) tente de prédire la rivière d'origine à partir des 128 dimensions.

== Fonctions de Perte et Équilibre des Gradients
L'optimisation repose sur une tension mathématique entre deux fonctions de perte :
- *Triplet Margin Loss ($L_t$)* : Rapproche $A$ et $P$, et éloigne $A$ et $N$ avec une marge stricte (`margin = 0.2`).
- *Cross-Entropy Loss ($L_d$)* : Pénalise la reconnaissance de la rivière par la tête GRL.
- *Pondération* : La perte totale est calculée via $L = L_t + (0.05 \cdot L_d)$. Le coefficient de $0.05$ (trouvé empiriquement) empêche le GRL de dominer la Triplet Loss et d'entraîner un _Mode Collapse_.
- *Optimiseur* : `AdamW` avec un _Weight Decay_ de $10^{-4}$ pour limiter le surapprentissage, couplé à un _Scheduler_ `ReduceLROnPlateau`.
- *Hard mining*: Cross-Domain exigé

== Validation Zéro-Shot et Métrique Proxy
évaluation sur le jeu de validation (une rivière jamais vue à l'entraînement) s'effectue par calcul de distances :
- *Calcul* : Mesure de la distance euclidienne (_Pairwise Distance_) entre l'Ancre et le Positif (censée tendre vers $0$), et entre l'Ancre et le Négatif (censée être supérieure à la marge).
- *Métrique (PR-AUC)* : Les distances sont évaluées via l'aire sous la courbe Précision-Rappel (_Average Precision_). voir différence avec le ROC-AUC

#pagebreak()
= Rapport 3 : Étude Comparative des Scopes et Impact du GRL

Cette section compile les résultats issus des entraînements sur les trois types de données (`manual_crop`, `manual_crop_tensor`, et `no_mask`) avec une comparaison directe entre les splits aléatoires (apprentissage global) et les splits géographiques (généralisation sur une rivière inconnue : l'Aire).

== Tableau Récapitulatif (Meilleur Kappa score)

#table(
  columns: (2fr, 1fr, 1.5fr, 1fr),
  fill: (_, row) => if row == 0 { luma(230) } else { white },
  [*Scope*], [*GRL*], [*Split Type*], [*Kappa*],
  [No Mask], [Sans], [Aléatoire (All)], [0.7342],
  [No Mask], [Sans], [Géo (Ziplo/Aire)], [0.3815],
  [No Mask], [Avec], [Aléatoire (All)], [0.7174],
  [No Mask], [Avec], [Géo (Ziplo/Aire)], [0.3047],
  [Manual Crop], [Sans], [Aléatoire (All)], [0.6546],
  [Manual Crop], [Sans], [Géo (Ziplo/Aire)], [0.1354],
  [Manual Crop Tensor], [Sans], [Aléatoire (All)], [0.5955],
  [Manual Crop Tensor], [Sans], [Géo (Ziplo/Aire)], [0.1812],
)

== Graphique de Généralisation (X/Y PlotChart)

Le graphique suivant compare la performance sur tout le dataset (axe X) par rapport à la généralisation sur une rivière inconnue (axe Y). La ligne pointillée représente la généralisation théorique parfaite ($X = Y$).

#v(1em)
#align(center)[
  #block(width: 300pt, height: 260pt, {
    let size = 220pt
    let padding = 40pt
    
    // Grille de fond
    for i in (0, 0.2, 0.4, 0.6, 0.8, 1.0) {
      place(dx: padding, dy: (1-i)*size + padding, line(length: size, stroke: 0.2pt + luma(200)))
      place(dx: i*size + padding, dy: padding, line(start: (0pt,0pt), end: (0pt, size), stroke: 0.2pt + luma(200)))
    }
    
    // Axes
    place(dx: padding, dy: size + padding, line(length: size + 10pt, stroke: 1pt))
    place(dx: padding, dy: padding - 10pt, line(start: (0pt, 10pt), end: (0pt, size + 10pt), stroke: 1pt))
    
    // Diagonale (Baseline de généralisation parfaite)
    place(dx: padding, dy: padding, line(start: (0pt, size), end: (size, 0pt), stroke: (dash: "dashed", paint: gray)))
    
    // Labels Axes
    place(dx: size/2 + padding, dy: size + padding + 20pt, text(size: 9pt, weight: "bold")[Performance Globale (Split All)])
    place(dx: 15pt, dy: size/2 + padding, rotate(-90deg, text(size: 9pt, weight: "bold")[Performance Zéro-Shot (Aire)]))
    
    // Graduation
    for i in (0, 0.5, 1.0) {
      place(dx: i*size + padding - 5pt, dy: size + padding + 5pt, text(size: 7pt)[#i])
      place(dx: padding - 15pt, dy: (1-i)*size + padding - 5pt, text(size: 7pt)[#i])
    }

    // Points de données (X=All, Y=Géo)
    let points = (
      (0.73, 0.38, "No Mask", blue),
      (0.71, 0.30, "No Mask (GRL)", rgb("#00aa00")),
      (0.65, 0.14, "Manual", red),
      (0.60, 0.18, "Manual Tensor", orange),
    )
    
    for (x, y, name, color) in points {
      let px = x * size
      let py = (1 - y) * size
      place(dx: px + padding, dy: py + padding, {
        circle(radius: 4pt, fill: color, stroke: white + 0.5pt)
        place(dx: 6pt, dy: -6pt, text(size: 8pt, weight: "bold", fill: color)[#name])
      })
    }
  })
]

#v(1em)
#align(center)[
  _Plus un point est proche de la diagonale, meilleure est sa généralisation géographique._
]

== Analyse des Observations
- *L'effet "No Mask"* : Contrairement aux attentes initiales, le modèle `no_mask` (en bleu) est celui qui s'approche le plus de la diagonale. Il conserve une performance de 0.38 sur l'Aire, suggérant que le "bruit" contextuel (arbres, berges) aide paradoxalement à la robustesse.
- *Le Gap de Domaine* : Le modèle `manual_crop` (en rouge) est le plus éloigné de la diagonale ($Y \approx 0.14$). Cela indique que le détourage strict force le modèle à apprendre des textures très spécifiques aux rivières d'entraînement, qui ne se retrouvent pas sur l'Aire.
- *Potentiel du GRL* : L'implémentation future avec le **gel du backbone** (Freezing) déjà codé dans `train_grl.py` vise à remonter le point vert (`No Mask (GRL)`) vers le haut pour qu'il dépasse la version sans GRL en termes de généralisation pure.
- *Scope Tensor* : Le mode "Super-Tensor" (`manual_crop_tensor`) montre une résilience légèrement supérieure au détourage RGB simple sur le split géographique (0.18 vs 0.13), validant l'intérêt des canaux physiques (NDTI + Limon).

== Analyse GRL (Modèles 1-6)

#v(1em)
#align(center)[
  #if os.exists("comparison_grl.png") [
    #image("comparison_grl.png", width: 90%)
    #text(size: 8pt, style: "italic")[Figure : Performance Kappa des modèles GRL 1 à 6]
  ] else [
    #rect(width: 80%, height: 100pt, fill: luma(240))[Image GRL manquante]
  ]
]

== Analyse Triplet (Modèles 1-6)

#v(1em)
#align(center)[
  #if os.exists("comparison_triplet.png") [
    #image("comparison_triplet.png", width: 90%)
    #text(size: 8pt, style: "italic")[Figure : Performance ROC-AUC des modèles Triplet 1 à 6]
  ] else [
    #rect(width: 80%, height: 100pt, fill: luma(240))[Image Triplet manquante]
  ]
]

#pagebreak()

= Source



== Architecture et Adaptation de Domaine (GRL)
+ *Unsupervised Domain Adaptation (GRL)* : Ganin, Y., & Lempitsky, V. (2015). _"Unsupervised Domain Adaptation by Backpropagation"_. Proceedings of the 32nd International Conference on Machine Learning (ICML).
+ *Extraction de Features (Backbone)* : Tan, M., & Le, Q. (2021). _"EfficientNetV2: Smaller Models and Faster Training"_. International Conference on Machine Learning (ICML).
+ *Implémentation GRL PyTorch* : "Gradient Reversal Layer in PyTorch". https://github.com/tadeephuy/GradientReversal

== Traitement d'Image et Filtres Spatiaux (Channel Splitting)
+ *Indice de Coloration / Turbidité (NDTI)* : Lacaux, J. P., et al. (2007). _"Classification of ponds from high-spatial resolution remote sensing"_. (Source pionnière justifiant le ratio des bandes Verte et Rouge pour isoler optiquement les composants aqueux en suspension).
+ *Index de Différence Normalisé (NDTI/NGRDI)* : Index Database (IDB). https://www.indexdatabase.de/db/i-single.php?id=390
+ *Égalisation d'Histogramme Adaptative (CLAHE)* : Zuiderveld, K. (1994). _"Contrast Limited Adaptive Histogram Equalization"_. Graphics Gems IV.
+ *Algorithme CLAHE* : "Histogram Equalization: CLAHE Algorithm". Medium. https://medium.com/imagecraft/histogram-equalization-clahe-algorithm-8841d402fc76
+ *Opérateur de Contours Spatiaux (Sobel)* : Sobel, I., & Feldman, G. (1968). _"A 3x3 Isotropic Gradient Operator for Image Processing"_. (Utilisé dans l'isolation géométrique de la mousse hydrophobe).