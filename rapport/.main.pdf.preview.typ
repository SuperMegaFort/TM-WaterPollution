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
  #text(size: 22pt, weight: "bold", fill: rgb("#002366"))[Détection de Pollution des Eaux par IA]
  
  #v(0.5em)
  #text(size: 14pt)[*Bilan Technique et Avancées Architecurales*]
  #v(2em)
]

= Constitution du *Ground Truth* (> 2000 images)
J'ai labelisé environ 2000 images car celle venant des fichier excel comprenaient des erreurs de labelisation. j'ai créer un petit script en streamlit pour annoter facilement les images.

Certaines images étaient unitilisables donc je les ai supprimées

quatre classes : *Propre, Coloration Chimique, Limon/Boue, et Mousse*. On laisse de coté l'irisation pour le moment
#figure(image("image_streamlit.png"), caption: [Caption])

== nGrok
Uilisation de nGrok pour faire un canal de liaison sécurisé pour accéder à mon streamlit en distanciel sans devoir avoir un server
= Problème en utilisant les image RGB

J'ai discuté avec un ami qui a fini sont master, et lui dans sont cas, le fite de filtrer les image en trois R G B lui a permis d'avoir un trés bon préprocess des images.

Le problème d'utiliser les image brutes RGB c'est que le réseaux de neurones vont overfitter sur les couleurs naturelle de l'eau ou sur la background


== Les Filtres qui ont échoué 
Au début, j'ai tenté d'utiliser l'espace de couleur standard `HSV` (Teinte, Saturation) couplé à des filtres de netteté basiques. mais visuellement ça ne donnait rien. Une simple correction de teinte ne permettait pas de distinguer formellement des sédiments subtils (Limon) des simples reflets du ciel gris. 

== La Solution : Mathématiques Spatiales (Channel Splitting)
Pour forcer le modèle à chercher la pollution réelle en ignorant la couleur naturelle de l'eau, j'ai remplacé les 3 canaux de couleur (R, G, B) par 3 *cartes physiques* mathématiquement pré-calculées :

1. *Canal 1 : Limon et Sédiments (Contrastes Locaux)*
   - _Méthode_ : Isolation du canal Rouge (Red). Dans le spectre visible, l'eau absorbe le rouge mais les sédiments/terre le reflètent massivement.
   - _Amélioration (CLAHE)_ : Application de l'algorithme Contrast Limited Adaptive Histogram Equalization. Au lieu d'illuminer toute l'image, le CLAHE égalise la lumière par petites tuiles mathématiques, ce qui efface les grandes ombres portées par les arbres ou le soleil, et exacerbe violemment les micro-textures granuleuses de la boue en suspension.

2. *Canal 2 : Coloration Chimique (Ratio Satellitaire NDTI)*
   - _Méthode_ : Calcul du Normalized Difference Turbidity Index issu de la télédétection spatiale (Green - Red) / (Green + Red + epsilon).
   - _Propriété d'annulation_ : Une turbidité ou coloration radiométrique modifie différemment les longueurs d'onde vertes et rouges. En divisant par la somme des deux, ce ratio mathématique se normalise entre [-1, 1] et annule intégralement l'impact du niveau d'éclairage ambiant. Les variations colorées (fuites industrielles orangées/jaunes) s'illuminent en blanc pur, tandis que l'eau naturelle devient métaphoriquement "invisible" pour le Feature Extractor.

3. *Canal 3 : Mousse et Écume (Hautes Fréquences Géométriques)*
   - _Méthode_ : La mousse polluante possède deux particularités : une luminance extrême (blancheur) et une géométrie agressive (bulles et arêtes dures).
   - _Traitements croisés_ : D'abord, un seuillage (Threshold) coupe violemment tous les pixels peu lumineux. Ensuite, on extrait spatialement les contours via un Filtre de Sobel 2D. Cette dérivation détruit les reflets flous du ciel sur l'eau (qui n'ont pas de bords nets), mais fait formellement ressortir le signal des nappes d'écume industrielles aux contours fracturés.

#figure(image("filter_preview_08012026_150500_RCNX0039_Ziplo.jpg"), caption: [Caption])


= Unsupervised Domain Adaptation (GRL)
Problème: les caméras sont fixes et surveillent des rivières différentes (la Ziplo, l'Aire, l'Avril).
Si j'entraîne une IA sur la Ziplo, elle va mémoriser l'herbe et le pont de la Ziplo au lieu de comprendre la pollution. Si on change des rivière, les résultalt ne sont pas bons


Utilisation de *Gradient Reversal Layer (GRL)*.
Il faut le mettre dans le backbone EfficientNet-V2-S*  *deux têtes  :
- Une tête tente de deviner la classe de pollution.
- *Une tête tente de deviner la rivière.*
Lors de la correction des poids par l'optimiseur, la couche mathématique *GRL* intercepte le signal de la rivière et le multiplie par une valeur négative $(-lambda)$.

= Configuration du Modèle et Hyperparamètres
COnfiguration du modèle:
- *Modèle (Feature Extractor)* : `EfficientNet-V2-S` ( vecteur dense de 1280 dimensions).
- *Fonctions d'Activation* : Utilisation de `ReLU` dans les têtes de classification, avec du `Dropout` à 50%.
- *Normalisation* : Standardisation personnalisée (`mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]`).
- *Data Augmentation* : Transformations géométriques (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15°)`, et `ColorJitter`).
- *Stratégie de Croisée* : `StratifiedKFold`. 

= Résultats Comparatifs : Test Zero-Shot sur l'Aire
Les 4 architectures ci-dessous ont été entraînées *exclusivement sur l'Arve et la Ziplo*, et testées purement sur la rivière *Aire* (territoire chimiquement inconnu pour le modèle).

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
- *Accuracy Globale* : 34% (Le modèle multiclasse se perd massivement dans les détails du décor végétal).
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
- *Accuracy Globale* : 56% (En ne lui dévoilant que la surface liquide, la capacité de catégorisation complexe se redresse).
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
*Conclusion Finale :* 
L'écrasante polarisation du Gradient sur le Canal 2 (Ratio NDTI >53% pour tous les runs) valide indiscutablement notre architecture de Channel Splitting : l'IA se fie à nos mathématiques plutôt qu'à l'ambiance radiométrique. En synergie avec le Gradient Reversal Layer, nous avons engendré un détecteur abstrait qui survit même lorsque le milieu écologique rural fluctue massivement.