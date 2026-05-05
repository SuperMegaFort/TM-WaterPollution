= Stratégie d'Évaluation : Fiabilité et Traitement du Signal

== Objectif du Jalon
Fiabiliser les prédictions brutes du réseau de neurones face aux perturbations de l'environnement naturel (lumière, animaux, météo).

== La Constitution du Ground Truth et de la Base de Données
*Ce que nous avons essayé :* Au début du projet, nous avons tenté d'utiliser les labels existants fournis dans un fichier Excel.
*Le problème rencontré :* L'inspection des données a révélé de nombreuses erreurs de labellisation dans ce fichier source, rendant impossible l'entraînement d'un modèle d'intelligence artificielle fiable (phénomène de *garbage in, garbage out*). De plus, certaines images étaient tout simplement inutilisables (floues, corrompues, etc.).

*Ce que nous avons choisi :* Il a fallu reprendre le contrôle de la qualité des données en amont. Pour ce faire, j'ai développé un petit script via la bibliothèque *Streamlit* permettant d'annoter les images rapidement et visuellement. J'ai ainsi re-labellisé manuellement plus de 2000 images, en supprimant celles inexploitables. Les données ont été classées selon quatre catégories distinctes : *Propre, Coloration Chimique, Limon, et Mousse* (les irisations ont été temporairement écartées de l'étude pour cette phase).

== La cohérence temporelle des prédictions
*Ce que nous avons essayé :* Au départ, le modèle évaluait chaque image de manière totalement indépendante.
*Le problème rencontré :* L'analyse visuelle des courbes de confiance (ROC-AUC) montrait un signal très bruité. Un reflet de soleil soudain ou le passage d'un oiseau générait des "pics" de pollution isolés et aberrants sur une seule image, créant de fausses alertes temporelles.

*Ce que nous avons choisi :* Plutôt que de complexifier l'architecture IA (ex: ajouter un réseau récurrent BiLSTM qui aurait surchargé la mémoire VRAM), nous avons opté pour un lissage en post-traitement via un filtre médian (fenêtre de 11 images). Contrairement à une moyenne mobile qui aurait dilué l'erreur sur les images adjacentes et créé de fausses pentes douces lors des vraies pollutions, la médiane élimine instantanément les bruits de type "poivre et sel" tout en préservant les fronts raides (les changements brusques d'état) caractéristiques des véritables déversements de pollution. C'est l'application du Rasoir d'Ockham : une solution algorithmique simple, très peu coûteuse en RAM, et redoutablement efficace.

== Le filtrage des images de nuit et infrarouges
*Ce que nous avons essayé :* Analyser l'intégralité des dossiers d'images provenant des caméras.
*Le problème rencontré :* Les caméras capturent des images 24h/24. Celles de nuit, souvent sous-exposées ou capturées par des capteurs Infrarouges (IR) en niveaux de gris, faussaient totalement les prédictions du réseau de neurones qui avait été entraîné sur des images de jour.

*Ce que nous avons choisi :* L'implémentation d'un filtrage mathématique strict en amont de l'inférence. L'application calcule la luminosité moyenne de l'image, mais surtout la *variance entre les canaux Rouge, Vert et Bleu (RGB)*. Si la luminosité est trop faible (`< 40`) ou la variance de couleur quasi nulle (`< 3.0`), cela prouve mathématiquement que l'image est nocturne ou monochromatique (IR). Le backend supprime alors physiquement ces fichiers inutiles avant même de les passer au modèle, économisant ainsi l'espace disque et accélérant significativement l'inférence globale.


= Ingénierie des Caractéristiques (Feature Engineering) : Le Filtrage RGB

L'utilisation d'images brutes RGB posait un risque majeur de surapprentissage (overfitting) sur les couleurs naturelles de l'eau ou sur l'arrière-plan environnant. Suite à des discussions techniques et aux retours d'expérience d'anciens étudiants, une stratégie de *Channel Splitting* a été explorée pour isoler les composantes physiques de l'image.

== Les Filtres qui ont échoué
Initialement, nous avons tenté de convertir les images dans l'espace de couleur standard `HSV` (Teinte, Saturation, Valeur) couplé à des algorithmes d'égalisation d'histogramme (CLAHE) pour améliorer la netteté. Cependant, visuellement, les résultats n'étaient pas probants. Une simple correction de teinte ne permettait pas de distinguer formellement des sédiments subtils (Limon) de simples reflets optiques dus à un ciel gris.

== La Solution : Mathématiques Spatiales (Channel Splitting)
Pour forcer le réseau de neurones à concentrer son attention sur les marqueurs réels de pollution en ignorant la couleur "naturelle" de l'eau, nous avons remplacé les trois canaux d'entrée classiques (R, G, B) par trois "cartes physiques" distinctes :

1. *Canal 1 : Limon et Sédiments*
  - *Méthode* : Isolation stricte du canal Rouge (R) de l'image.
  - *Justification* : Dans le spectre visible, l'eau claire absorbe très rapidement les ondes rouges, tandis que la terre et les sédiments en suspension les reflètent énormément.
  - *Amélioration* : Application de l'algorithme "Contrast Limited Adaptive Histogram Equalization" (CLAHE) pour maximiser le contraste visuel de la turbidité.

2. *Canal 2 : Coloration Chimique (Ratio Satellitaire NDTI)*
  - *Méthode* : Calcul de la différence normalisée via la formule mathématique (Green - Red) / (Green + Red + Epsilon).
  - *Justification (Propriété d'annulation)* : Les rejets chimiques modifient la réflectance des longueurs d'onde Vertes et Rouges. En divisant par la somme des deux, ce ratio normalise la valeur entre $[-1, 1]$ et annule intégralement l'impact du niveau d'éclairage ambiant (ex: l'ombre d'un nuage ou d'un arbre ne modifiera pas le ratio de la couleur sous-jacente).

3. *Canal 3 : Mousse et Écume (Hautes Fréquences Géométriques)*
  - *Méthode* : Exploitation de la luminance extrême de la mousse.
  - *Traitements croisés* : Un seuillage est d'abord appliqué pour supprimer tous les pixels de faible intensité. Ensuite, un *filtre de Sobel* (opérateur de contours spatiaux) est appliqué pour accentuer les bords francs de la mousse et supprimer les ombres ou les reflets de l'eau qui, par nature, possèdent des bords flous.

Grâce à cette approche, le tenseur d'entrée du modèle EfficientNet n'est plus une simple photographie, mais un triplet mathématique hautement spécialisé (Limon, NDTI, Sobel) normalisé autour d'une moyenne de `0.5`, facilitant ainsi grandement la convergence lors de l'apprentissage.
