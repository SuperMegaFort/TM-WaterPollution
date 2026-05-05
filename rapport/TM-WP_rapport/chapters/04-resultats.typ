= Déploiement Logiciel et Persistance des Données

== Objectif du Jalon
Le modèle IA devait être encapsulé dans un outil utilisable par des non-informaticiens (biologistes, gardes-chasse), compatible sur différents systèmes d'exploitation (Windows/Mac/Linux), et capable de sauvegarder les résultats de manière pérenne.

== Ce que nous avons essayé : Sauvegarde dans un fichier CSV et base de données
Initialement, les prédictions (scores de l'IA et labels humains validés via l'interface UI) étaient uniquement sauvegardées dans un fichier `labels.csv` à côté des images. 

== Problème rencontré
Si l'utilisateur déplaçait une image ou perdait le fichier CSV, tout le travail d'inférence et de validation humaine était perdu. De plus, des problèmes de verrouillage de fichiers sous Windows (Permission Denied) faisaient crasher le backend Python lors de la lecture/écriture simultanée.

== Ce que nous avons choisi : Tagging EXIF Multi-plateforme et Architecture Micro-services
Nous avons pris deux décisions architecturales majeures :
1.  *Tagging EXIF Universel :* Nous avons abandonné la dépendance exclusive au CSV. Les scores de l'IA et les validations humaines sont désormais gravés directement dans "l'ADN" de l'image (métadonnées EXIF). Pour assurer la compatibilité inter-systèmes, nous encodons ces données en `UTF-8` pour Mac/Linux (ImageDescription) et en `UTF-16LE` pour les tags XP spécifiques à l'explorateur Windows.
2.  *Encapsulation via PyInstaller :* Au lieu d'exiger l'installation de Python et PyTorch par l'utilisateur final, nous avons utilisé PyInstaller couplé à Flask et PyWebView. Le logiciel se comporte comme une application native (avec un frontend Web interactif), gérant l'inférence lourde en arrière-plan via des Threads (ThreadPoolExecutor) pour ne pas figer l'interface.