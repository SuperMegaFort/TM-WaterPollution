= Méthodologie : Ingénierie Logicielle et Architecture

== Objectif du Jalon
Créer une architecture logicielle modulaire, rapide et capable de s'exécuter sur n'importe quel système d'exploitation.

== 3.1 La persistance des données et le défi EXIF
*Ce que nous avons essayé :* Sauvegarder les validations de l'utilisateur uniquement dans un fichier `labels.csv` accompagnant le dossier d'images.
*Le problème rencontré :* Si les images sont déplacées par l'utilisateur, le lien avec le CSV est rompu. Nous avons donc basculé sur l'injection de métadonnées EXIF. Cependant, un bug critique est apparu lors des tests sous Windows : le modèle plantait car Windows verrouillait le fichier pendant la lecture `Pillow`, et les scores EXIF s'affichaient à `0.0` dans l'explorateur Windows.

*Ce que nous avons choisi :*
1. L'utilisation d'un bloc `with Image.open(...)` pour forcer la libération de la mémoire et déverrouiller le fichier sous Windows.
2. Une double implémentation d'écriture EXIF. Pour Linux/macOS, l'encodage classique UTF-8 et ASCII (`UserComment`). Pour Windows, nous avons injecté les données dans les balises spécifiques `XPComment` et `XPSubject`, en forçant l'encodage en `UTF-16LE`.
3. L'écriture sur le disque étant l'opération I/O la plus lente, nous l'avons encapsulée dans un `ThreadPoolExecutor` pour exécuter la sauvegarde en multithreading asynchrone.

== 3.2 L'Interface Utilisateur (UI) et l'Encapsulation
*Ce que nous avons essayé :* Exécuter le code via des lignes de commande ou un simple notebook Jupyter.
*Le problème rencontré :* Manque d'ergonomie pour l'inspection manuelle des données douteuses et nécessité d'installer un environnement Python lourd pour les utilisateurs finaux.

*Ce que nous avons choisi :* Une architecture Microservices Locale. Le backend est propulsé par *Flask*, exposant une API REST. Le frontend est une interface web (HTML/JS/CSS) reprenant les codes visuels d'un IDE technique, avec intégration de `Chart.js` pour la visualisation interactive de la confiance du modèle.
L'ensemble a été "packagé" avec *PyInstaller* et le wrapper *PyWebView*, permettant de livrer un exécutable final autonome (`.exe` ou `.app`) qui utilise le moteur web natif du système (WebKit sur Mac, EdgeHTML sur Windows, Qt sur Linux) sans nécessiter de terminal.
