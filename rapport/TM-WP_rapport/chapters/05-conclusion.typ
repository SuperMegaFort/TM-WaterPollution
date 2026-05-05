= Conclusion et Perspectives Futures

Ce projet a répondu avec succès au défi d'adapter un modèle de Deep Learning en une application concrète, robuste et utilisable pour la surveillance environnementale. En partant de l'architecture PyTorch de base, nous avons construit un pipeline complet gérant la donnée depuis le fichier brut jusqu'à l'exportation de métadonnées enrichies.

Le principal défi de ce projet n'a pas seulement été l'inférence, mais bien l'ingénierie logicielle requise pour rendre ce modèle exploitable : compatibilité inter-systèmes d'exploitation, traitement du signal post-inférence pour la cohérence temporelle, et ergonomie de l'interface utilisateur. L'approche choisie — privilégier des solutions mathématiques élégantes et peu coûteuses (comme le filtre médian) plutôt qu'une sur-complexification algorithmique (BiLSTM) — a prouvé son efficacité en termes de performances.

== Perspectives Futures
À l'avenir, cet outil de labellisation intégré permettra de constituer rapidement de nouveaux jeux de données vérifiés par les experts, facilitant ainsi le réentraînement continu du modèle via Domain Adaptation (GRL) sur de nouveaux cours d'eau. Une adaptation pour le *Edge Computing*, où un modèle quantifié serait embarqué directement sur les caméras in situ, constituerait l'étape technologique suivante.
