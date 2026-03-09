import os
import shutil
import datetime

# 1. Définir les dossiers (à remplacer par tes propres chemins)
# Attention sous Windows, garde bien le 'r' devant les guillemets
dossier_source = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/Photos/Sélection de photos événement pollution/Avril"
dossier_destination = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/data"

def renommer_images_par_date(source, destination):
    # Créer le dossier de destination s'il n'existe pas
    if not os.path.exists(destination):
        os.makedirs(destination)
        print(f"Dossier créé : {destination}")

    compteur = 0

    # Parcourir tous les fichiers du dossier source
    for nom_fichier in os.listdir(source):
        # Vérifier qu'il s'agit bien d'une image PNG (ou autre)
        if nom_fichier.lower().endswith('.jpg'):
            chemin_source = os.path.join(source, nom_fichier)

            # Récupérer la date de modification/création du fichier
            timestamp = os.path.getmtime(chemin_source)
            date_creation = datetime.datetime.fromtimestamp(timestamp)
            nom_sans_ext, extension = os.path.splitext(nom_fichier)
            
            # Formater la date en AnnéeMoisJour_HeureMinuteSeconde
            date_formatee = date_creation.strftime("%d%m%Y_%H%M%S")

            nom_riviere = "Avril" # Récupérer le nom du dossier source (ex: "2021.03.22")
            # Créer le nouveau nom
            nouveau_nom = f"{date_formatee}_{nom_sans_ext}_{nom_riviere}{extension.lower()}"
            chemin_destination = os.path.join(destination, nouveau_nom)

            # Copier le fichier avec son nouveau nom (copy2 conserve les métadonnées)
            shutil.copy2(chemin_source, chemin_destination)
            print(f"Succès : {nom_fichier} -> {nouveau_nom}")
            compteur += 1

    print(f"\nOpération terminée. {compteur} images ont été renommées et copiées.")

# Lancer la fonction
renommer_images_par_date(dossier_source, dossier_destination)