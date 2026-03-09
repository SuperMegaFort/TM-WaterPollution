import os
import pandas as pd

def consolider_datasets(dossier_images, fichier_csv_ml, fichier_csv_ziplo, fichier_sortie):
    print("1. Chargement des CSV existants...")
    
    dictionnaire_classes = {}
    
    # Charger dataset_ml_final.csv (s'il existe)
    if os.path.exists(fichier_csv_ml):
        df_ml = pd.read_csv(fichier_csv_ml)
        for _, row in df_ml.iterrows():
            dictionnaire_classes[row['Nom_Image']] = row['Classe']
        print(f"   -> {len(df_ml)} images chargées depuis {fichier_csv_ml}")
    else:
        print(f"   -> Attention: {fichier_csv_ml} introuvable.")
        
    # Charger dataset_ziplo.csv (s'il existe)
    if os.path.exists(fichier_csv_ziplo):
        df_ziplo = pd.read_csv(fichier_csv_ziplo)
        for _, row in df_ziplo.iterrows():
            nom_img = row['Nom_Image']
            classe_ziplo = row['Classe']
            
            # Si l'image est bizarrement dans les DEUX fichiers, 
            # on garde par sécurité la classe la plus élevée (la pollution prime sur le 0)
            if nom_img in dictionnaire_classes:
                dictionnaire_classes[nom_img] = max(dictionnaire_classes[nom_img], classe_ziplo)
            else:
                dictionnaire_classes[nom_img] = classe_ziplo
        print(f"   -> {len(df_ziplo)} images chargées depuis {fichier_csv_ziplo}")
    else:
        print(f"   -> Attention: {fichier_csv_ziplo} introuvable.")
        
    print(f"\n2. Parcours du dossier d'images : {dossier_images}")
    donnees_finales = []
    compteur_connues = 0
    compteur_nouvelles = 0
    
    for nom_fichier in os.listdir(dossier_images):
        if nom_fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
            if nom_fichier in dictionnaire_classes:
                # L'image a déjà une classe grâce à l'un des deux CSV
                donnees_finales.append({
                    "Nom_Image": nom_fichier,
                    "Classe": dictionnaire_classes[nom_fichier]
                })
                compteur_connues += 1
            else:
                # L'image n'est dans AUCUN des CSV -> On lui met la classe 0 !
                donnees_finales.append({
                    "Nom_Image": nom_fichier,
                    "Classe": 0
                })
                compteur_nouvelles += 1

    print("\n3. Création du fichier CSV final consolidé...")
    df_final = pd.DataFrame(donnees_finales)
    
    if not df_final.empty:
        df_final.to_csv(fichier_sortie, index=False)
        print(f"Succès ! {compteur_connues} images ont gardé leur classe d'origine.")
        print(f"Ajout de {compteur_nouvelles} images restantes avec la classe 0.")
        print(f"Total : {len(df_final)} images prêtes pour l'entraînement !")
        print(f"Fichier généré : {fichier_sortie}")
    else:
        print("Erreur : Aucune image trouvée dans le dossier.")

# --- PARAMÈTRES (à vérifier/adapter) ---
dossier_des_images = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/data"

# Mets ici les chemins exacts vers les CSV que tu as générés précédemment
fichier_ml = "dataset_ml_final.csv"  
fichier_ziplo = "dataset_ziplo.csv"  

# Le nom de ton fichier ultime pour le Machine Learning
fichier_csv_final = "dataset_complet.csv"

# --- Lancement ---
consolider_datasets(dossier_des_images, fichier_ml, fichier_ziplo, fichier_csv_final)