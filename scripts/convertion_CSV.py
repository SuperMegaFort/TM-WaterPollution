import os
import pandas as pd
from datetime import datetime

def generer_dataset_ml(dossier_images, fichier_excel, fichier_sortie):
    print("1. Chargement du tableau des étiquettes (labels)...")
    
    # Lire le fichier Excel sans en-tête (utilise pd.read_csv si tu utilises la version csv)
    df_brut = pd.read_excel(fichier_excel, header=None)
    
    # Extraire les heures et les données
    heures = df_brut.iloc[2, 3:51].astype(str).tolist()
    colonnes_a_garder = [0] + list(range(3, 51))
    
    df_donnees = df_brut.iloc[4:, colonnes_a_garder].copy()
    df_donnees.columns = ["Date"] + heures
    
    # Format long
    df_long = pd.melt(
        df_donnees, 
        id_vars=["Date"], 
        var_name="Heure", 
        value_name="Pollution"
    )
    
    # Nettoyage : enlever les cases vides
    df_long = df_long.dropna(subset=["Date", "Pollution"])
    
    # S'assurer du bon format de la date (YYYY-MM-DD) et nettoyer le texte
    df_long['Date'] = pd.to_datetime(df_long['Date']).dt.strftime('%Y-%m-%d')
    df_long['Pollution'] = df_long['Pollution'].astype(str).str.strip()
    
    # FILTRE TRES IMPORTANT : On ne garde que les lignes où la pollution est 1, 2, 3 ou 4
    # (cela va automatiquement ignorer les cases "Nuit", "Pas d'image", etc.)
    df_long = df_long[df_long['Pollution'].isin(['1', '2', '3', '4'])]

    # Créer un "dictionnaire" pour rechercher très vite : clé = (Date, Heure) -> valeur = Pollution
    mapping_labels = {}
    for _, row in df_long.iterrows():
        cle = (row['Date'], str(row['Heure']))
        mapping_labels[cle] = row['Pollution']

    print(f"2. Parcours du dossier d'images : {dossier_images}")
    donnees_entrainement = []
    images_non_trouvees = 0
    
    # Parcourir toutes les images de ton dossier 'data'
    for nom_fichier in os.listdir(dossier_images):
        if nom_fichier.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Le nom est structuré comme : 20210322_143000_...
            # On sépare le nom grâce aux tirets du bas '_'
            parties = nom_fichier.split('_')
            
            if len(parties) >= 2:
                date_str = parties[0]  # ex: 20210322
                heure_str = parties[1] # ex: 143000
                
                try:
                    # On convertit les éléments du nom de fichier au format du tableau Excel
                    date_formatee = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    heure_formatee = f"{heure_str[:2]}:{heure_str[2:4]}:{heure_str[4:6]}"
                    
                    cle = (date_formatee, heure_formatee)
                    
                    # On cherche si on a un niveau de pollution pour ce moment précis
                    if cle in mapping_labels:
                        pollution = mapping_labels[cle]
                        donnees_entrainement.append({
                            "Nom_Image": nom_fichier,
                            "Label_Pollution": pollution
                        })
                    else:
                        # L'image n'est pas dans le tableau, ou c'était marqué "Nuit"
                        images_non_trouvees += 1
                        
                except Exception as e:
                    pass

    # 3. Sauvegarder le résultat final
    print("3. Création du fichier d'entraînement...")
    df_final = pd.DataFrame(donnees_entrainement)
    
    if not df_final.empty:
        df_final.to_csv(fichier_sortie, index=False)
        print(f"\nSuccès ! {len(df_final)} images ont été associées à un niveau de pollution (1, 2, 3 ou 4).")
        print(f"Fichier généré : {fichier_sortie}")
    else:
        print("\nAttention : Aucune correspondance n'a été trouvée. Vérifie le format des noms d'images.")
        
    if images_non_trouvees > 0:
        print(f"Note : {images_non_trouvees} images ont été ignorées (car le tableau disait 'Nuit', était vide, etc.).")

# --- PARAMÈTRES (à modifier avec tes propres chemins) ---
dossier_des_images = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/data"
fichier_excel_source = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/Photos/Aire 2021/GraphiqueAirepiègePhotos_1.xlsx"
fichier_csv_final = "dataset_train.csv"

# --- Lancement ---
generer_dataset_ml(dossier_des_images, fichier_excel_source, fichier_csv_final)