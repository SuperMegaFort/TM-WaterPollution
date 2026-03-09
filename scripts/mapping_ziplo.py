import os
import re
import pandas as pd
from datetime import datetime

def associer_pollution_ziplo(dossier_images, fichier_csv, fichier_sortie):
    print("1. Lecture du fichier CSV des pollutions Ziplo...")
    
    # Lire le fichier CSV
    df_poll = pd.read_csv(fichier_csv)
    
    # CORRECTION ICI : Ajout de errors='coerce'
    # Cela va forcer la conversion. Si c'est écrit "Nuit", ça deviendra 'NaT' (Not a Time)
    df_poll['Debut_dt'] = pd.to_datetime(df_poll['Date'] + ' ' + df_poll['Heure début'], format='%d.%m.%Y %H:%M', errors='coerce')
    df_poll['Fin_dt'] = pd.to_datetime(df_poll['Date'] + ' ' + df_poll['Heure fin'], format='%d.%m.%Y %H:%M', errors='coerce')
    
    # On supprime toutes les lignes où l'heure n'a pas pu être convertie (les fameuses "Nuits")
    df_poll = df_poll.dropna(subset=['Debut_dt', 'Fin_dt'])
    
    print(f"2. Parcours du dossier d'images : {dossier_images}")
    donnees_finales = []
    images_trouvees = 0
    images_ignorees = 0
    
    for nom_fichier in os.listdir(dossier_images):
        nom_minuscule = nom_fichier.lower()
        
        # On ne traite QUE les images, et QUE celles qui contiennent "ziplo"
        if nom_minuscule.endswith(('.jpg', '.jpeg', '.png')) and 'ziplo' in nom_minuscule:
            
            # Extraire la date et l'heure au début du nom (Ex: 08122025_153500)
            match = re.match(r"^(\d{8})_(\d{6})_", nom_fichier)
            
            if match:
                date_str = match.group(1)
                heure_str = match.group(2)
                
                try:
                    # Reconstruire une vraie date Python
                    date_image = datetime(
                        year=int(date_str[4:]),
                        month=int(date_str[2:4]),
                        day=int(date_str[:2]),
                        hour=int(heure_str[:2]),
                        minute=int(heure_str[2:4]),
                        second=int(heure_str[4:])
                    )
                    
                    # Chercher si la date de l'image tombe PILE dans un intervalle de pollution
                    match_intervalle = df_poll[(df_poll['Debut_dt'] <= date_image) & (date_image <= df_poll['Fin_dt'])]
                    
                    if not match_intervalle.empty:
                        classe_pollution = int(match_intervalle.iloc[0]['Pullution'])
                        donnees_finales.append({
                            "Nom_Image": nom_fichier,
                            "Classe": classe_pollution
                        })
                        images_trouvees += 1
                    else:
                        images_ignorees += 1
                        
                except Exception as e:
                    pass

    print("\n3. Création du fichier CSV final...")
    df_final = pd.DataFrame(donnees_finales)
    
    if not df_final.empty:
        df_final.to_csv(fichier_sortie, index=False)
        print(f"Succès ! {images_trouvees} images polluées Ziplo ont été associées à leur classe.")
        print(f"Fichier généré : {fichier_sortie}")
    else:
        print("Attention : Aucune image ne correspondait aux intervalles de temps.")
        
    if images_ignorees > 0:
        print(f"Note : {images_ignorees} images Ziplo ont été ignorées (pas de pollution signalée à leur heure exacte).")

# --- PARAMÈTRES ---
dossier_des_images = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/data"
fichier_csv_ziplo = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/Photos/ZIPLO 2025-2026/image_ziplo.csv" 
fichier_csv_final = "dataset_ziplo.csv"

# --- Lancement ---
associer_pollution_ziplo(dossier_des_images, fichier_csv_ziplo, fichier_csv_final)