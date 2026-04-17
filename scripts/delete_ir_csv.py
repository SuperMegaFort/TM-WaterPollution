import os
import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
# Remplace par le nom exact de ton fichier CSV (ex: 'dataset_ziplo.csv' ou 'mapping_pollution.csv')
CSV_PATH = 'dataset_complet.csv' 
IMAGE_DIR = 'data'
# Nom du nouveau fichier propre qui sera généré
OUTPUT_CSV = 'dataset_complet.csv' 

def sync_csv_with_folder(csv_path, image_dir, output_csv):
    print(f"1. Lecture du fichier CSV : {csv_path}")
    df = pd.read_csv(csv_path)
    initial_count = len(df)
    
    print(f"2. Inventaire des images dans le dossier : {image_dir}")
    # On crée un "Set" (ensemble) des fichiers existants pour une recherche ultra-rapide
    images_presentes = set(os.listdir(image_dir))
    
    print("3. Filtrage du CSV...")
    # On ne garde que les lignes dont le 'Nom_Image' est bien présent dans le dossier
    df_clean = df[df['Nom_Image'].isin(images_presentes)]
    final_count = len(df_clean)
    
    # 4. Sauvegarde
    df_clean.to_csv(output_csv, index=False)
    
    # 5. Bilan
    print("\n" + "="*30)
    print("BILAN DE LA SYNCHRONISATION")
    print("="*30)
    print(f"Lignes avant nettoyage : {initial_count}")
    print(f"Lignes après nettoyage : {final_count}")
    print(f"Lignes IR supprimées   : {initial_count - final_count}")
    print(f"Nouveau fichier créé   : {output_csv}")
    print("="*30)

# Lancement
sync_csv_with_folder(CSV_PATH, IMAGE_DIR, OUTPUT_CSV)