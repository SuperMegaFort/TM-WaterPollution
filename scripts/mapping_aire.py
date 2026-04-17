import os
import shutil
import pandas as pd
from datetime import datetime
from PIL import Image
from PIL.ExifTags import TAGS

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = 'Photos/Aire 2021/GraphiqueAirepiègePhotos_1.csv'
INPUT_DIR = 'Photos/Aire 2021/data'  # Remplace par le chemin correct si besoin
OUTPUT_DIR = 'data'                   # Le dossier où seront stockées les images renommées
LOCATION_NAME = 'Aire'                # Le suffixe pour le nouveau nom de fichier

# Date du passage à l'heure d'été (Le jour de transition inclus dans l'hiver, le lendemain en été)
# /!\ Attention: Mis sur 2021 par défaut au vu de ton fichier. 
# Modifie l'année si tu as un dossier 2025.
TRANSITION_DATE_STR = "27.03.2021" 
transition_date = datetime.strptime(TRANSITION_DATE_STR, "%d.%m.%Y").date()

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================
def extract_exif_date(image_path):
    """Extrait la date de création originale depuis les métadonnées de l'image (EXIF)."""
    try:
        image = Image.open(image_path)
        exifdata = image._getexif()
        if exifdata is not None:
            for tag_id, value in exifdata.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal':
                    val_str = str(value).strip().strip('\x00')
                    return datetime.strptime(val_str, '%Y:%m:%d %H:%M:%S')
    except Exception:
        pass
    
    mtime = os.path.getmtime(image_path)
    return datetime.fromtimestamp(mtime)

def round_to_30min(dt):
    """Arrondit l'heure à la tranche de 30 minutes inférieure."""
    minute = 0 if dt.minute < 30 else 30
    return dt.replace(minute=minute, second=0, microsecond=0)

# ==========================================
# SCRIPT PRINCIPAL
# ==========================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH, header=None)
dates_csv = df[0].dropna().values 

mapping_data = []

print("Traitement des images en cours...")

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
        
    filepath = os.path.join(INPUT_DIR, filename)
    
    dt = extract_exif_date(filepath)
    date_str = dt.strftime("%d.%m.%Y")
    
    if date_str not in dates_csv:
        continue
        
    row_idx = df[df[0] == date_str].index[0]
    time_bin = round_to_30min(dt).strftime("%H:%M:00")
    is_summer = dt.date() > transition_date
    
    try:
        if is_summer:
            row_times = df.iloc[1, 1:49].values
            col_offset = list(row_times).index(time_bin)
            col_idx = 1 + col_offset
        else:
            row_times = df.iloc[0, 3:51].values
            col_offset = list(row_times).index(time_bin)
            col_idx = 3 + col_offset
    except ValueError:
        continue 
        
    # 4. Lire le label de pollution et appliquer la nouvelle règle (Vide = 0)
    label = df.iloc[row_idx, col_idx]
    
    # Si la case est vide (NaN), c'est de l'eau propre (0)
    if pd.isna(label):
        label_str = "0"
    else:
        label_str = str(label).strip()
        
        # Si la case contient juste des espaces ou est vide, c'est de l'eau propre (0)
        if label_str == "":
            label_str = "0"
        # Si c'est la nuit, on ignore l'image
        elif label_str.lower() == "nuit":
            continue
        # Si c'est un chiffre décimal exporté par pandas (ex: "4.0"), on le nettoie en "4"
        elif label_str.endswith('.0'):
            label_str = label_str[:-2]
            
    # 5. Renommage et copie
    original_name = os.path.splitext(filename)[0]
    new_name = f"{dt.strftime('%d%m%Y_%H%M%S')}_{original_name}_{LOCATION_NAME}.jpg"
    
    new_filepath = os.path.join(OUTPUT_DIR, new_name)
    shutil.copy2(filepath, new_filepath)
    
    # 6. Ajout au DataFrame final
    mapping_data.append({
        'Nom_Image': new_name,
        'Classe': label_str
    })

# 7. Sauvegarde
mapping_df = pd.DataFrame(mapping_data)

if len(mapping_df) > 0:
    mapping_df.to_csv('mapping_pollution.csv', index=False)
    print(f"✅ Terminé ! {len(mapping_df)} images validées et copiées dans le dossier '{OUTPUT_DIR}'.")
    print("✅ Le fichier 'mapping_pollution.csv' a été généré avec succès.")
    
    # Petit affichage du résumé des classes trouvées
    print("\nRépartition des classes trouvées :")
    print(mapping_df['Classe'].value_counts())
else:
    print("⚠️ Aucune image valide n'a été trouvée.")