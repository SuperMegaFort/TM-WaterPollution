import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Le dossier contenant tes images (peut contenir des sous-dossiers, le script va fouiller)
TARGET_DIR = 'data'  # Remplace par 'dataset/train' ou 'dataset/val' si tes images sont déjà triées
THRESHOLD = 5.0      # Le seuil de variance de couleur en dessous duquel l'image est jugée N&B

def process_ir_images(directory):
    removed_count = 0
    checked_count = 0
    skipped_count = 0
    
    # Parcourir tous les fichiers et sous-dossiers
    for root, dirs, files in os.walk(directory):
        # On filtre pour ne garder que les images
        images = [f for f in files if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        
        if not images:
            continue
            
        print(f"\nAnalyse du dossier : {root}")
        
        for file in tqdm(images, desc="Vérification des images"):
            filepath = os.path.join(root, file)
            
            # 1. OPTIMISATION : Extraire l'heure depuis le nom du fichier
            # Format attendu : 25032021_081000_... (Les 8 premiers chars sont la date, puis '_', puis l'heure)
            match = re.match(r"^\d{8}_(\d{2})", file)
            
            check_image = True # Par défaut on vérifie
            
            if match:
                heure = int(match.group(1))
                # On NE vérifie QUE si l'heure est >= 16 (16h à 23h) OU < 10 (00h à 09h)
                if 9 <= heure < 16:
                    check_image = False
            else:
                # Si le nom du fichier n'est pas au bon format, on vérifie l'image par sécurité
                pass
                
            # 2. VÉRIFICATION INFRAROUGE
            if check_image:
                checked_count += 1
                try:
                    img = Image.open(filepath).convert('RGB')
                    img_np = np.array(img)
                    
                    # Calcul de la variance des couleurs
                    std_dev = np.std(img_np, axis=2)
                    mean_std_dev = np.mean(std_dev)
                    
                    # Si c'est en dessous du seuil, c'est du N&B (Infrarouge) -> On supprime !
                    if mean_std_dev < THRESHOLD:
                        os.remove(filepath)
                        removed_count += 1
                except Exception as e:
                    print(f"Erreur de lecture avec {file}: {e}")
            else:
                skipped_count += 1

    # 3. BILAN
    print("\n" + "="*40)
    print("BILAN DU NETTOYAGE INFRAROUGE")
    print("="*40)
    print(f"Images ignorées (journée 10h-16h) : {skipped_count}")
    print(f"Images analysées (nuit/matin/soir) : {checked_count}")
    print(f"Images IR supprimées avec succès   : {removed_count}")
    print("="*40)

if __name__ == "__main__":
    process_ir_images(TARGET_DIR)