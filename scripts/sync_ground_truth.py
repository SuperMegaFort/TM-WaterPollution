import os
import pandas as pd
import shutil
import glob

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
GT_DIR   = os.path.join(BASE_DIR, "ground_truth")
CSV_PATH = os.path.join(GT_DIR, "ground_truth.csv")

def sync_ground_truth():
    if not os.path.exists(CSV_PATH):
        print(f"❌ Erreur : {CSV_PATH} introuvable.")
        return

    print(f"📖 Lecture de {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Cartographier l'emplacement actuel de TOUTES les images dans ground_truth/
    print("🔍 Cartographie des dossiers ground_truth/...")
    current_locations = {}
    # On cherche dans tous les sous-dossiers numériques de ground_truth
    for subdir in glob.glob(os.path.join(GT_DIR, "[0-9]*")):
        if os.path.isdir(subdir):
            label_folder = os.path.basename(subdir)
            for img_name in os.listdir(subdir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    current_locations[img_name] = subdir

    stats = {"moved": 0, "copied": 0, "ignored": 0, "errors": 0}

    # 2. Synchroniser chaque image du CSV
    for _, row in df.iterrows():
        img_name = row['Nom_Image']
        
        # Gestion propre des labels (éviter le format 0.0)
        label_val = row['Label']
        if pd.isna(label_val):
            continue
            
        label = str(int(label_val))
            
        target_dir = os.path.join(GT_DIR, label)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, img_name)
        
        # Cas A : L'image est déjà dans ground_truth/ mais peut-être au mauvais endroit
        if img_name in current_locations:
            src_dir = current_locations[img_name]
            src_path = os.path.join(src_dir, img_name)
            
            if src_dir != target_dir:
                try:
                    shutil.move(src_path, target_path)
                    print(f"🚚 MOVE: {img_name} ({os.path.basename(src_dir)} -> {label})")
                    stats["moved"] += 1
                except Exception as e:
                    print(f"❌ Erreur Move {img_name}: {e}")
                    stats["errors"] += 1
            else:
                # Déjà au bon endroit
                stats["ignored"] += 1
                
        # Cas B : L'image n'est pas dans ground_truth/, on la COPIE depuis data/
        else:
            data_path = os.path.join(DATA_DIR, img_name)
            if os.path.exists(data_path):
                try:
                    shutil.copy2(data_path, target_path)
                    print(f"✨ COPY: {img_name} (data -> {label})")
                    stats["copied"] += 1
                except Exception as e:
                    print(f"❌ Erreur Copy {img_name}: {e}")
                    stats["errors"] += 1
            else:
                # L'image n'est trouvée nulle part
                # print(f"⚠️ Image introuvable : {img_name}")
                stats["errors"] += 1

    print("\n" + "="*40)
    print("✅ SYNCHRONISATION TERMINÉE")
    print(f"   - Images déplacées (inter-GT) : {stats['moved']}")
    print(f"   - Images copiées (depuis data) : {stats['copied']}")
    print(f"   - Images déjà bien placées     : {stats['ignored']}")
    print(f"   - Erreurs (introuvables, etc)  : {stats['errors']}")
    print("="*40)

if __name__ == "__main__":
    sync_ground_truth()
