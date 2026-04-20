import os
import sys
import io
import json
import shutil
import re
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat
import piexif

# --- CONFIGURATION CHEMINS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

try:
    from pipeline.train_grl import WaterPollutionGRL, get_transforms
except ImportError:
    print("❌ Erreur : Impossible de trouver pipeline/train_grl.py")

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# 🟢 POINTE VERS LA NOUVELLE INTERFACE
@app.route('/')
def index():
    return app.send_static_file('index_2.html')

# --- CHARGEMENT DU MODÈLE IA ---
# ⚠️ MODIFIE CETTE LIGNE AVEC LE CHEMIN EXACT DE TON FICHIER .pth SI BESOIN
MODEL_PATH = os.path.join(BASE_DIR, "models", "grl", "no_mask", "no_grl", "train_all", "best_grl_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Chargement du modèle : {MODEL_PATH} sur {device}...")

global_model = None
val_transform = None

try:
    if os.path.exists(MODEL_PATH):
        global_model = WaterPollutionGRL(num_domains=1, num_classes=2, backbone='efficientnet_v2_m', use_grl=False)
        global_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        global_model.to(device)
        global_model.eval()
        _, val_transform = get_transforms(scope="no_mask")
        print("✅ Modèle IA Prêt pour l'inférence.")
    else:
        print(f"❌ Erreur : Fichier modèle introuvable à {MODEL_PATH}")
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")


# --- ROUTE 1 : SERVEUR D'IMAGES ---
@app.route('/image/<path:filepath>')
def serve_image(filepath):
    if not filepath.startswith('/'):
        filepath = '/' + filepath
    return send_file(filepath)


# --- ROUTE 2 : OUVRIR UN DOSSIER EXISTANT (SANS IA) ---
@app.route('/load_existing', methods=['POST'])
def load_existing():
    data = request.json
    folder_path = data.get('folder_path')
    
    if not folder_path or not os.path.isdir(folder_path):
        return jsonify({"error": "Dossier invalide."}), 400
        
    csv_path = os.path.join(folder_path, "labels.csv")
    results = []
    
    if os.path.exists(csv_path):
        import csv
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = os.path.join(folder_path, row.get('Nom_Image', ''))
                if os.path.exists(img_path):
                    results.append({
                        "name": row.get('Nom_Image'),
                        "path": img_path,
                        "date": row.get('Date', 'N/A'),
                        "time": row.get('Heure', 'N/A'),
                        "score": float(row.get('Confidence_Modele', 0.0)),
                        "label": int(row.get('Label_Utilisateur', 0)),
                        "status": "ok"
                    })
    else:
        return jsonify({"error": "Aucun fichier labels.csv trouvé dans ce dossier. Lancez d'abord un Nouvel Import IA."}), 400
        
    return jsonify({"predictions": results, "dest_dir": folder_path})


# --- ROUTE 3 : NOUVEL IMPORT AVEC IA ---
@app.route('/import_and_predict', methods=['POST'])
def import_and_predict():
    data = request.json
    source_dir = data.get('source_dir')
    workspace_dir = data.get('workspace_dir')
    river = data.get('river', "Unknown").strip().capitalize()
    pov = data.get('pov', "1").strip()
    
    if not all([source_dir, workspace_dir]):
        return jsonify({"error": "Dossiers source ou workspace manquants."}), 400
        
    if not os.path.isdir(source_dir):
        return jsonify({"error": "Dossier source invalide."}), 400
        
    valid_exts = {'.jpg', '.png', '.jpeg'}
    files = []
    for f in os.listdir(source_dir):
        if os.path.splitext(f)[-1].lower() in valid_exts:
            files.append(os.path.join(source_dir, f))
            
    if not files:
        return jsonify({"error": "Aucune image trouvée dans le dossier source."}), 400
        
    files.sort(key=lambda x: os.path.getmtime(x))
    start_date_str = datetime.fromtimestamp(os.path.getmtime(files[0])).strftime("%Y%m%d")
    end_date_str = datetime.fromtimestamp(os.path.getmtime(files[-1])).strftime("%Y%m%d")
    
    folder_name = f"{start_date_str}_to_{end_date_str}" if start_date_str != end_date_str else start_date_str
    dest_dir = os.path.join(workspace_dir, river, str(pov), folder_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    results = []
    
    for f in files:
        base_name = os.path.basename(f)
        last_dot = base_name.rfind('.')
        original_sans_ext = base_name[:last_dot] if last_dot != -1 else base_name
        ext = base_name[last_dot:] if last_dot != -1 else '.jpg'
        
        if re.match(r"^\d{8}_\d{6}_", base_name):
            new_name = base_name
        else:
            mod_time = datetime.fromtimestamp(os.path.getmtime(f))
            prefix = mod_time.strftime("%d%m%Y_%H%M%S")
            new_name = f"{prefix}_{original_sans_ext}_{river}{ext}"
            
        dest_path = os.path.join(dest_dir, new_name)
        if not os.path.exists(dest_path):
            shutil.copy2(f, dest_path)
            
        try:
            img = Image.open(dest_path).convert("RGB")
            gray = img.convert("L")
            avg_brightness = ImageStat.Stat(gray).mean[0]
            
            if avg_brightness < 40:
                # 🟢 NOUVEAU : On supprime physiquement le fichier s'il est trop sombre
                os.remove(dest_path) 
                print(f"🗑️ Image de nuit supprimée : {new_name}")
                continue # On passe à la suivante sans l'ajouter aux résultats
                
            input_tensor = val_transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = global_model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                score_polluted = probs[0][1].item()
                predicted_label = 1 if score_polluted >= 0.5 else 0
                
            match = re.match(r"^(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})", new_name)
            date_fmt = f"{match.group(1)}/{match.group(2)}/{match.group(3)}" if match else "N/A"
            time_fmt = f"{match.group(4)}:{match.group(5)}:{match.group(6)}" if match else "N/A"

            results.append({
                "name": new_name, "path": dest_path, "date": date_fmt, "time": time_fmt,
                "score": score_polluted, "label": predicted_label, "status": "ok"
            })
        except Exception as e:
            print(f"Erreur d'inférence avec {new_name}: {e}")
            
    return jsonify({"predictions": results, "dest_dir": dest_dir})


# --- ROUTE 4 : SAUVEGARDE & ÉCRITURE EXIF ---
@app.route('/save', methods=['POST'])
def save_labels():
    data = request.json
    labels = data.get('labels', [])
    dest_dir = data.get('dest_dir')
    
    if not labels or not dest_dir:
        return jsonify({"error": "Données incomplètes."}), 400
        
    csv_path = os.path.join(dest_dir, "labels.csv")
    
    try:
        import csv
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Date", "Heure", "Nom_Image", "Confidence_Modele", "Label_Utilisateur"])
            
            for item in labels:
                writer.writerow([
                    item.get('date', ''), item.get('time', ''), item.get('name', ''), 
                    item.get('score', 0), item.get('label', 0)
                ])
                
                # --- EXIF TAGGING ULTRA-ROBUSTE ---
                if item.get('status') != 'night':
                    img_path = item.get('path')
                    if os.path.exists(img_path) and img_path.lower().endswith(('.jpg', '.jpeg')):
                        try:
                            try:
                                exif_dict = piexif.load(img_path)
                            except Exception:
                                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
                            
                            if piexif.ExifIFD.MakerNote in exif_dict.get("Exif", {}):
                                del exif_dict["Exif"][piexif.ExifIFD.MakerNote]

                            tag_text = "POLLUE" if item.get('label') == 1 else "PROPRE"
                            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = f"Status: {tag_text}".encode('utf-8')
                            exif_dict["0th"][piexif.ImageIFD.Software] = b"WaterWatcher AI"
                            
                            user_comment = b"ASCII\x00\x00\x00" + f"Water Pollution: {tag_text}".encode('ascii')
                            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
                            
                            exif_bytes = piexif.dump(exif_dict)
                            piexif.insert(exif_bytes, img_path)
                        except Exception as e:
                            print(f"[ERREUR EXIF] Impossible de tagger {item.get('name')}: {e}")

        return jsonify({"success": True, "csv_path": csv_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print(" Démarrage du serveur Flask sur http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)