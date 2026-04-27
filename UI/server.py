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

# Import des fonctions d'entraînement pour récupérer la classe du modèle
from pipeline.train_grl import WaterPollutionGRL, get_transforms

app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)), static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index_v1.html')

# Modèle demandé par l'utilisateur : no mask, no grl, no siamois, train all
MODEL_PATH = os.path.join(BASE_DIR, "models", "grl", "no_mask", "no_grl", "train_all", "best_grl_model.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

print(f"Chargement du modèle : {MODEL_PATH} sur {device}...")

global_model = None

try:
    if os.path.exists(MODEL_PATH):
        # Instanciation (sans GRL, backbone = efficientnet_v2_m)
        global_model = WaterPollutionGRL(num_domains=1, num_classes=2, backbone='efficientnet_v2_m', use_grl=False)
        global_model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        global_model.to(device)
        global_model.eval()
        print("✅ Modèle IA Prêt pour l'inférence.")
    else:
        print("❌ Erreur : Fichier modèle introuvable :", MODEL_PATH)
except Exception as e:
    print(f"❌ Erreur lors du chargement : {e}")

# Transform "no_mask" classique : RGB (ImageNet normalization)
_, val_transform = get_transforms(scope="no_mask")

@app.route('/image/<path:filepath>')
def serve_image(filepath):
    if not filepath.startswith('/'):
        filepath = '/' + filepath
    return send_file(filepath)

@app.route('/import_and_predict', methods=['POST'])
def import_and_predict():
    data = request.json
    source_dir = data.get('source_dir')
    river = data.get('river', "Unknown").strip().capitalize()
    pov = data.get('pov', "1").strip()
    
    if not source_dir:
        return jsonify({"error": "Dossier source manquant."}), 400
        
    if not os.path.isdir(source_dir):
        return jsonify({"error": "Dossier source invalide."}), 400
        
    valid_exts = {'.jpg', '.png', '.jpeg'}
    files = []
    for f in os.listdir(source_dir):
        if os.path.splitext(f)[-1].lower() in valid_exts:
            files.append(os.path.join(source_dir, f))
            
    if not files:
        return jsonify({"error": "Aucune image trouvée dans le dossier source."}), 400
        
    dest_dir = source_dir
    
    results = []
    
    for f in files:
        base_name = os.path.basename(f)
        dest_path = f  # Pas de copie, évaluation sur place dans le dossier source
        new_name = base_name
            
        try:
            # Toujours utiliser l'image copiée
            img = Image.open(dest_path).convert("RGB")
            
            gray = img.convert("L")
            avg_brightness = ImageStat.Stat(gray).mean[0]
            
            if avg_brightness < 40:
                results.append({
                    "name": new_name,
                    "path": dest_path,
                    "score": -1,
                    "label": -1,
                    "status": "night"
                })
                continue
                
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
                "name": new_name,
                "path": dest_path,
                "date": date_fmt,
                "time": time_fmt,
                "score": score_polluted,
                "label": predicted_label,
                "status": "ok"
            })
        except Exception as e:
            print(f"Erreur d'inférence avec {new_name}: {e}")
            
    return jsonify({"predictions": results, "dest_dir": dest_dir})

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
                    item.get('date', ''), 
                    item.get('time', ''), 
                    item.get('name', ''), 
                    item.get('score', 0), 
                    item.get('label', 0)
                ])
                
                # --- EXIF TAGGING (Ultra-Robuste) ---
                if item.get('status') != 'night':
                    img_path = item.get('path')
                    if os.path.exists(img_path) and img_path.lower().endswith(('.jpg', '.jpeg')):
                        try:
                            # 1. Chargement sécurisé (crée un dico vide si pas d'EXIF)
                            try:
                                exif_dict = piexif.load(img_path)
                            except Exception:
                                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
                            
                            # 2. Nettoyage des MakerNotes (souvent corrompus par les appareils photo, fait planter la sauvegarde)
                            if piexif.ExifIFD.MakerNote in exif_dict.get("Exif", {}):
                                del exif_dict["Exif"][piexif.ExifIFD.MakerNote]

                            # Définition du label
                            tag_text = "POLLUE" if item.get('label') == 1 else "PROPRE"
                            
                            # 3. Ajout dans la Description (Visible nativement sur Mac et Windows)
                            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = f"{tag_text}".encode('utf-8')
                            
                            # 4. Ajout dans Software (Pour identifier l'IA)
                            exif_dict["0th"][piexif.ImageIFD.Software] = b"WaterWatcher AI"
                            
                            # 5. Ajout dans UserComment (Standard EXIF universel)
                            user_comment = b"ASCII\x00\x00\x00" + f"{tag_text}".encode('ascii')
                            exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
                            
                            # 6. Sauvegarde forcée
                            exif_bytes = piexif.dump(exif_dict)
                            piexif.insert(exif_bytes, img_path)
                            print(f"[EXIF] Tag {tag_text} ajouté avec succès sur {item.get('name')}")
                            
                        except Exception as e:
                            print(f"[ERREUR EXIF] Impossible de tagger {item.get('name')}: {e}")

        return jsonify({"success": True, "csv_path": csv_path})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Démarrage du serveur Flask sur http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
