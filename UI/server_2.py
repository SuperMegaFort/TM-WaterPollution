import os
import sys
import io
import json
import shutil
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat
import piexif
import scipy.signal

# --- CONFIGURATION CHEMINS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

try:
    from pipeline.train_grl import WaterPollutionGRL, get_transforms
except ImportError:
    print("❌ Erreur : Impossible de trouver pipeline/train_grl.py")

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

@app.route('/get_config', methods=['GET'])
def get_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify({})

@app.route('/set_config', methods=['POST'])
def set_config():
    data = request.json
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
    # Sécurité multi-plateforme : Windows vs Mac
    if os.name == 'nt': # Si on est sous Windows
        if filepath.startswith('/'):
            filepath = filepath[1:]
    else:
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
        
    results = []
    valid_exts = {'.jpg', '.png', '.jpeg'}
    
    files = sorted([f for f in os.listdir(folder_path) if os.path.splitext(f)[-1].lower() in valid_exts])
    
    if not files:
        return jsonify({"error": "Aucune image trouvée dans ce dossier."}), 400
        
    for f in files:
        img_path = os.path.join(folder_path, f)
        
        score = 0.0
        user_label = 0
        ai_label = 0
        try:
            exif_dict = piexif.load(img_path)
            user_comment = exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, b"")
            if user_comment.startswith(b"ASCII\x00\x00\x00"):
                json_str = user_comment[8:].decode('ascii')
                data_exif = json.loads(json_str)
                score = data_exif.get("score", 0.0)
                user_label = data_exif.get("user", 0)
                ai_label = data_exif.get("ai", 0)
            else:
                desc = exif_dict.get("0th", {}).get(piexif.ImageIFD.ImageDescription, b"").decode('utf-8')
                if desc:
                    data_exif = json.loads(desc)
                    score = data_exif.get("score", 0.0)
                    user_label = data_exif.get("user", 0)
                    ai_label = data_exif.get("ai", 0)
        except Exception:
            pass
            
        match = re.match(r"^(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})", f)
        date_fmt = f"{match.group(1)}/{match.group(2)}/{match.group(3)}" if match else "N/A"
        time_fmt = f"{match.group(4)}:{match.group(5)}:{match.group(6)}" if match else "N/A"
        
        results.append({
            "name": f,
            "path": img_path,
            "date": date_fmt,
            "time": time_fmt,
            "score": float(score),
            "label": int(user_label),
            "ai_label": int(ai_label),
            "status": "ok"
        })
        
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
    
    # --- PHASE 1 : COPIE & INFÉRENCE EN MULTITHREAD ---
    def process_image(f):
        base_name = os.path.basename(f)
        last_dot = base_name.rfind('.')
        ext = base_name[last_dot:] if last_dot != -1 else '.jpg'
        
        if re.match(r"^\d{8}_\d{6}_", base_name):
            new_name = base_name
        else:
            mod_time = datetime.fromtimestamp(os.path.getmtime(f))
            prefix = mod_time.strftime("%d%m%Y_%H%M%S")
            new_name = f"{prefix}_{river}{ext}"
            
        dest_path = os.path.join(dest_dir, new_name)
        if not os.path.exists(dest_path):
            shutil.copy2(f, dest_path)
            
        try:
            # 1. Ouverture et lecture en mémoire sécurisée
            with Image.open(dest_path) as img:
                img_rgb = img.convert("RGB")
                gray = img.convert("L")
                
                # Calcul de la luminosité
                avg_brightness = ImageStat.Stat(gray).mean[0]

                # Calcul de la variance RGB (Détection IR / Noir et Blanc)
                stat = ImageStat.Stat(img_rgb)
                r_mean, g_mean, b_mean = stat.mean[:3]
                color_variance = max(abs(r_mean - g_mean), abs(g_mean - b_mean), abs(r_mean - b_mean))

            # 2. Le fichier est maintenant déverrouillé (fin du 'with')
            # On vérifie les seuils pour le supprimer physiquement
            if avg_brightness < 40 or color_variance < 3.0:
                os.remove(dest_path) 
                print(f"🗑️ Image de nuit/IR supprimée : {new_name}")
                return None  # <-- TRES IMPORTANT : Continue la boucle, ne fais pas de 'return'
                
            # 3. Inférence PyTorch sur l'image valide
            input_tensor = val_transform(img_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = global_model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                score_polluted = probs[0][1].item()
                predicted_label = 1 if score_polluted >= 0.5 else 0
                
            # 4. Formatage des dates
            match = re.match(r"^(\d{2})(\d{2})(\d{4})_(\d{2})(\d{2})(\d{2})", new_name)
            date_fmt = f"{match.group(1)}/{match.group(2)}/{match.group(3)}" if match else "N/A"
            time_fmt = f"{match.group(4)}:{match.group(5)}:{match.group(6)}" if match else "N/A"

            return {
                "name": new_name, "path": dest_path, "date": date_fmt, "time": time_fmt,
                "score": score_polluted, "label": predicted_label, "status": "ok"
            }
        except Exception as e:
            print(f"Erreur d'inférence avec {new_name}: {e}")
            return None

    max_threads = min(32, (os.cpu_count() or 1) * 2)
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = {executor.submit(process_image, f): f for f in files}
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
                
    # Trier les résultats par date/nom pour la timeline
    results.sort(key=lambda x: x["name"])
            
    # --- PHASE 2 : LISSAGE TEMPÓREL (FILTRE MÉDIAN) ---
    scores = [r["score"] for r in results if r["status"] == "ok"]
    
    if len(scores) > 3:
        smoothed_scores = scipy.signal.medfilt(scores, kernel_size=11)
        idx = 0
        for r in results:
            if r["status"] == "ok":
                r["score"] = float(smoothed_scores[idx])
                r["label"] = 1 if r["score"] >= 0.5 else 0
                idx += 1    
                
    # --- PHASE 3 : ÉCRITURE DES EXIF EN MULTITHREAD ---
    def write_exif(r):
        # 1. On extrait le chemin et le nom à partir de 'r' (et non 'item')
        img_path = r.get('path')
        img_name = r.get('name')
        
        # Sécurité : vérifier que le chemin existe bien
        if not img_path or not os.path.exists(img_path):
            return

        try:
            # 2. Chargement des EXIF existants
            try:
                exif_dict = piexif.load(img_path)
            except Exception:
                exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "Interop": {}, "1st": {}, "thumbnail": None}
            
            # Nettoyage des MakerNotes corrompues
            if piexif.ExifIFD.MakerNote in exif_dict.get("Exif", {}):
                del exif_dict["Exif"][piexif.ExifIFD.MakerNote]

            # 3. Formatage universel avec le Score
            tag_text = "POLLUE" if r.get('label') == 1 else "PROPRE"
            score_val = r.get('score', 0.0)
            
            info_str = f"Status: {tag_text} | Confiance: {score_val:.4f}"
            
            # 4. Standard EXIF (Mac/Linux)
            exif_dict["0th"][piexif.ImageIFD.ImageDescription] = info_str.encode('utf-8')
            exif_dict["0th"][piexif.ImageIFD.Software] = b"WaterWatcher AI"
            exif_dict["Exif"][piexif.ExifIFD.UserComment] = b"ASCII\x00\x00\x00" + info_str.encode('ascii', 'ignore')
            
            # 5. Standard Windows (Tags XP) - Encodage UTF-16LE
            win_encoded = info_str.encode('utf-16le')
            exif_dict["0th"][40092] = win_encoded
            exif_dict["0th"][40095] = win_encoded
            
            # 6. Sauvegarde
            exif_bytes = piexif.dump(exif_dict)
            piexif.insert(exif_bytes, img_path)
            
        except Exception as e:
            # On utilise 'img_name' extrait de 'r'
            print(f"[ERREUR EXIF] Impossible de tagger {img_name}: {e}")

    # Lancement du Multithreading
    from concurrent.futures import ThreadPoolExecutor
    
    # max_workers=None laisse Python choisir le nombre optimal de threads selon ton CPU
    with ThreadPoolExecutor(max_workers=None) as executor:
        for r in results:
            if r["status"] == "ok":
                executor.submit(write_exif, r)

    return jsonify({"predictions": results, "dest_dir": dest_dir})
# --- ROUTE 4 : SAUVEGARDE & ÉCRITURE EXIF ---
@app.route('/save', methods=['POST'])
def save_labels():
    data = request.json
    labels = data.get('labels', [])
    dest_dir = data.get('dest_dir')
    
    if not labels or not dest_dir:
        return jsonify({"error": "Données incomplètes."}), 400
        
    try:
        for item in labels:
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

                        ai_val = 1 if item.get('ai_label') == 1 else 0
                        user_val = 1 if item.get('label') == 1 else 0
                        score_val = float(item.get('score', 0.0))
                        json_tag = json.dumps({"user": user_val, "ai": ai_val, "score": score_val})
                        
                        exif_dict["0th"][piexif.ImageIFD.ImageDescription] = json_tag.encode('utf-8')
                        exif_dict["0th"][piexif.ImageIFD.Software] = b"WaterWatcher AI"
                        
                        user_comment = b"ASCII\x00\x00\x00" + json_tag.encode('ascii')
                        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment
                        
                        exif_bytes = piexif.dump(exif_dict)
                        piexif.insert(exif_bytes, img_path)
                    except Exception as e:
                        print(f"[ERREUR EXIF] Impossible de tagger {item.get('name')}: {e}")

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print(" Démarrage du serveur Flask sur http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)