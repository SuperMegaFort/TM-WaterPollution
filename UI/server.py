import os
import sys
import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from PIL import Image

# --- CONFIGURATION CHEMINS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import des fonctions d'entraînement pour récupérer la classe du modèle
from pipeline.train_grl import WaterPollutionGRL, get_transforms

app = Flask(__name__)
CORS(app)

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

@app.route('/predict', methods=['POST'])
def predict():
    if global_model is None:
        return jsonify({"error": "Le modèle PyTorch n'a pas pu être chargé côté serveur."}), 500

    if 'images' not in request.files:
        return jsonify({"error": "Aucune image envoyée"}), 400
        
    files = request.files.getlist('images')
    results = []
    
    # Mode Batch pour que ce soit rapide, mais on le fait image par image par sécurité pour ce proto
    for file in files:
        if file.filename == '':
            continue
            
        try:
            image_bytes = file.read()
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # --- FILTRAGE DE NUIT (Faible Luminosité) ---
            from PIL import ImageStat
            gray = img.convert("L")
            avg_brightness = ImageStat.Stat(gray).mean[0]
            
            if avg_brightness < 40: # Seuil Nuit
                results.append({
                    "name": file.filename,
                    "score": -1,
                    "label": -1,
                    "status": "night"
                })
                continue
            
            # Preprocessing
            input_tensor = val_transform(img).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = global_model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                
                # prob[0][0] = propre, prob[0][1] = pollué
                score_polluted = probs[0][1].item()
                predicted_label = 1 if score_polluted >= 0.5 else 0
                
            results.append({
                "name": file.filename,
                "score": score_polluted,
                "label": predicted_label,
                "status": "ok"
            })
        except Exception as e:
            print(f"Erreur locale avec {file.filename}: {e}")
            results.append({
                "name": file.filename,
                "score": 0.0,
                "label": 0,
                "error": str(e)
            })
            
    return jsonify({"predictions": results})

if __name__ == '__main__':
    print("🚀 Démarrage du serveur Flask sur http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
