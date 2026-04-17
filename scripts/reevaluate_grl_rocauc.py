import os
import torch
import numpy as np
import glob
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import sys

# Ajouter le dossier racine au path pour les imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "pipeline"))

import train_grl

def reevaluate():
    # Détection du device (priorité MPS pour Mac Silicon)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Utilisation du device : {device}")
    
    # Trouver tous les modèles GRL entraînés
    model_paths = glob.glob(os.path.join(BASE_DIR, "models/grl/**/best_grl_model.pth"), recursive=True)
    
    if not model_paths:
        print("❌ Aucun modèle trouvé dans models/grl/")
        return

    print(f"🔍 Trouvé {len(model_paths)} modèles à évaluer.")
    
    for model_path in model_paths:
        save_dir = os.path.dirname(model_path)
        summary_path = os.path.join(save_dir, "train_summary.txt")
        
        if not os.path.exists(summary_path):
            print(f"⚠️ Summary manquant pour {save_dir}, on ignore.")
            continue
            
        # Charger la configuration depuis le summary
        config = {}
        with open(summary_path, "r") as f:
            for line in f:
                if ":" in line:
                    key, val = line.split(":", 1)
                    config[key.strip()] = val.strip()
        
        backbone = config.get("Backbone", "efficientnet_v2_m")
        kappa = config.get("Meilleur Kappa Validation", "N/A")
        print(f"\n──────────────────────────────────────────────────")
        print(f"📁 Modèle : {backbone} | Folder: ...{save_dir[-40:]}")
        print(f"📊 Kappa actuel : {kappa}")

        # Paramètres d'évaluation
        scope = config.get("Scope", "no_mask")
        use_grl = config.get("GRL", "False") == "True"
        dropout = float(config.get("Dropout", 0.5))
        
        # Chemins vers les données
        csv_path = os.path.join(BASE_DIR, f"data_preprocessed/dataset_{scope}.csv")
        img_dir = os.path.join(BASE_DIR, f"data_preprocessed/{scope}")
        
        if not os.path.exists(csv_path):
            print(f"❌ Erreur: CSV introuvable {csv_path}")
            continue

        # Récupérer les rivières pour reproduire le split exact
        t_rivers_raw = config.get("Train Rivers", "None")
        v_rivers_raw = config.get("Val Rivers", "None")
        
        # Parsing sécurisé (None ou liste de strings)
        t_rivers = None if t_rivers_raw == "None" else eval(t_rivers_raw)
        v_rivers = None if v_rivers_raw == "None" else eval(v_rivers_raw)

        try:
            # Charger les données de validation (split identique grâce au random_state=42 dans train_grl)
            _, val_data, domain_map = train_grl.load_data_split(csv_path, t_rivers, v_rivers, return_domain_map=True)
            _, val_transform = train_grl.get_transforms(scope)
            val_dataset = train_grl.PollutionDataset(val_data, img_dir, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
            
            # Initialiser et charger le modèle
            num_domains = len(domain_map)
            model = train_grl.WaterPollutionGRL(num_domains=num_domains, backbone=backbone, dropout=dropout, use_grl=use_grl)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            
            all_labels = []
            all_probs = []
            
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    # Forcer le alpha à 1.0 (ou neutre) car c'est une évaluation simple
                    if use_grl:
                        class_preds, _ = model(images, alpha=1.0)
                    else:
                        class_preds = model(images)
                    
                    # Probabilités pour la classe 'polluted' (index 1)
                    probs = torch.softmax(class_preds, dim=1)[:, 1]
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    
                    # Nettoyage mémoire express
                    del images, labels, class_preds, probs
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            # Calcul du ROC-AUC
            if len(set(all_labels)) > 1:
                auc = roc_auc_score(all_labels, all_probs)
            else:
                auc = 0.5 # Valeur par défaut si une seule classe présente
                
            print(f"✅ Nouveau ROC-AUC : {auc:.4f}")
            
            # Mise à jour du fichier train_summary.txt
            with open(summary_path, "r") as f:
                lines = f.readlines()
                
            new_lines = []
            auc_written = False
            for line in lines:
                if "Meilleure ROC-AUC Validation" in line:
                    new_lines.append(f"Meilleure ROC-AUC Validation : {auc:.4f}\n")
                    auc_written = True
                else:
                    new_lines.append(line)
            
            if not auc_written:
                # Insérer après le Kappa pour garder un ordre propre
                for i, line in enumerate(list(new_lines)):
                    if "Meilleur Kappa Validation" in line:
                        new_lines.insert(i+1, f"Meilleure ROC-AUC Validation : {auc:.4f}\n")
                        break
            
            with open(summary_path, "w") as f:
                f.writelines(new_lines)
                
        except Exception as e:
            print(f"❌ Erreur lors de l'évaluation du modèle {save_dir} : {e}")

    print("\n🎉 Re-évaluation terminée pour tous les modèles GRL/CNN !")

if __name__ == "__main__":
    reevaluate()
