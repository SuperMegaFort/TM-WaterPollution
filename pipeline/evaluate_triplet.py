"""
==============================================================================
 ÉVALUATION : Few-Shot Anomaly Detection via Triplet Extractor
==============================================================================
 Évalue l'espace latent (Metric Learning). 
 Le modèle requiert un "Pointeur" (Reference Image) d'eau PROPRE de la 
 nouvelle rivière. Il comparera mathématiquement toutes les autres images
 à ce Pointeur. Si la distance explose -> Pollution détectée !
==============================================================================
"""

import os
import cv2
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report, confusion_matrix

from train_triplet import load_data_split, WaterPollutionSiamese, get_transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(BASE_DIR, "models", "triplet")
RESULTS_DIR     = os.path.join(BASE_DIR, "evaluation_results", "triplet")

# ─────────────────────────────────────────────
# UTILS VIZ (Grad-CAM & Deprocess)
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x_query, x_anchor):
        """ Grad-CAM calculé sur la DISTANCE par rapport à l'ancre """
        emb_q = self.model(x_query)
        emb_a = self.model(x_anchor)
        
        # Le Score est la distance L2 : on veut voir ce qui augmente la distance
        distance = F.pairwise_distance(emb_a, emb_q)
        
        self.model.zero_grad()
        distance.backward(retain_graph=True)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU sur la map
        cam = cv2.resize(cam, (x_query.shape[-1], x_query.shape[-2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, distance.item()

def deprocess_image(tensor, is_tensor=False):
    if is_tensor:
        mean, std = np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])
    else:
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return np.uint8(255 * img)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[:, :, ::-1] # BGR to RGB
    img = np.float32(img) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class SimpleInferenceDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        folder = "clean" if item["label"] == 0 else "polluted"
        path = os.path.join(self.img_dir, folder, item["name"])
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, item["label"], item["name"], path

def evaluate_few_shot(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    model_path = Path(args.model).resolve()
    try:
        rel_path = model_path.parent.relative_to(MODELS_DIR)
    except ValueError:
        rel_path = model_path.parent.name
        
    eval_str = f"eval_{'_'.join([r.lower() for r in args.val_rivers])}" if args.val_rivers else "eval_all"
    RESULT_RUN_DIR = os.path.join(RESULTS_DIR, str(rel_path), eval_str)
    os.makedirs(RESULT_RUN_DIR, exist_ok=True)

    # 1. Charger les données
    _, val_data = load_data_split(args.csv, args.train_rivers, args.val_rivers)
    
    # Trouver une image de référence propre (Anchor)
    clean_images = [d for d in val_data if d["label"] == 0]
    if not clean_images:
        raise ValueError("Le dataset de validation ne contient aucune image 'Propre' (0) pour servir de Point de Référence !")
    reference_item = random.choice(clean_images)
    print(f"⚓ Point d'ancrage (Anchor) : {reference_item['name']}")

    # 2. Transforms et Dataset
    is_tensor = "tensor" in args.train_scope
    _, val_t = get_transforms(is_tensor=is_tensor)
    val_dataset = SimpleInferenceDataset(val_data, args.dir, transform=val_t)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False) # Batch 1 pour simplifier Grad-CAM/Reference

    # 3. Charger le modèle
    model = WaterPollutionSiamese(backbone='efficientnet_v2_s', embedding_dim=128).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    model.eval()
    print("✅ Modèle Latent chargé.")

    # ======= EMBEDDING DE REFERENCE (ANCHOR) =======
    with torch.no_grad():
        folder = "clean" if reference_item["label"] == 0 else "polluted"
        ref_path = os.path.join(args.dir, folder, reference_item["name"])
        ref_img_pil = Image.open(ref_path).convert("RGB")
        ref_tensor = val_t(ref_img_pil).unsqueeze(0).to(device)
        ref_embedding = model(ref_tensor)

    # ======= INFERENCE =======
    all_distances = []
    all_labels = []
    all_names = []
    all_paths = []
    
    from tqdm import tqdm
    print("\n⏳ Calcul des distances dans l'espace latent...")
    with torch.no_grad():
        for img_tensor, label, name, path in tqdm(val_loader, desc="Inference"):
            if name[0] == reference_item["name"]: continue
            
            img_tensor = img_tensor.to(device)
            embedding = model(img_tensor)
            dist = F.pairwise_distance(ref_embedding, embedding).item()
            
            all_distances.append(dist)
            all_labels.append(label.item())
            all_names.append(name[0])
            all_paths.append(path[0])

    # ======= ANALYSE PR-AUC & ROC-AUC =======
    roc_auc = roc_auc_score(all_labels, all_distances)
    precision, recall, thresholds = precision_recall_curve(all_labels, all_distances)
    
    # Seuil optimal F1
    fscore = (2 * precision * recall) / (precision + recall + 1e-8)
    ix = np.argmax(fscore)
    optimal_threshold = thresholds[ix] if ix < len(thresholds) else thresholds[-1]
    
    preds = (np.array(all_distances) >= optimal_threshold).astype(int)

    print("\n" + "="*50)
    print(" RAPPORT D'ÉVALUATION SIAMOY")
    print("="*50)
    print(f"ROC-AUC SCORE : {roc_auc:.4f}")
    print(f"Seuil Optimal : {optimal_threshold:.4f}")
    report = classification_report(all_labels, preds, target_names=["Propre", "Pollué"], zero_division=0)
    print(report)

    # ======= SAUVEGARDE GRAPHIQUES =======
    # Matrice
    cm = confusion_matrix(all_labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Propre", "Pollué"], yticklabels=["Propre", "Pollué"])
    plt.title(f"Confusion Matrix (Dist Threshold={optimal_threshold:.2f})")
    plt.savefig(os.path.join(RESULT_RUN_DIR, "confusion_matrix.png"))
    plt.close()

    # Predictions CSV
    import pandas as pd
    pd.DataFrame({
        "Image": all_names, "Path": all_paths,
        "True_Label": all_labels, "Pred_Label": preds, "Score": all_distances
    }).to_csv(os.path.join(RESULT_RUN_DIR, "predictions.csv"), index=False)

    # ======= GRAD-CAM SAMPLES =======
    print("\n📸 Génération des Heatmaps de distance (Siam-CAM)...")
    target_layer = model.feature_extractor[0][-1] # EfficientNet last feature layer
    cam_extractor = GradCAM(model, target_layer)
    
    sample_indices = random.sample(range(len(all_names)), min(10, len(all_names)))
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    plt.suptitle(f"Siam-CAM : Zones de divergence par rapport à la référence\n{str(rel_path)}", fontsize=16)
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        q_path = all_paths[idx]
        q_img_pil = Image.open(q_path).convert("RGB")
        q_tensor = val_t(q_img_pil).unsqueeze(0).to(device).requires_grad_()
        
        # Heatmap (Q vs Reference Anchor)
        cam_mask, distance = cam_extractor(q_tensor, ref_tensor)
        
        orig_img = deprocess_image(val_t(q_img_pil), is_tensor=is_tensor)
        cam_image = show_cam_on_image(orig_img, cam_mask)
        
        true_lbl = all_labels[idx]
        pred_lbl = preds[idx]
        color = "green" if true_lbl == pred_lbl else "red"
        
        axes[i].imshow(cam_image)
        axes[i].set_title(f"T: {true_lbl} | P: {pred_lbl}\nDist: {distance:.2f}", color=color)
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_RUN_DIR, "grad_cam_samples.png"))
    plt.close()
    
    # ======= EXPORT JSON SUMMARY =======
    summary = {
        "model_type": "Triplet",
        "scope": args.train_scope,
        "train_rivers": args.train_rivers if args.train_rivers else "all",
        "val_rivers": args.val_rivers if args.val_rivers else "all",
        "roc_auc": float(roc_auc),
        "f1": float(fscore[ix]),
        "threshold": float(optimal_threshold)
    }
    with open(os.path.join(RESULT_RUN_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"📊 Résumé JSON sauvegardé dans : {os.path.join(RESULT_RUN_DIR, 'summary.json')}")

    print(f"✅ Évaluation terminée. Résultats dans : {RESULT_RUN_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Triplet SIAM Model")
    parser.add_argument("--model", type=str, required=True, help="Nom complet du fichier pth (ex: models/triplet/train_arve_ziplo/best_siamese_model.pth)")
    parser.add_argument("--dir", default=None, help="Dossier contenant les images")
    parser.add_argument("--csv", default=None, help="Fichier CSV")
    
    parser.add_argument("--train_rivers", nargs="+", default=None)
    parser.add_argument("--val_rivers", nargs="+", default=None)
    parser.add_argument('--train_scope', type=str, default="manual_crop",
                        choices=["no_mask", "manual_crop", "manual_crop_tensor"],
                        help="Dossier de données dans data_preprocessed/")
    
    # NOUVEAU: Le Pointeur De Référence !
    parser.add_argument("--reference_clean", type=str, default=None, help="Chemin vers UNE photo propre du nouveau cours d'eau.")
    
    args = parser.parse_args()
    
    if args.dir is None:
        args.dir = os.path.join(BASE_DIR, "data_preprocessed", args.train_scope)
    if args.csv is None:
        csv_name = f"dataset_{args.train_scope}.csv"
        args.csv = os.path.join(BASE_DIR, "data_preprocessed", csv_name)
    
    evaluate_few_shot(args)
