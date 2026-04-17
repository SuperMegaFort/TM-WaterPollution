"""
==============================================================================
 EVALUATION & GRAD-CAM — Water Pollution Detection
==============================================================================
 Évalue le modèle entraîné sur l'ensemble de validation sans le ré-entraîner.
 Génère :
   1. Une Matrice de Confusion (sauvegardée dans ./evaluation_results/)
   2. Un Rapport de Classification (Précision, Rappel, F1-Score)
   3. Des heatmaps Grad-CAM pour comprendre où le modèle regarde.
==============================================================================
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score, accuracy_score, f1_score, roc_auc_score
import cv2
import argparse
from pathlib import Path

# Import depuis train_grl.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from train_grl import WaterPollutionGRL, load_data_split, PollutionDataset, get_transforms, BASE_DIR, MODELS_DIR
except ImportError:
    print("Erreur : impossible d'importer train_grl. Assurez-vous d'être dans le bon environnement.")
    sys.exit(1)

EVAL_DIR = os.path.join(os.path.dirname(MODELS_DIR), "evaluation_results")
os.makedirs(EVAL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# GRAD-CAM CUSTOM IMPLEMENTATION
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None, use_grl=False):
        if use_grl:
            out_class, _ = self.model(x, alpha=0.0)
        else:
            out_class = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(out_class, dim=1).item()
            
        self.model.zero_grad()
        score = out_class[0, class_idx]
        score.backward(retain_graph=True)
        
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (x.shape[-1], x.shape[-2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        pred_prob = torch.softmax(out_class, dim=1)[0, class_idx].item()
        
        return cam, class_idx, pred_prob


def deprocess_image(tensor):
    """ Annule la normalisation ImageNet pour l'affichage RGB """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    return np.uint8(255 * img)


def show_cam_on_image(img, mask):
    """ Superpose la heatmap sur l'image RGB """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[:, :, ::-1] # BGR to RGB
    img = np.float32(img) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# ─────────────────────────────────────────────
# ÉVALUATION PRINCIPALE
# ─────────────────────────────────────────────
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"❌ Modèle introuvable : {model_path}.")
        return
        
    try:
        rel_path = model_path.parent.relative_to(MODELS_DIR)
    except ValueError:
        rel_path = model_path.parent.name
        
    use_grl = "with_grl" in str(model_path)
    eval_str = f"eval_{'_'.join([r.lower() for r in args.val_rivers])}" if args.val_rivers else "eval_all"
    
    eval_run_dir = os.path.join(BASE_DIR, "evaluation_results", "grl", str(rel_path), eval_str)
    os.makedirs(eval_run_dir, exist_ok=True)
    
    from train_grl import load_data_split
    train_data, val_data, domain_map = load_data_split(args.csv, args.train_rivers, args.val_rivers, return_domain_map=True)
    num_classes = 2 # Mode strict binaire
    num_domains = len(domain_map)
    
    model = WaterPollutionGRL(num_domains=num_domains, num_classes=num_classes, backbone=args.backbone, use_grl=use_grl)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    print(f"✅ Modèle GRL chargé avec succès (GRL={use_grl}).")

    # 2. Charger les données de validation
    _, val_transform = get_transforms()
    val_dataset = PollutionDataset(val_data, args.dir, transform=val_transform)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    # 3. Calcul de la Matrice de Confusion
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\n⏳ Évaluation sur le dataset de validation...")
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            if use_grl:
                out_class, _ = model(images, alpha=0.0)
            else:
                out_class = model(images)
            
            probs = F.softmax(out_class, dim=1)
            preds = torch.argmax(out_class, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy()) # Proba d'être pollué
            
    # Extraction des chemins d'images originaux garantis par shuffle=False
    all_names = []
    all_paths = []
    for item in val_dataset.data:
        name = item["name"]
        label = item["label"]
        lbl_name = val_dataset.lbl_names[label]
        path = os.path.join(val_dataset.img_dir, lbl_name, name)
        all_names.append(name)
        all_paths.append(path)
            
    # Rapport
    print("\n" + "="*50)
    print(" RAPPORT DE CLASSIFICATION")
    print("="*50)
    
    unique_classes = sorted(list(set(d["label"] for d in train_data)))
    
    names_dict = {0: "Propre (0)", 1: "Pollué (1)"}
    lbl_dict   = {0: "Propre", 1: "Pollué"}

    t_names = [names_dict.get(c, f"Class {c}") for c in unique_classes]
    t_labels = [lbl_dict.get(c, f"C{c}") for c in unique_classes]
    
    report = classification_report(all_labels, all_preds, labels=unique_classes, target_names=t_names, zero_division=0)
    print(report)

    # Matrice
    cm = confusion_matrix(all_labels, all_preds, labels=unique_classes)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=t_labels, yticklabels=t_labels)
    plt.xlabel('Prédiction')
    plt.ylabel('Vrai Label')
    plt.title(f'Matrice de Confusion ({str(rel_path)})')
    
    cm_path = os.path.join(eval_run_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"💾 Matrice de confusion sauvegardée : {cm_path}")

    # 4. Export JSON Summary (Pour Notebook de comparaison)
    summary = {
        "model_type": "GRL",
        "scope": args.train_scope,
        "grl": use_grl,
        "train_rivers": args.train_rivers if args.train_rivers else "all",
        "val_rivers": args.val_rivers if args.val_rivers else "all",
        "kappa": float(cohen_kappa_score(all_labels, all_preds)),
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "f1": float(f1_score(all_labels, all_preds, average='binary', zero_division=0)),
        "roc_auc": float(roc_auc_score(all_labels, all_probs))
    }
    with open(os.path.join(eval_run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    print(f"📊 Résumé JSON sauvegardé dans : {os.path.join(eval_run_dir, 'summary.json')}")

    # 4. Export CSV des prédictions (Pour App d'Analyse)
    import pandas as pd
    df = pd.DataFrame({
        "Image": all_names,
        "Path": all_paths,
        "True_Label": all_labels,
        "Pred_Label": all_preds,
        "Score": all_probs 
    })
    df.to_csv(os.path.join(eval_run_dir, "predictions.csv"), index=False)
    print(f"📄 Logs de prédictions sauvegardés dans : {os.path.join(eval_run_dir, 'predictions.csv')}")

    # 4. Input Saliency (Importance globale des canaux) - COMMENTÉ POUR GAIN DE TEMPS/MÉMOIRE
    # print("\n🔬 Analyse de l'importance des canaux (Input Gradients)...")
    # channel_importance = np.zeros(3)
    # 
    # for images, labels, _ in val_loader:
    #     images = images.to(device).requires_grad_()
    #     if use_grl:
    #         out_class, _ = model(images, alpha=0.0)
    #     else:
    #         out_class = model(images)
    #     
    #     # Backward sur la vraie classe
    #     for i in range(len(labels)):
    #         score = out_class[i, labels[i]]
    #         model.zero_grad()
    #         score.backward(retain_graph=True)
    #         
    #     grads = images.grad.abs().cpu().numpy() # Shape: (Batch, 3, H, W)
    #     channel_importance += np.sum(grads, axis=(0, 2, 3))
    #     
    # total_imp = np.sum(channel_importance)
    # if total_imp > 0:
    #     imp_percent = (channel_importance / total_imp) * 100
    #     print("   -> Canal 1 (Limon/Boue)  : {:05.2f}% d'importance globale".format(imp_percent[0]))
    #     print("   -> Canal 2 (Ratios NDTI) : {:05.2f}% d'importance globale".format(imp_percent[1]))
    #     print("   -> Canal 3 (Mousse/Iris): {:05.2f}% d'importance globale".format(imp_percent[2]))

    # 5. Grad-CAM Analysis (Sur quelques images)
    print("\n📸 Génération des Heatmaps Grad-CAM spatiales...")
    
    # Identifier la dernière couche de convolution
    if model.backbone_name == 'resnet18':
        target_layer = model.feature_extractor[7]
    elif 'efficientnet_v2' in model.backbone_name:
        # La dernière couche Conv2dNormActivation se trouve à la fin du bloc "features"
        target_layer = model.feature_extractor[0][-1]
    
    cam_extractor = GradCAM(model, target_layer)
    
    import random
    sample_indices = random.sample(range(len(val_dataset)), min(10, len(val_dataset)))
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    plt.suptitle(f"Grad-CAM Heatmaps - Modèle: {str(rel_path)}", fontsize=16)
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        img_tensor, true_lbl, dom_lbl = val_dataset[idx]
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # Original (RGB)
        orig_img = deprocess_image(img_tensor)
        
        # Heatmap
        cam_mask, pred_class, prob = cam_extractor(img_batch, class_idx=None, use_grl=use_grl)
        cam_image = show_cam_on_image(orig_img, cam_mask)
        
        title_color = "green" if pred_class == true_lbl else "red"
        axes[i].imshow(cam_image)
        axes[i].set_title(f"True: {true_lbl} | Pred: {pred_class}\nConf: {prob*100:.1f}%", color=title_color)
        axes[i].axis('off')

    plt.tight_layout()
    cam_path = os.path.join(eval_run_dir, "grad_cam_samples.png")
    plt.savefig(cam_path)
    plt.close()
    
    # Nettoyage mémoire MPS
    if "mps" in str(device):
        torch.mps.empty_cache()
        
    print(f"💾 Exemples Grad-CAM sauvegardés : {cam_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained Baseline/GRL Water Pollution Model")
    parser.add_argument("--model", type=str, required=True, help="Chemin complet vers best_grl_model.pth")
    parser.add_argument("--dir", default=None, help="Dossier contenant les images")
    parser.add_argument("--csv", default=None, help="Fichier CSV")
    parser.add_argument("--batch", type=int, default=8, help="Taille de batch pour l'évaluation (réduire si OOM)")
    
    parser.add_argument("--train_rivers", nargs="+", default=None)
    parser.add_argument("--val_rivers", nargs="+", default=None)
    parser.add_argument('--train_scope', type=str, default="smart_crop",
                        choices=["no_mask", "smart_crop", "smart_crop_tensor", "manual_crop", "manual_crop_tensor"],
                        help="Dossier de données dans data_preprocessed/")
    parser.add_argument('--backbone', type=str, default='efficientnet_v2_m', choices=['resnet18', 'efficientnet_v2_s', 'efficientnet_v2_m'])
    
    args = parser.parse_args()
    
    if args.dir is None:
        args.dir = os.path.join(BASE_DIR, "data_preprocessed", args.train_scope)
    if args.csv is None:
        csv_name = f"dataset_{args.train_scope}.csv"
        args.csv = os.path.join(BASE_DIR, "data_preprocessed", csv_name)
    
    evaluate(args)
