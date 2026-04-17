"""
==============================================================================
 WATER POLLUTION DETECTION — Unsupervised Domain Adaptation (GRL)
==============================================================================
 Entraîne un modèle PyTorch basé sur un Gradient Reversal Layer pour
 minimiser l'impact du background (domaine: cours d'eau/date).

 Paramètres par défaut :
   - Dataset : data_preprocessed/with_mask_and_filter
   - Backbone: ResNet18
   - Split   : 80% Train, 20% Val (stratifié par classes et domaines)
==============================================================================
"""

import os
import csv
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import StratifiedKFold
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score, roc_auc_score

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "grl")

LABEL_NAMES_BIN = {0: "clean", 1: "polluted"}
# LABEL_NAMES_MULTI = {0: "0_Propre", 1: "1_Coloration", 2: "2_Limon", 3: "3_Mousse"}

# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────
class PollutionDataset(Dataset):
    def __init__(self, csv_data, img_dir, transform=None):
        """
        csv_data : liste de dicts -> [{'path': str, 'label': int, 'domain': int}, ...]
        img_dir  : dossier de base (contient clean/polluted ou 0_Propre...)
        """
        self.data = csv_data
        self.img_dir = img_dir
        self.transform = transform
        
        # Déduction du mode binaire/multiclasse
        unique_labels = set(d["label"] for d in csv_data)
        self.num_classes = len(unique_labels)
        self.lbl_names = LABEL_NAMES_BIN if self.num_classes == 2 else LABEL_NAMES_MULTI


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        name = item["name"]
        label = item["label"]
        domain = item["domain"]
        
        path = os.path.join(self.img_dir, self.lbl_names[label], name)
        
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long), torch.tensor(domain, dtype=torch.long)


# ─────────────────────────────────────────────
# 2. GRADIENT REVERSAL LAYER (GRL)
# ─────────────────────────────────────────────
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0):
        super(GradientReversalLayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


# ─────────────────────────────────────────────
# 3. MODÈLE
# ─────────────────────────────────────────────
class WaterPollutionGRL(nn.Module):
    def __init__(self, num_domains, num_classes=2, backbone='efficientnet_v2_m', dropout=0.5, use_grl=False):
        super().__init__()
        self.use_grl = use_grl
        self.backbone_name = backbone
        
        if backbone == 'resnet50':
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            feature_dim = base_model.fc.in_features
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
        elif backbone == 'efficientnet_v2_m':
            base_model = models.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights.DEFAULT)
            feature_dim = base_model.classifier[1].in_features
            self.feature_extractor = nn.Sequential(base_model.features, base_model.avgpool)
        elif backbone == 'densenet121':
            base_model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            feature_dim = base_model.classifier.in_features
            self.feature_extractor = nn.Sequential(base_model.features, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)))
        elif backbone == 'convnext_tiny':
            base_model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            feature_dim = base_model.classifier[2].in_features
            self.feature_extractor = nn.Sequential(base_model.features, base_model.avgpool)
        else:
            raise NotImplementedError("Backbone non supporté.")

        self.class_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        if self.use_grl:
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(alpha=1.0),
                nn.Flatten(),
                nn.Linear(feature_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.Dropout(dropout),
                nn.Linear(128, num_domains)
            )

    def forward(self, x, alpha=None):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        class_output = self.class_classifier(features)
        
        if self.use_grl:
            if alpha is not None:
                self.domain_classifier[0].alpha = alpha
            domain_output = self.domain_classifier(features)
            return class_output, domain_output
            
        return class_output


# ─────────────────────────────────────────────
# 4. UTILITAIRES DE DONNÉES
# ─────────────────────────────────────────────
def extract_river_name(filename):
    base = filename.split('.')[0]
    parts = base.split('_')
    return parts[-1].lower() if len(parts) > 1 else "unknown"

def load_data_split(csv_path, train_rivers=None, val_rivers=None, return_domain_map=False):
    """ Filtre le dataset par nom de rivière ou Stratified Split si non spécifié """
    data = []
    unique_rivers = set()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_label = int(row["Label"])
            r_name = extract_river_name(row["Nom_Image"])
            unique_rivers.add(r_name)
            
            data.append({
                "name": row["Nom_Image"],
                "label": 0 if raw_label == 0 else 1,
                "river": r_name
            })
            
    domain_map = {r: i for i, r in enumerate(sorted(unique_rivers))}
    for d in data:
        d["domain"] = domain_map[d["river"]]
            
    if train_rivers or val_rivers:
        train_data = []
        val_data = []
        
        t_list = [r.lower() for r in train_rivers] if train_rivers else []
        v_list = [r.lower() for r in val_rivers] if val_rivers else []
        
        for d in data:
            in_train = any(r in d["river"] for r in t_list)
            in_val   = any(r in d["river"] for r in v_list)
            
            if in_train:
                train_data.append(d)
            elif in_val:
                val_data.append(d)
            else:
                if v_list and not t_list:
                    train_data.append(d)
                elif t_list and not v_list:
                    val_data.append(d)
                    
        print(f"[SPLIT] Train : {train_rivers} | Val : {val_rivers}")
        
        if len(train_data) == 0 or len(val_data) == 0:
            print("[WARNING] Fallback vers split aléatoire.")
            return load_data_split(csv_path, None, None, return_domain_map)
            
        return (train_data, val_data, domain_map) if return_domain_map else (train_data, val_data)
    else:
        labels = [d["label"] for d in data]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(data, labels))
        
        train_data = [data[i] for i in train_idx]
        val_data   = [data[i] for i in val_idx]
        
        return (train_data, val_data, domain_map) if return_domain_map else (train_data, val_data)


def get_transforms(scope="smart_crop"):
    """Augmentation On-The-Fly : Géométrie et Lumières strictes"""
    
    # Si on utilise un Super-Tenseur (smart ou manual), on fait une normalisation neutre (centrée sur 0.5)
    # On désactive aussi le ColorJitter qui détruirait le NDTI
    if "_tensor" in scope:
        train_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return train_transform, val_transform

    # Sinon, traitement RGB classique (ImageNet + ColorJitter)
    train_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


# ─────────────────────────────────────────────
# 5. BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────
def train_model(args):
    # Création dynamique des dossiers de sauvegarde
    mask_status = args.train_scope
    grl_status  = "with_grl" if args.use_grl else "no_grl"
    rivers_str  = f"train_{'_'.join([r.lower() for r in args.train_rivers])}" if args.train_rivers else "train_all"
    freeze_str  = "freeze" if args.freeze_backbone else "unfreeze"
    
    hyper_str = f"{args.backbone}_lr_{args.lr}_drp_{args.dropout}_{freeze_str}"
    
    save_dir = os.path.join(MODELS_DIR, mask_status, grl_status, rivers_str, hyper_str)
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[INFO] Device utilisé : {device}")
    print(f"[INFO] Sauvegarde GRL dans : {save_dir}")

    # 1. Setup données
    train_data, val_data, domain_map = load_data_split(args.csv, args.train_rivers, args.val_rivers, return_domain_map=True)
    print(f"[INFO] Train split: {len(train_data)} images | Val split: {len(val_data)} images")
    
    num_classes = 2 # Force le binaire
    num_domains = len(domain_map)
    print(f"[INFO] Classes cibles: {num_classes} | Domaines identifiés: {num_domains}")

    
    train_transform, val_transform = get_transforms(args.train_scope)
    train_dataset = PollutionDataset(train_data, args.dir, transform=train_transform)
    val_dataset   = PollutionDataset(val_data, args.dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)
    
    # 2. Setup Modèle

    model = WaterPollutionGRL(num_domains=num_domains, num_classes=num_classes, backbone=args.backbone, dropout=args.dropout, use_grl=args.use_grl).to(device)
    
    # --- OPTION: GEL DU BACKBONE ---
    if args.freeze_backbone:
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
    # -------------------------------
    # # -------------------------------
    
    criterion_class  = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss() if args.use_grl else None
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    # 3. Training Loop
    best_f1 = 0.0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*40}\nÉpoque {epoch}/{args.epochs}\n{'='*40}")
        
        # ── TRAIN PHASE ──
        model.train()
        train_loss, train_class_loss, train_domain_loss = 0.0, 0.0, 0.0
        
        # Adaptation du GRL alpha (monte de 0 à 1 selon la progression de l'entraînement)
        p = float(epoch - 1) / args.epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        pbar = tqdm(train_loader, desc="Train", dynamic_ncols=True, leave=False)
        for images, labels, domains in pbar:
            images, labels, domains = images.to(device), labels.to(device), domains.to(device)
            
            optimizer.zero_grad()
            
            if args.use_grl:
                class_preds, domain_preds = model(images, alpha=alpha)
                loss_c = criterion_class(class_preds, labels)
                loss_d = criterion_domain(domain_preds, domains)
                loss = loss_c + loss_d
                train_domain_loss += loss_d.item()
            else:
                class_preds = model(images)
                loss_c = criterion_class(class_preds, labels)
                loss = loss_c
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_class_loss += loss_c.item()
            
            if args.use_grl:
                pbar.set_postfix({'CELoss': f"{loss_c.item():.4f}", 'DomLoss': f"{loss_d.item():.4f}"})
            else:
                pbar.set_postfix({'CELoss': f"{loss.item():.4f}"})
                
            # LIBÉRATION MÉMOIRE (MPS M1/M2/M3)
            del images, labels, domains, class_preds, loss, loss_c
            if args.use_grl:
                del domain_preds, loss_d
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # ── VAL PHASE ──
        model.eval()
        val_loss, val_class_loss, val_domain_loss = 0.0, 0.0, 0.0
        
        all_class_preds, all_class_labels = [], []
        all_class_probs = []
        all_domain_preds, all_domain_labels = [], []

        with torch.no_grad():
            for images, labels, domains in tqdm(val_loader, desc="Valid", dynamic_ncols=True, leave=False):
                images, labels, domains = images.to(device), labels.to(device), domains.to(device)
                
                if args.use_grl:
                    class_preds, domain_preds = model(images, alpha=alpha)
                    loss_c = criterion_class(class_preds, labels)
                    loss_d = criterion_domain(domain_preds, domains)
                    loss = loss_c + loss_d
                    val_domain_loss += loss_d.item()
                    
                    all_domain_preds.extend(torch.argmax(domain_preds, dim=1).cpu().numpy())
                    all_domain_labels.extend(domains.cpu().numpy())
                else:
                    class_preds = model(images)
                    loss_c = criterion_class(class_preds, labels)
                    loss = loss_c
                
                val_loss += loss.item()
                val_class_loss += loss_c.item()
                
                # Accuracy tracking
                all_class_preds.extend(torch.argmax(class_preds, dim=1).cpu().numpy())
                all_class_labels.extend(labels.cpu().numpy())
                all_class_probs.extend(torch.softmax(class_preds, dim=1)[:, 1].cpu().numpy())
                
                # LIBÉRATION MÉMOIRE
                del images, labels, domains, class_preds, loss, loss_c
                if args.use_grl:
                    del domain_preds, loss_d
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        # ── METRIQUES ──
        v_acc_class = (np.array(all_class_preds) == np.array(all_class_labels)).mean()
        v_kappa_class = cohen_kappa_score(all_class_labels, all_class_preds)
        try:
            v_roc_auc = roc_auc_score(all_class_labels, all_class_probs)
        except Exception:
            v_roc_auc = 0.5
        
        print("\n[RÉSULTATS DE VALIDATION]")
        print(f" Loss Classe  : {val_class_loss/len(val_loader):.4f}  | Précision Classe  : {v_acc_class*100:.2f}% (Kappa: {v_kappa_class:.4f}, ROC-AUC: {v_roc_auc:.4f})")
        
        if args.use_grl:
            v_acc_domain = (np.array(all_domain_preds) == np.array(all_domain_labels)).mean()
            print(f" Loss Domaine : {val_domain_loss/len(val_loader):.4f}  | Précision Domaine : {v_acc_domain*100:.2f}% (ciblée vers {100/num_domains:.2f}%)")
            print(f" Si Précision Domaine = ~{100/num_domains:.2f}%, le GRL fonctionne parfaitement !")

        scheduler.step(v_kappa_class)

        # ── SAUVEGARDE ──
        if v_kappa_class > best_f1:
            best_f1 = v_kappa_class
            
            torch.save(model.state_dict(), os.path.join(save_dir, "best_grl_model.pth"))
            print(f" ⭐ Nouveau meilleur modèle sauvegardé ! (Kappa = {best_f1:.4f})")
            
            with open(os.path.join(save_dir, "train_summary.txt"), "w") as f:
                f.write(f"Meilleur Kappa Validation : {best_f1:.4f}\n")
                f.write(f"Meilleure ROC-AUC Validation : {v_roc_auc:.4f}\n")
                f.write(f"Scope: {args.train_scope}\n")
                f.write(f"GRL: {args.use_grl}\n")
                f.write(f"Train Rivers: {args.train_rivers}\n")
                f.write(f"Val Rivers: {args.val_rivers}\n")
                f.write(f"Backbone: {args.backbone}\n")
                f.write(f"Dropout: {args.dropout}\n")
                f.write(f"Freeze Backbone: {args.freeze_backbone}\n")
                f.write(f"Learning Rate: {args.lr}\n")

    print(f"\n[Terminé] Meilleur F1 Score atteint : {best_f1:.4f}")
    print(f"💾 Modèle et stats sauvegardés dans : {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline/GRL Water Pollution Model")
    parser.add_argument("--dir",   default=None, help="Dossier contenant les images")
    parser.add_argument("--csv",   default=None, help="Fichier CSV de métadonnées")
    parser.add_argument("--batch", type=int, default=8, help="Taille de batch")
    parser.add_argument("--epochs",type=int, default=10, help="Nombre d'époques")
    parser.add_argument("--lr",    type=float, default=1e-4, help="Learning rate")
    
    parser.add_argument("--use_grl", action="store_true", help="Activer le Gradient Reversal Layer")
    parser.add_argument('--train_scope', type=str, default="smart_crop",
                        choices=["no_mask", "smart_crop", "smart_crop_tensor", "manual_crop", "manual_crop_tensor"],
                        help="Dossier de données dans data_preprocessed/")
                        
    parser.add_argument("--backbone", type=str, default="efficientnet_v2_m", choices=["resnet50", "efficientnet_v2_m", "densenet121", "convnext_tiny"], help="Architecture d'extraction")
    parser.add_argument("--dropout", type=float, default=0.5, help="Taux d'oubli")
    parser.add_argument("--freeze_backbone", action="store_true", help="Geler les poids du backbone pré-entraîné")
    
    parser.add_argument("--train_rivers", nargs="+", default=None, help="Liste de rivières pour l'entraînement (ex: Ziplo Aire)")
    parser.add_argument("--val_rivers", nargs="+", default=None, help="Liste de rivières pour la validation (ex: Vuillonnex)")
    
    args = parser.parse_args()
    
    if args.dir is None:
        args.dir = os.path.join(BASE_DIR, "data_preprocessed", args.train_scope)
    if args.csv is None:
        csv_name = f"dataset_{args.train_scope}.csv"
        args.csv = os.path.join(BASE_DIR, "data_preprocessed", csv_name)

    train_model(args)
