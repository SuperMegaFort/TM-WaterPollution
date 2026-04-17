"""
==============================================================================
 WATER POLLUTION DETECTION — Siamese / Triplet Network
==============================================================================
 Entraîne un extracteur de caractéristiques (Metric Learning) utilisant 
 la Triplet Loss. L'espace latent permet ensuite de détecter les anomalies 
 par rapport à une image de référence (Few-Shot/Zero-Shot Domain Adaptation).
==============================================================================
"""

import os
import csv
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models", "triplet")

# DATASET_DEFAULT = os.path.join(BASE_DIR, "ground_truth")
# CSV_DEFAULT     = os.path.join(BASE_DIR, "ground_truth", "ground_truth.csv")
# MODELS_DIR      = os.path.join(BASE_DIR, "models", "triplet")

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
        
class ReferenceTripletDataset(Dataset):
    def __init__(self, data, img_dir, transform=None):
        self.data = data
        self.img_dir = img_dir
        self.transform = transform
        
        # 1. Organiser les données par rivière et par label (0=Propre, 1=Pollué)
        self.river_data = {}
        for item in data:
            r = item['river']
            lbl = item['label']
            if r not in self.river_data:
                self.river_data[r] = {0: [], 1: []}
            self.river_data[r][lbl].append(item)
            
        # 2. Définir l'Ancre de Référence (Reference Anchor) pour chaque rivière
        self.references = {}
        for r, splits in self.river_data.items():
            # On prend la toute première image propre de la rivière comme référence absolue
            if len(splits[0]) > 0:
                self.references[r] = splits[0][0]
            else:
                self.references[r] = None # Sécurité si une rivière n'a aucune image propre
                
        # On ne garde dans le dataset itérable que les images des rivières qui ont une référence
        self.valid_data = [d for d in self.data if self.references.get(d['river']) is not None]

    def __len__(self):
        return len(self.valid_data)
        
    def _load_image(self, item):
        folder = "clean" if item["label"] == 0 else "polluted"
        # Adapte "clean/polluted" selon l'arborescence réelle de tes dossiers pré-traités
        path = os.path.join(self.img_dir, folder, item["name"])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        # L'image actuelle que l'on veut évaluer
        current_item = self.valid_data[idx]
        river = current_item["river"]
        
        # L'ANCRE (A) : Toujours l'image de référence propre de cette même rivière
        ref_item = self.references[river]
        anchor_img = self._load_image(ref_item)
        
        # 1. Définition du Positif et du Négatif
        if current_item["label"] == 0:
            # L'image actuelle est propre -> devient le POSITIF (P)
            pos_item = current_item
            
            # On cherche un NÉGATIF (N) : une image polluée de la même rivière
            polluted_candidates = self.river_data[river][1]
            if len(polluted_candidates) > 0:
                neg_item = random.choice(polluted_candidates)
            else:
                # Fallback : pollution d'une autre rivière si aucune disponible ici
                all_polluted = [d for d in self.data if d["label"] == 1]
                neg_item = random.choice(all_polluted)
        else:
            # L'image actuelle est polluée -> devient le NÉGATIF (N)
            neg_item = current_item
            
            # On cherche un POSITIF (P) : une autre image propre de la même rivière
            clean_candidates = self.river_data[river][0]
            if len(clean_candidates) > 1:
                # On évite si possible de reprendre exactement la même image que l'ancre de référence
                pos_candidates = [c for c in clean_candidates if c["name"] != ref_item["name"]]
                pos_item = random.choice(pos_candidates) if pos_candidates else ref_item
            else:
                pos_item = ref_item

        # 2. Chargement effectif des images
        positive_img = self._load_image(pos_item)
        negative_img = self._load_image(neg_item)

        # 3. Retour Triplet + Labels de domaine pour GRL
        return (
            anchor_img, positive_img, negative_img, 
            torch.tensor(current_item["label"], dtype=torch.float),
            torch.tensor(ref_item["domain_id"], dtype=torch.long),
            torch.tensor(pos_item["domain_id"], dtype=torch.long),
            torch.tensor(neg_item["domain_id"], dtype=torch.long)
        )

class WaterPollutionSiamese(nn.Module):
    def __init__(self, backbone='efficientnet_v2_s', embedding_dim=128, use_grl=False, num_domains=2):
        super().__init__()
        self.use_grl = use_grl
        
        self.backbone_name = backbone
        if backbone == 'efficientnet_v2_s':
            base_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
            feature_dim = base_model.classifier[1].in_features
            
            # Extraction des caractéristiques purement spatiales
            self.feature_extractor = nn.Sequential(
                base_model.features,
                base_model.avgpool
            )
        else:
            raise NotImplementedError("Backbone non supporté.")

        # Tête de projection vers l'Espace Latent Mathématique
        self.embedder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )
        
        # Tête de Domaine (GRL) optionnelle pour détruire l'environnement
        if self.use_grl:
            self.domain_classifier = nn.Sequential(
                GradientReversalLayer(alpha=1.0),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(True),
                nn.Dropout(0.5),
                nn.Linear(64, num_domains)
            )

    def forward(self, x, alpha=None):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        embeddings = self.embedder(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        
        if self.use_grl:
            if alpha is not None:
                self.domain_classifier[0].alpha = alpha
            # 🎯 GRL appliqué STRICTEMENT sur l'embedding final (128d) pour empêcher la triche
            domains = self.domain_classifier(embeddings)
            return embeddings, domains
        
        return embeddings


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
                "label": 0 if raw_label == 0 else 1, # Projection purement binaire pour Siamese
                "original_label": raw_label,         # Pour folder mapping
                "river": r_name
            })
            
    # GRL Domain ID Mapping 
    domain_map = {r: i for i, r in enumerate(sorted(unique_rivers))}
    for d in data:
        d["domain_id"] = domain_map[d["river"]]
            
    if train_rivers or val_rivers:
        train_data, val_data = [], []
        t_list = [r.lower() for r in train_rivers] if train_rivers else []
        v_list = [r.lower() for r in val_rivers] if val_rivers else []
        
        for d in data:
            name_lower = d["name"].lower()
            if any(r in name_lower for r in t_list):
                train_data.append(d)
            elif any(r in name_lower for r in v_list):
                val_data.append(d)
            else:
                if v_list and not t_list:
                    train_data.append(d)
                elif t_list and not v_list:
                    val_data.append(d)
        
        print(f"[SPLIT MANUEL] Rivières Train : {train_rivers if train_rivers else 'TOUT LE RESTE'}")
        print(f"[SPLIT MANUEL] Rivières Val   : {val_rivers if val_rivers else 'TOUT LE RESTE'}")
        
        if len(train_data) == 0 or len(val_data) == 0:
            print("[WARNING] 0 images trouvées ! Fallback vers split.")
            return load_data_split(csv_path, None, None, return_domain_map)
            
        return (train_data, val_data, domain_map) if return_domain_map else (train_data, val_data)
    else:
        labels = [d["label"] for d in data]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf.split(data, labels))
        
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        
        return (train_data, val_data, domain_map) if return_domain_map else (train_data, val_data)


def get_transforms(is_tensor=False):
    """Augmentation On-The-Fly : Géométrie et Lumières strictes (Sans color distortion)"""
    if is_tensor:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    else:
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return train_transform, val_transform


def train_triplet(args):
    # Création dynamique des dossiers de sauvegarde
    mask_status = args.train_scope
    grl_status  = "with_grl" if args.use_grl else "no_grl"
    rivers_str  = f"train_{'_'.join([r.lower() for r in args.train_rivers])}" if args.train_rivers else "train_all"
    
    hyper_str = f"dim_{args.latent_dim}_lr_{args.lr}_m_{args.margin}"
    
    save_dir = os.path.join(MODELS_DIR, mask_status, grl_status, rivers_str, hyper_str)
    os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[INFO] Device utilisé : {device}")
    print(f"[INFO] Sauvegarde du modèle dans : {save_dir}")

    # 1. Dataset & Loader
    train_data, val_data, domain_map = load_data_split(args.csv, args.train_rivers, args.val_rivers, return_domain_map=True)
    
    num_domains = len(domain_map)
    print(f"[INFO] Train split: {len(train_data)} images | Val split: {len(val_data)} images")
    print(f"[INFO] Domaines GRL détectés ({num_domains}) : {domain_map}")
    
    is_tensor = "tensor" in args.train_scope
    train_t, val_t = get_transforms(is_tensor=is_tensor)
    train_dataset = ReferenceTripletDataset(train_data, args.dir, transform=train_t)
    val_dataset   = ReferenceTripletDataset(val_data, args.dir, transform=val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=0)

    # 2. Modèle & Optimisation
    model = WaterPollutionSiamese(backbone='efficientnet_v2_s', embedding_dim=args.latent_dim, use_grl=args.use_grl, num_domains=num_domains).to(device)
    
    # # --- AJOUT : GEL DU BACKBONE ---
    # for param in model.feature_extractor.parameters():
    #     param.requires_grad = False
    # # -------------------------------
    
    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2)
    criterion_domain = nn.CrossEntropyLoss() if args.use_grl else None
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)

    best_roc_auc = 0.0

    # 3. Boucle d'entraînement
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*40}\nÉpoque {epoch}/{args.epochs}\n{'='*40}")
        
        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        
        # Adaptation de l'Alpha GRL (de 0 vers 1)
        p = float(epoch - 1) / args.epochs
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        
        pbar = tqdm(train_loader, desc=f"Train", ncols=90)
        
        for anchor, positive, negative, _, dom_a, dom_p, dom_n in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            dom_a, dom_p, dom_n = dom_a.to(device), dom_p.to(device), dom_n.to(device)
            
            optimizer.zero_grad()
            
            # Concaténer pour optimiser la passe MPS
            combined = torch.cat([anchor, positive, negative], dim=0)
            
            if args.use_grl:
                embeddings, domains_pred = model(combined, alpha=alpha)
            else:
                embeddings = model(combined)
                
            emb_a, emb_p, emb_n = torch.split(embeddings, anchor.size(0))
            
            loss_t = criterion_triplet(emb_a, emb_p, emb_n)
            
            if args.use_grl:
                domains_true = torch.cat([dom_a, dom_p, dom_n], dim=0)
                loss_d = criterion_domain(domains_pred, domains_true)
                loss = loss_t + (0.05 * loss_d)
            else:
                loss = loss_t
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if args.use_grl:
                pbar.set_postfix({'Tri_L': f"{loss_t.item():.2f}", 'Dom_L': f"{loss_d.item():.2f}"})
            else:
                pbar.set_postfix({'Loss (Triplet)': f"{loss.item():.4f}"})
            
            # Libération agressive de la RAM (indispensable pour les Mac MPS avec des tenseurs concaténés)
            del anchor, positive, negative, combined, embeddings, emb_a, emb_p, emb_n, loss
            if args.use_grl:
                del domains_pred, domains_true
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        # --- VALIDATION Proxy via ROC-AUC ---
        model.eval()
        val_loss = 0.0
        all_true_labels = [] 
        all_distances = []
        
        with torch.no_grad():
            for anchor, positive, negative, _, dom_a, dom_p, dom_n in tqdm(val_loader, desc="Valid", ncols=90):
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                
                # Concaténer pour optimiser l'inférence
                combined = torch.cat([anchor, positive, negative], dim=0)
                
                if args.use_grl:
                    embeddings, _ = model(combined, alpha=alpha)
                else:
                    embeddings = model(combined)
                    
                emb_a, emb_p, emb_n = torch.split(embeddings, anchor.size(0))
                
                loss = criterion_triplet(emb_a, emb_p, emb_n)
                val_loss += loss.item()
                
                # Validation Proxy Metric: 
                # La distance entre (Anchor, Positive) devrait être petite (Normal/Paire identique)
                # La distance entre (Anchor, Negative) devrait être grande (Anomalie/Paire différente)
                dist_p = F.pairwise_distance(emb_a, emb_p)
                dist_n = F.pairwise_distance(emb_a, emb_n)
                
                # 0 = Paire identique (devrait avoir une petite distance)
                # 1 = Paire d'anomalie/différente (devrait avoir une grande distance)
                all_distances.extend(dist_p.cpu().numpy())
                all_true_labels.extend([0] * len(dist_p))
                
                all_distances.extend(dist_n.cpu().numpy())
                all_true_labels.extend([1] * len(dist_n))
                
                # Libération de la mémoire
                del anchor, positive, negative, combined, embeddings, emb_a, emb_p, emb_n, loss, dist_p, dist_n
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        # Calculer le ROC-AUC de validation global (plus robuste aux petits datasets)
        try:
            val_roc_auc = roc_auc_score(all_true_labels, all_distances)
        except ValueError:
            # S'il manque des classes
            val_roc_auc = 0.5

        print(f"\n[RÉSULTATS DE VALIDATION]")
        print(f" Loss Moyenne  : {val_loss/len(val_loader):.4f}")
        print(f" Score ROC-AUC  : {val_roc_auc:.4f}  (1.0 = Séparation Parfaite)")

        scheduler.step(val_roc_auc)

        # Sauvegarde
        if val_roc_auc > best_roc_auc:
            best_roc_auc = val_roc_auc
            t_r = sorted([r.lower() for r in args.train_rivers]) if args.train_rivers else []
            
            torch.save(model.state_dict(), os.path.join(save_dir, "best_siamese_model.pth"))
            print(f" ⭐ Nouveau meilleur modèle sauvegardé ! L'espace latent converge (ROC-AUC: {best_roc_auc:.4f})")
            
            # Enregistrer les stats
            with open(os.path.join(save_dir, "train_summary.txt"), "w") as f:
                f.write(f"Meilleure ROC-AUC Validation : {best_roc_auc:.4f}\n")
                f.write(f"Scope: {args.train_scope}\n")
                f.write(f"GRL: {args.use_grl}\n")
                f.write(f"Train Rivers: {args.train_rivers}\n")
                f.write(f"Val Rivers: {args.val_rivers}\n")
                f.write(f"Latent Dim: {args.latent_dim}\n")
                f.write(f"Learning Rate: {args.lr}\n")
                f.write(f"Margin: {args.margin}\n")

    print(f"\n[Terminé] Meilleure séparation ROC-AUC des clusters : {best_roc_auc:.4f}")
    print(f"💾 Modèle et stats sauvegardés dans : {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese/Triplet Water Pollution Model")
    parser.add_argument("--dir",   default=None, help="Dossier contenant les images (auto si --train_scope utilisé)")
    parser.add_argument("--csv",   default=None, help="Fichier CSV (auto si --train_scope utilisé)")
    parser.add_argument('--train_scope', type=str, default="manual_crop",
                        choices=["no_mask", "manual_crop", "manual_crop_tensor"],
                        help="Dossier de données dans data_preprocessed/")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs",type=int, default=10, help="Nombre d'époques")
    parser.add_argument("--lr",    type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--use_grl", action="store_true", help="Activer le Gradient Reversal Layer pour détruire l'info domaine de l'espace latent")
    
    parser.add_argument("--latent_dim", type=int, default=128, help="Dimension de l'espace latent (ex: 1, 16, 64)")
    parser.add_argument("--margin", type=float, default=0.2, help="Marge de la Triplet Loss")
    
    parser.add_argument("--train_rivers", nargs="+", default=None, help="Rivières pour l'entraînement (ex: Ziplo Arve)")
    parser.add_argument("--val_rivers", nargs="+", default=None, help="Rivières pour la validation (ex: Aire)")
    
    args = parser.parse_args()
    
    if args.dir is None:
        args.dir = os.path.join(BASE_DIR, "data_preprocessed", args.train_scope)
    if args.csv is None:
        csv_name = f"dataset_{args.train_scope}.csv"
        args.csv = os.path.join(BASE_DIR, "data_preprocessed", csv_name)
        
    train_triplet(args)
