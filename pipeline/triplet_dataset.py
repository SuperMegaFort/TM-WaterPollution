import os
import csv
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class TripletDataset(Dataset):
    def __init__(self, csv_data, img_dir, transform=None):
        """
        Génère des Triplets pour l'apprentissage de métrique (Siamese Network).
        csv_data : liste de dicts -> [{'name': str, 'label': int, 'domain': str}, ...]
        """
        self.data = csv_data
        self.img_dir = img_dir
        self.transform = transform
        
        # Binaire uniquement (0: Clean, 1: Polluted)
        self.lbl_names = {0: "clean", 1: "polluted"}
        
        # Mapping des indices par classe pour piocher les Positifs et Négatifs rapidement
        self.indices_by_label = {0: [], 1: []}
        for idx, item in enumerate(self.data):
            lbl = item["label"]
            if lbl in self.indices_by_label:
                self.indices_by_label[lbl].append(idx)

    def __len__(self):
        return len(self.data)

    def _load_image(self, item):
        # Fallback dynamique : lit le raw ground_truth ("0", "1"...) ou le pre-processed ("clean", "polluted")
        path_raw = os.path.join(self.img_dir, str(item.get("original_label", item["label"])), item["name"])
        path_proc = os.path.join(self.img_dir, self.lbl_names[item["label"]], item["name"])
        
        path = path_raw if os.path.exists(path_raw) else path_proc
        
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

    def __getitem__(self, index):
        # 1. Anchor
        anchor_item = self.data[index]
        anchor_label = anchor_item["label"]
        anchor_domain = anchor_item.get("domain_id", 0)
        
        # 2. Positive (HARD MINING : Cross-Domain exigé)
        # On cherche un Positif qui a un "domain_id" strictement différent de l'Ancre
        pool = self.indices_by_label[anchor_label]
        cross_domain_pool = [i for i in pool if self.data[i].get("domain_id", 0) != anchor_domain]
        
        if len(cross_domain_pool) > 0:
            positive_idx = random.choice(cross_domain_pool)
        else:
            # Fallback si on ne s'entraîne que sur 1 seule rivière
            positive_idx = random.choice(pool)
            while positive_idx == index and len(pool) > 1:
                positive_idx = random.choice(pool)
                
        positive_item = self.data[positive_idx]
        
        # 3. Negative
        # On choisit une image de la classe DIFFÉRENTE
        negative_label = 1 if anchor_label == 0 else 0
        
        # S'il n'y a pas de classe opposée dans le dataset de validation (cas extrême), 
        # on retourne l'anchor par sécurité.
        if len(self.indices_by_label[negative_label]) == 0:
            negative_item = anchor_item
            negative_label = anchor_label
        else:
            negative_idx = random.choice(self.indices_by_label[negative_label])
            negative_item = self.data[negative_idx]

        # Chargement physique des 3 images
        anchor_img = self._load_image(anchor_item)
        positive_img = self._load_image(positive_item)
        negative_img = self._load_image(negative_item)
        
        return (
            anchor_img, positive_img, negative_img, 
            torch.tensor(anchor_label, dtype=torch.long),
            torch.tensor(anchor_item.get("domain_id", 0), dtype=torch.long),
            torch.tensor(positive_item.get("domain_id", 0), dtype=torch.long),
            torch.tensor(negative_item.get("domain_id", 0), dtype=torch.long)
        )
