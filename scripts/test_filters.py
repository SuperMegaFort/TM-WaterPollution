"""
==============================================================================
 TESTEUR DE FILTRE RGB — Visualisation des Prétraitements
==============================================================================
 Ce script prend quelques images aléatoires du Ground Truth, applique 
 le masque de la webcam, et montre l'effet du filtre RGB (utilisé pour 
 séparer les taches de pollution du fond).
 
 Vous pouvez modifier les valeurs dans la fonction `apply_custom_filter` 
 ci-dessous pour trouver la meilleure séparation possible.
==============================================================================
"""

import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "ground_truth")
MASK_DIR = os.path.join(BASE_DIR, "mask")
CSV_PATH = os.path.join(DATA_DIR, "ground_truth.csv")
SAVE_DIR = os.path.join(BASE_DIR, "evaluation_results")

# ─────────────────────────────────────────────
# 1. OUTILS MASQUES (Copie de la pipeline)
# ─────────────────────────────────────────────
def load_masks():
    masks = {}
    for name in ["mask_aire.png", "mask_ziplo.png"]:
        path = os.path.join(MASK_DIR, name)
        if os.path.exists(path):
            with Image.open(path) as m:
                img_gray = m.convert("L")
                masks[name.split(".")[0]] = {"image": img_gray.copy(), "bbox": img_gray.getbbox()}
    return masks

def apply_mask(image, img_name, masks):
    m_info = masks.get("mask_ziplo") if "ziplo" in img_name.lower() else masks.get("mask_aire")
    if not m_info: return image
    mask_img = m_info["image"].resize(image.size, Image.Resampling.NEAREST)
    return Image.composite(image, Image.new("RGB", image.size, (0, 0, 0)), mask_img).crop(m_info["bbox"])


# ─────────────────────────────────────────────
# 2. FILTRE RGB À TESTER & MODIFIER
# ─────────────────────────────────────────────
def apply_custom_filter(image_pil):
    """
    Essayez de modifier les seuils ici pour mieux faire ressortir 
    les taches (mousse blanche / coloration brune) dans l'eau !
    """
    img = np.array(image_pil)
    r, g, b = cv2.split(img)
    
    # PARAMÈTRES ACTUELS:
    # 1. Taches brunes (Le Rouge domine le Vert et le Bleu de 20)
    brown_mask = (r > g + 20) & (r > b + 20)
    
    # 2. Taches très sombres (Rivière polluée noire)
    dark_mask  = (r < 60) & (g < 60) & (b < 60)
    
    # NOUVEAU: 3. Taches très blanches (Mousse)
    # L'eau normale n'est jamais parfaitement blanche, la mousse oui.
    white_mask = (r > 200) & (g > 200) & (b > 200)

    # NOUVEAU: 4. Taches bleues claires / laiteuses (ex: peinture ou rejet chimique)
    # L'eau normale est bleu-vert sombre. L'eau bleu clair polluée a G et B très élevés, et R bas.
    # On cherche (B > 150 ou G > 150) avec une différence marquée avec R.
    light_blue_mask = (b > 130) & (g > 130) & (r < 100) & (b > r + 40)

    # Combinaison de toutes les pollutions recherchées
    pollution_mask = brown_mask | dark_mask | white_mask | light_blue_mask
    
    # Création de l'image finale
    filtered = np.zeros_like(img)
    
    # Les zones considérées comme de la pollution gardent leur couleur d'origine (ou sont amplifiées)
    filtered[pollution_mask] = img[pollution_mask]
    
    # L'eau "normale" (bleu/vert) est fortement assombrie pour faire ressortir la tâche
    filtered[~pollution_mask] = img[~pollution_mask] // 4
    
    return Image.fromarray(filtered)


# ─────────────────────────────────────────────
# 3. VISUALISATION MATPLOTLIB
# ─────────────────────────────────────────────
def visualize_filters(num_samples=5):
    os.makedirs(SAVE_DIR, exist_ok=True)
    masks = load_masks()
    
    # Charger des images polluées valides
    polluted_images = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            classe = int(row["Label"])
            # On veut voir l'effet sur les classes polluées (1, 3, 4, 6)
            if classe in [1, 3, 4, 6]:
                path = os.path.join(DATA_DIR, str(classe), row["Nom_Image"])
                if os.path.exists(path):
                    polluted_images.append((row["Nom_Image"], path, classe))
                    
    if len(polluted_images) < num_samples:
        print("Pas assez d'images polluées trouvées.")
        return
        
    samples = random.sample(polluted_images, num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 4 * num_samples))
    plt.suptitle("Test du Filtre de Prétraitement (Original vs Filtré)", fontsize=16)
    
    for i, (name, path, classe) in enumerate(samples):
        img_pil = Image.open(path).convert("RGB")
        
        # 1. Appliquer le masque (isoler l'eau)
        masked_img = apply_mask(img_pil, name, masks)
        
        # 2. Appliquer le filtre custom
        filtered_img = apply_custom_filter(masked_img)
        
        # Affichage Original Masqué
        axes[i, 0].imshow(masked_img)
        axes[i, 0].set_title(f"Original + Masque\n(Image: {name[:12]}... | Classe: {classe})")
        axes[i, 0].axis('off')
        
        # Affichage Filtré
        axes[i, 1].imshow(filtered_img)
        axes[i, 1].set_title(f"Après Filtre\n(Met en évidence Brun/Noir/Blanc)")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "filter_preview.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"✅ Aperçu généré avec succès !")
    print(f"L'image de comparaison est sauvegardée ici : {save_path}")
    print(f"Ouvrez 'scripts/test_filters.py' pour modifier les valeurs de seuil RGB si nécessaire.")


if __name__ == "__main__":
    visualize_filters(5)
