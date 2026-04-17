import os
import argparse
import random
import glob

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Indispensable sur Mac pour éviter le crash GUI après Tkinter
import matplotlib.pyplot as plt
from PIL import Image

# ─────────────────────────────────────────────
# TRANSFORMATION : PollutionFilters
# ─────────────────────────────────────────────
class PollutionFilters(object):
    """
    Transform PyTorch personnalisée : Channel Splitting & Ratio.
    Transforme une image RGB (PIL ou Tensor) en un tenseur 3 canaux mathématiques
    conçus pour annuler le fond de la rivière et isoler la pollution.
    """
    def __init__(self):
        pass

    def __call__(self, img):
        # 1. Convertir l'entrée en tableau Numpy (H, W, 3) valeurs 0-255
        if isinstance(img, torch.Tensor):
            if img.is_floating_point():
                arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                arr = img.permute(1, 2, 0).numpy().astype(np.uint8)
        else:
            arr = np.array(img.convert('RGB'))

        # Split channels en Float pour éviter les underflow/overflow
        R = arr[:, :, 0].astype(np.float32)
        G = arr[:, :, 1].astype(np.float32)
        B = arr[:, :, 2].astype(np.float32)

        # ---------------------------------------------------------------------
        # CANAL 1 : DÉTECTION LIMON/BOUE (Rouge + CLAHE)
        # ---------------------------------------------------------------------
        # Instanciation locale pour éviter les erreurs de pickle (PyTorch DataLoader multiprocessing)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # La terre et la boue ont une forte signature rouge.
        R_uint8 = np.clip(R, 0, 255).astype(np.uint8)
        c1 = clahe.apply(R_uint8)
        c1_float = c1.astype(np.float32) / 255.0

        # ---------------------------------------------------------------------
        # CANAL 2 : DÉTECTION COLORATION (Ratio / NDTI Inverse)
        # ---------------------------------------------------------------------
        # Formule : (Vert - Rouge) / (Vert + Rouge + epsilon)
        # L'eau "propre" reflète nettement plus le bleu et vert que le rouge (Ratio Positif).
        # Une eau polluée par la terre/produits chimiques bruns augmente drastiquement le rouge
        # ce qui fait chuter ou inverser ce ratio (Ratio Négatif).
        # Avantage: cette fraction annule l'intensité globale (les ombres du pont disparaissent !).
        epsilon = 1e-5
        ratio = (G - R) / (G + R + epsilon)
        
        # Ramener l'échelle de [-1, 1] vers [0, 1] pour le réseau
        c2_float = (ratio + 1.0) / 2.0
        c2_float = np.clip(c2_float, 0.0, 1.0)

        # ---------------------------------------------------------------------
        # CANAL 3 : DÉTECTION MOUSSE, ÉCUME ET IRISATION (Sobel + High-Pass)
        # ---------------------------------------------------------------------
        # On calcule d'abord la luminance / Grayscale de l'image
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        
        # 3a. Filtre de Sobel (Magnitude des gradients spatiaux) -> Capture l'Irisation
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        
        max_sobel = np.max(sobel_mag)
        sobel_norm = sobel_mag / (max_sobel + epsilon) if max_sobel > 0 else sobel_mag
        
        # 3b. Seuil de Luminance croisée (Thresholding) -> Capture la Mousse blanche brillante
        lum_norm = gray.astype(np.float32) / 255.0
        
        # On combine la texture nerveuse de l'irisation et la brillance intense de la mousse
        # L'élévation au carré de lum_norm agit comme un "seuil doux" qui punit les zones moyennes
        c3_float = np.clip(sobel_norm + (lum_norm ** 3), 0.0, 1.0)

        # ---------------------------------------------------------------------
        # CONSTRUCTION DU TENSEUR FINAL
        # ---------------------------------------------------------------------
        # Empiler les 3 cartes en (Canaux, Hauteur, Largeur)
        composite = np.stack([c1_float, c2_float, c3_float], axis=0)
        
        return torch.tensor(composite, dtype=torch.float32)


# ─────────────────────────────────────────────
# VISUALISATION TEST
# ─────────────────────────────────────────────
def visualize_filters(image_path):
    img_pil = Image.open(image_path).convert("RGB")
    
    # Redimensionnement optionnel pour voir l'effet du Crop PyTorch avant
    # img_pil = img_pil.resize((256, 256)).crop((16, 16, 240, 240))
    
    # 1. Utiliser le Transform comme dans le DataLoader
    transform_pipeline = transforms.Compose([
        PollutionFilters()
    ])
    
    # 2. Obtenir le Tenseur Faux-RGB (3, H, W)
    tensor_img = transform_pipeline(img_pil)
    
    # 3. Préparer pour Matplotlib (H, W, 3)
    np_img = tensor_img.permute(1, 2, 0).numpy()
    
    c1 = np_img[:, :, 0]
    c2 = np_img[:, :, 1]
    c3 = np_img[:, :, 2]
    
    # Affichage en grille 2 lignes / 3 colonnes pour avoir de très grandes images
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f"Channel Splitting & Ratio -> {os.path.basename(image_path)}", fontsize=20, fontweight="bold")
    
    # Rendre l'axe plat pour faciliter l'indexation, sauf la dernière case (2,3) qu'on laissera vide ou cachée
    ax = axes.flatten()
    
    # A. Image Originale (Haut Gauche)
    ax[0].imshow(np.array(img_pil))
    ax[0].set_title("1. Original RGB", fontsize=14)
    ax[0].axis("off")
    
    # B. Canal 1 : Limon (Bas Gauche)
    ax[3].imshow(c1, cmap="gray")
    ax[3].set_title("3. Canal 1 : Limon (Rouge+CLAHE)", fontsize=14)
    ax[3].axis("off")
    
    # C. Canal 2 : Coloration (Bas Milieu)
    ax[4].imshow(c2, cmap="gray") 
    ax[4].set_title("4. Canal 2 : Ratio (G-R)/(G+R)", fontsize=14)
    ax[4].axis("off")
    
    # D. Canal 3 : Mousse / Irisation (Bas Droite)
    ax[5].imshow(c3, cmap="gray")
    ax[5].set_title("5. Canal 3 : Texture/Sobel + Mousse/Lum", fontsize=14)
    ax[5].axis("off")
    
    # E. Composite Faux-RGB à injecter dans le réseau (Haut Milieu)
    ax[1].imshow(np_img)
    ax[1].set_title("2. Composite Final 'RGB' (Tensor Network)", fontsize=14)
    ax[1].axis("off")
    
    # D. Case cachée (Haut Droite)
    ax[2].axis("off")
    
    plt.tight_layout()
    out_name = f"filter_preview_{os.path.basename(image_path)}"
    plt.savefig(out_name)
    print(f"📊 Prévisualisation sauvegardée sous : {out_name}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test du Filtre de Pollution de l'Eau")
    parser.add_argument("--img", type=str, default=None, help="Chemin vers une image spécifique")
    args = parser.parse_args()
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.img and os.path.exists(args.img):
        visualize_filters(args.img)
    else:
        # 3. Ouvrir l'explorateur de fichiers pour que l'utilisateur choisisse manuellement
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw() # Cacher la fenêtre principale vide
        
        # Mettre la fenêtre Tkinter au premier plan (surtout sur Mac)
        root.call('wm', 'attributes', '.', '-topmost', True)
        
        filepath = filedialog.askopenfilename(
            title="SÉLECTIONNEZ L'IMAGE À TESTER",
            initialdir=os.path.join(BASE_DIR, "data_preprocessed_multiclass", "no_mask"),
            filetypes=[("Images JPEG/PNG", "*.jpg *.jpeg *.png"), ("Tous", "*.*")]
        )
        
        root.destroy()  # Détruire explicitement la fenêtre pour éviter un conflit avec Matplotlib
        
        if filepath:
            visualize_filters(filepath)
        else:
            print("Aucune image sélectionnée. Fermeture.")
