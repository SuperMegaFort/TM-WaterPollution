import cv2
import numpy as np
import matplotlib.pyplot as plt

def appliquer_masque(image_path, mask_path):
    # 1. Charger l'image originale et le masque
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # OpenCV lit en BGR, on passe en RGB
    
    # Le masque doit être lu en niveaux de gris (Noir et Blanc)
    masque = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 2. S'assurer que le masque fait exactement la même taille que l'image
    if img.shape[:2] != masque.shape:
        masque = cv2.resize(masque, (img.shape[1], img.shape[0]))

    # 3. Binariser le masque (sécurité pour s'assurer qu'il n'y a que du 0 et du 255)
    _, masque_binaire = cv2.threshold(masque, 127, 255, cv2.THRESH_BINARY)

    # 4. Appliquer le masque ! 
    # cv2.bitwise_and garde les pixels de l'image uniquement là où le masque est blanc
    img_isolee = cv2.bitwise_and(img_rgb, img_rgb, mask=masque_binaire)

    # 5. Affichage pour le test
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("1. Image Originale")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(masque_binaire, cmap='gray')
    plt.title("2. Ton Masque (Rivière en blanc)")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_isolee)
    plt.title("3. Résultat pour l'IA")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- TEST ---
# Teste avec l'image que tu m'as envoyée et un masque noir/blanc que tu as dessiné
appliquer_masque('dataset/train/0/08122025_144500_RCNX0040_Ziplo.jpg', 'mask/mask_ziplo.png')