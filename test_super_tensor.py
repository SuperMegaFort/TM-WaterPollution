
import os
import cv2
import numpy as np
from PIL import Image

def create_super_tensor(img_pil):
    img_rgb = np.array(img_pil)
    
    # --- CANAL 1 : Le Relief (CLAHE sur Niveaux de gris) ---
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    canal_1 = clahe.apply(gray)
    
    # --- CANAL 2 : La Chimie (NDTI Normalisé) ---
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    ndti_math = (R - G) / (R + G + 1e-7)
    canal_2 = cv2.normalize(ndti_math, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    canal_2 = canal_2.astype(np.uint8)
    
    # --- CANAL 3 : L'Intensité (Saturation pure) ---
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    canal_3 = hsv[:, :, 1]
    
    # --- ASSEMBLAGE ---
    super_tensor_np = np.dstack((canal_1, canal_2, canal_3))
    return Image.fromarray(super_tensor_np)

# Test with a dummy image
img = Image.new('RGB', (224, 224), color = 'red')
try:
    res = create_super_tensor(img)
    print("Success!")
    res.save("test_tensor.png")
    print("Saved test_tensor.png")
except Exception as e:
    print(f"Failed: {e}")
