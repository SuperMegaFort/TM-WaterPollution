import os
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
# Remplace par le chemin vers un dossier contenant quelques images de test (ex: ground_truth/1/)
TEST_IMG_DIR = "ground_truth/1" 
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
WATER_CLASSES = [21, 26, 60, 128]

print(f"Chargement du modèle sur {DEVICE}...")
model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_id)
model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(DEVICE)
model.eval()

# ─────────────────────────────────────────────
# 2. FONCTIONS DU PIPELINE
# ─────────────────────────────────────────────
def get_water_bbox(img_pil):
    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    mask = torch.nn.functional.interpolate(logits, size=img_pil.size[::-1], mode="bilinear", align_corners=False)
    mask = mask.argmax(dim=1)[0]
    
    water_mask = torch.zeros_like(mask, dtype=torch.uint8)
    for cls in WATER_CLASSES:
        water_mask |= (mask == cls).to(torch.uint8)
        
    water_mask_np = (water_mask.cpu().numpy() * 255).astype(np.uint8)
    contours, _ = cv2.findContours(water_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, water_mask_np
        
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 2500:
        return None, water_mask_np
        
    x, y, w, h = cv2.boundingRect(largest_contour)
    return (x, y, x + w, y + h), water_mask_np

def pad_to_square(img_pil):
    width, height = img_pil.size
    if width == height: return img_pil
    size = max(width, height)
    result = Image.new("RGB", (size, size), (0, 0, 0))
    result.paste(img_pil, ((size - width) // 2, (size - height) // 2))
    return result

def create_super_tensor(img_pil):
    img_rgb = np.array(img_pil)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    canal_1 = clahe.apply(gray)
    
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    ndti_math = (R - G) / (R + G + 1e-7)
    canal_2 = cv2.normalize(ndti_math, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    canal_3 = hsv[:, :, 1]
    
    return Image.fromarray(np.dstack((canal_1, canal_2, canal_3)))

# ─────────────────────────────────────────────
# 3. TEST VISUEL SUR 3 IMAGES
# ─────────────────────────────────────────────
# Récupère 3 images au hasard
all_images = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
sample_images = random.sample(all_images, min(3, len(all_images)))

fig, axes = plt.subplots(len(sample_images), 4, figsize=(16, 4 * len(sample_images)))
fig.suptitle("Validation du Pipeline de Prétraitement", fontsize=16)

for i, img_name in enumerate(sample_images):
    img_path = os.path.join(TEST_IMG_DIR, img_name)
    orig_img = Image.open(img_path).convert("RGB")
    
    # Étape 1 : Inférence
    bbox, water_mask = get_water_bbox(orig_img)
    
    # Étape 2 : Dessiner la Bounding Box sur l'originale pour voir si l'IA a juste
    img_with_bbox = np.array(orig_img).copy()
    if bbox:
        cv2.rectangle(img_with_bbox, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 5)
        
    # Étape 3 : Application du Crop et Letterbox
    if bbox:
        cropped_img = orig_img.crop(bbox)
        square_img = pad_to_square(cropped_img).resize((512, 512), Image.LANCZOS)
    else:
        # Fallback si pas d'eau
        w, h = orig_img.size
        new_dim = min(w, h)
        left = (w - new_dim) / 2
        top = (h - new_dim) / 2
        square_img = orig_img.crop((left, top, left + new_dim, top + new_dim)).resize((512, 512), Image.LANCZOS)

    # Étape 4 : Super Tenseur
    super_tensor_img = create_super_tensor(square_img)

    # --- AFFICHAGE ---
    ax_orig = axes[i, 0] if len(sample_images) > 1 else axes[0]
    ax_mask = axes[i, 1] if len(sample_images) > 1 else axes[1]
    ax_crop = axes[i, 2] if len(sample_images) > 1 else axes[2]
    ax_tens = axes[i, 3] if len(sample_images) > 1 else axes[3]

    ax_orig.imshow(img_with_bbox)
    ax_orig.set_title(f"1. Originale + Bounding Box\n{img_name}")
    ax_orig.axis('off')

    ax_mask.imshow(water_mask, cmap='gray')
    ax_mask.set_title("2. Masque SegFormer brut")
    ax_mask.axis('off')

    ax_crop.imshow(square_img)
    ax_crop.set_title("3. RGB (Crop + Bandes Noires)\n-> 512x512")
    ax_crop.axis('off')

    ax_tens.imshow(super_tensor_img)
    ax_tens.set_title("4. Super Tenseur (Ce que voit le CNN)\n(CLAHE / NDTI / Saturation)")
    ax_tens.axis('off')

plt.tight_layout()
plt.show()