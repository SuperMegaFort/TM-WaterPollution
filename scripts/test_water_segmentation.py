import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# Configuration
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
CROP_SIZE = 224

# ADE20K IDs corresponding to water (water=21, river=60, sea=26, lake=128)
WATER_CLASSES = [21, 26, 60, 128]

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def process_image(image_path, processor, model, output_dir):
    print(f"Processing: {image_path}")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size # (width, height)
    
    # 1. Inférence SegFormer
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits  # shape (batch_size, num_classes, height/4, width/4)
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=(original_size[1], original_size[0]), # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    # 2. Création du masque binaire (Eau = 255, Reste = 0)
    water_mask = np.isin(pred_seg, WATER_CLASSES).astype(np.uint8) * 255
    
    # 2.5 CORRECTION DES BANDES NOIRES
    # Le modèle IA hallucine parfois de l'eau dans les marges noires "letterbox" (ex: 3840x3840 padding).
    # On calcule la zone réelle de l'image (bounding box globale des pixels visibles)
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        rx, ry, rw, rh = cv2.boundingRect(coords)
        # On crée un masque géométrique de la vraie image
        valid_zone = np.zeros_like(water_mask)
        valid_zone[ry:ry+rh, rx:rx+rw] = 255
        # On efface purement et simplement toute prédiction d'eau en dehors du cadre
        water_mask = cv2.bitwise_and(water_mask, valid_zone)
    
    # 3. OpenCV : Trouver les contours
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    crop_rotated = None
    crop_inscribed = None
    rotated_box = None
    inscribed_box = None
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 1000:
            # Option C: Minimum Area Rotated Rectangle (Englobe toute la rivière, sans le fond)
            rect = cv2.minAreaRect(largest_contour)
            rotated_box = np.int32(cv2.boxPoints(rect))
            
            width = int(rect[1][0])
            height = int(rect[1][1])
            src_pts = rotated_box.astype("float32")
            # Point order mapping
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            crop_rotated_cv = cv2.warpPerspective(img_cv, M, (width, height))
            crop_rotated = Image.fromarray(cv2.cvtColor(crop_rotated_cv, cv2.COLOR_BGR2RGB))
            
            # Option D: Plus grand carré inscrit PURE EAU (Distance Transform)
            dist_transform = cv2.distanceTransform(water_mask, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist_transform)
            radius = int(max_val)
            # Le côté du plus grand carré inscrit dans ce cercle est r * sqrt(2)
            half_side = int(radius * 0.707)
            if half_side > 10:
                x1 = max(0, max_loc[0] - half_side)
                y1 = max(0, max_loc[1] - half_side)
                x2 = min(original_size[0], max_loc[0] + half_side)
                y2 = min(original_size[1], max_loc[1] + half_side)
                inscribed_box = (x1, y1, x2, y2)
                crop_inscribed = image.crop(inscribed_box)
    
    # 4. Affichage et Sauvegarde
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(image)
    axes[0].set_title("Image Originale")
    
    overlay = img_cv.copy()
    overlay[water_mask == 255] = [0, 255, 0] # L'eau en vert
    overlay = cv2.addWeighted(img_cv, 0.6, overlay, 0.4, 0)
    
    if rotated_box is not None:
        cv2.drawContours(overlay, [rotated_box], 0, (255, 0, 0), 4) # Box Rotative Bleu
    if inscribed_box is not None:
        x1, y1, x2, y2 = inscribed_box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 4) # Box Inscrite Jaune
            
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Mask SegFormer ADE20K")
    
    if crop_rotated:
        axes[2].imshow(crop_rotated.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS))
        axes[2].set_title("Opt C: Rotated Rect (Warped)")
    else:
        axes[2].imshow(np.zeros((224,224,3), dtype=np.uint8))
        axes[2].set_title("Aucune eau trouvée (Opt C)")
        
    if crop_inscribed:
        axes[3].imshow(crop_inscribed.resize((CROP_SIZE, CROP_SIZE), Image.LANCZOS))
        axes[3].set_title("Opt D: Plus pure zone d'eau")
    else:
        axes[3].imshow(np.zeros((224,224,3), dtype=np.uint8))
        axes[3].set_title("Aucune eau trouvée (Opt D)")

    for ax in axes:
        ax.axis("off")
        
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, os.path.basename(image_path))
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Sauvegardé: {out_path}")


if __name__ == "__main__":
    print(f"Chargement du processeur et du modèle: {MODEL_NAME}")
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Modèle chargé sur {device}.")

    # On utilise quelques vraies images du projet
    TEST_DIR = "data_preprocessed/no_mask/polluted"
    OUTPUT_DIR = "evaluation_results/segmentation_tests"
    
    import glob, random
    images = glob.glob(os.path.join(TEST_DIR, "*.jpg")) + glob.glob(os.path.join(TEST_DIR, "*.jpeg"))
    if not images:
        images = glob.glob("data_preprocessed/no_mask/clean/*.jpg")
        
    if images:
        sample_images = random.sample(images, min(5, len(images)))
        print(f"Test de {len(sample_images)} images...")
        for img_path in sample_images:
            process_image(img_path, processor, model, OUTPUT_DIR)
        print(f"\nTests terminés. Regardez les résultats dans le dossier '{OUTPUT_DIR}' !")
    else:
        print("Aucune image de test trouvée.")
