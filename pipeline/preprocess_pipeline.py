"""
==============================================================================
 PREPROCESSING PIPELINE — Water Pollution Detection (Refactored Smart Crop)
==============================================================================
 Génère deux versions du dataset prétraité :
   - no_mask    : image originale complète redimensionnée.
   - smart_crop : isolation dynamique par IA SegFormer.
   - manual_crop: isolation par zone d'intérêt manuelle (rois.json).

 CLASSIFICATION BINAIRE :
   label 0 (clean)    → images CSV classe 0  (Propre)
   label 1 (polluted) → images CSV classes 1,3,4,6  (Coloration, mousse, turbidité)

 DOMAINE (pour Gradient Reversal Layer) : Cartographié dynamiquement par nom_image
==============================================================================
"""

import os
import csv
import cv2
import json
import random
import argparse
import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIGURATION ET MODÈLE IA (SEGFORMER)
# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
processor = None
segmentation_model = None

def get_segformer():
    # Lazy loading: SegFormer ne se charge que si on utilise --smart-crop
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    global processor, segmentation_model
    if segmentation_model is None:
        print(f"\n[INFO] Chargement du modèle SegFormer sur {DEVICE}...")
        model_id = "nvidia/segformer-b0-finetuned-ade-512-512"
        processor = SegformerImageProcessor.from_pretrained(model_id)
        segmentation_model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(DEVICE)
        segmentation_model.eval()
    return processor, segmentation_model

# ─────────────────────────────────────────────
# CONFIGURATION PAR DÉFAUT
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "ground_truth")
CSV_IN      = os.path.join(DATA_DIR, "ground_truth.csv")
OUT_DIR     = os.path.join(BASE_DIR, "data_preprocessed")
RANDOM_SEED = 42

BINARY_MAP = {
    0: 0, 1: 1, 4: 1, 6: 1
}
LABEL_NAMES_BIN = {0: "clean", 1: "polluted"}
LABEL_NAMES_MULTI = {0: "0_Propre", 1: "1_Coloration", 2: "2_Limon", 3: "3_Mousse"}

IS_MULTICLASS = False
WATER_CLASSES = [21, 26, 60, 128] # ADE20K SegFormer IDs

# ─────────────────────────────────────────────
# CHARGEMENT DU CSV
# ─────────────────────────────────────────────
def load_csv(csv_path):
    rows_by_class = {}
    skipped_class = 0
    skipped_file = 0
    domain_map = {}
    domain_counter = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name   = row["Nom_Image"]
            raw_cls = row["Label"].strip()
            if not raw_cls:
                continue
            classe = int(raw_cls)
            
            if IS_MULTICLASS:
                if classe == 4 or classe == 6:
                    skipped_class += 1
                    continue
                label = classe
            else:
                label  = BINARY_MAP.get(classe)
                if label is None:
                    skipped_class += 1
                    continue

            path = os.path.join(DATA_DIR, str(classe), name)
            if not os.path.isfile(path):
                skipped_file += 1
                continue

            domain_str = name.split("_")[-1].split(".")[0]
            if domain_str not in domain_map:
                domain_map[domain_str] = domain_counter
                domain_counter += 1
            domain = domain_map[domain_str]

            entry = {"name": name, "path": path, "orig_class": classe, "label": label, "domain": domain}
            rows_by_class.setdefault(label, []).append(entry)

    total_loaded = sum(len(l) for l in rows_by_class.values())
    print(f"  CSV chargé → {total_loaded} images au total réparties en {len(rows_by_class)} classes")
    return rows_by_class, domain_map

# ─────────────────────────────────────────────
# SUPER-TENSEUR (CLAHE + NDTI + HSV)
# ─────────────────────────────────────────────
def create_super_tensor(img_pil):
    img_rgb = np.array(img_pil)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    canal_1 = clahe.apply(gray)
    
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    ndti_math = (R - G) / (R + G + 1e-7)
    canal_2 = cv2.normalize(ndti_math, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    canal_2 = canal_2.astype(np.uint8)
    
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    canal_3 = hsv[:, :, 1]
    
    super_tensor_np = np.dstack((canal_1, canal_2, canal_3))
    return Image.fromarray(super_tensor_np)

# ─────────────────────────────────────────────
# SÉLECTION MANUELLE (OpenCV ROI)
# ─────────────────────────────────────────────
def setup_manual_rois(all_rows):
    roi_file = os.path.join(BASE_DIR, "rois.json")
    rois = {}
    
    # Si le fichier existe déjà, on le charge pour éviter de tout refaire
    if os.path.exists(roi_file):
        print(f"\n  [INFO] Fichier de ROI existant trouvé ({roi_file}). Chargement...")
        with open(roi_file, "r", encoding="utf-8") as f:
            return json.load(f)
            
    rivers = {}
    for row in all_rows:
        river = row["name"].split("_")[-1].split(".")[0]
        if river not in rivers:
            rivers[river] = row["path"]
            
    print("\n" + "="*60)
    print("  🖌️  MODE DÉTOURAGE MANUEL (OpenCV)")
    print("="*60)
    print(" Instructions :")
    print(" 1. Une image par rivière va s'ouvrir.")
    print(" 2. Dessinez le rectangle autour de l'eau avec la souris.")
    print(" 3. Appuyez sur [ENTRÉE] ou [ESPACE] pour valider le rectangle.")
    print(" 4. Appuyez sur [c] pour annuler le rectangle et recommencer.")
    print(" ============================================================\n")
    
    for river, path in rivers.items():
        img = cv2.imread(path)
        if img is None: continue
            
        # Couper la bannière Reconyx pour l'affichage (7% du bas)
        h, w = img.shape[:2]
        banner_height = int(h * 0.07)
        img_disp = img[:h - banner_height, :] 
        
        window_name = f"Selection ROI - Riviere : {river} (ENTREE pour valider)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720) # Pour que ça ne dépasse pas de l'écran
        
        # Ouvre l'interface de sélection
        r = cv2.selectROI(window_name, img_disp, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        
        # Si r[2] (largeur) et r[3] (hauteur) > 0, l'utilisateur a bien validé un rectangle
        if r[2] > 0 and r[3] > 0:
            rois[river] = {"x": int(r[0]), "y": int(r[1]), "width": int(r[2]), "height": int(r[3])}
            print(f"  ✅ ROI validé pour la rivière '{river}'")
        else:
            print(f"  ⚠️ Aucun ROI défini pour la rivière '{river}' (Fallback sera utilisé)")
            
    # Sauvegarde pour les prochains entraînements
    with open(roi_file, "w", encoding="utf-8") as f:
        json.dump(rois, f, indent=4)
        
    print(f"\n  💾 Coordonnées sauvegardées dans 'rois.json' avec succès !\n")
    return rois

# ─────────────────────────────────────────────
# UTILITAIRES DE DÉTECTION ET CADRAGE (LETTERBOX)
# ─────────────────────────────────────────────
def get_water_bbox(img_pil):
    """Inférence SegFormer avec filtrage de la plus grande zone continue pour ignorer le bruit"""
    proc, model = get_segformer()
    inputs = proc(images=img_pil, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        
    mask = torch.nn.functional.interpolate(logits, size=img_pil.size[::-1], mode="bilinear", align_corners=False)
    mask = mask.argmax(dim=1)[0]
    
    water_mask = torch.zeros_like(mask, dtype=torch.uint8)
    for cls in WATER_CLASSES:
        water_mask |= (mask == cls).to(torch.uint8)
    
    water_mask_np = (water_mask.cpu().numpy() * 255).astype(np.uint8)
    
    h_img, w_img = water_mask_np.shape
    water_mask_np[0:int(h_img * 0.30), :] = 0 
    
    kernel = np.ones((7, 7), np.uint8)
    water_mask_np = cv2.morphologyEx(water_mask_np, cv2.MORPH_OPEN, kernel)
    water_mask_np = cv2.morphologyEx(water_mask_np, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(water_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours: return None
        
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    if area < 3000: return None
        
    x, y, w, h = cv2.boundingRect(largest_contour)
    if (w * h) > (h_img * w_img * 0.85): return None
    
    return (x, y, x + w, y + h)

def pad_to_square(img_pil):
    width, height = img_pil.size
    if width == height: return img_pil
    size = max(width, height)
    result = Image.new("RGB", (size, size), (0, 0, 0))
    result.paste(img_pil, ((size - width) // 2, (size - height) // 2))
    return result

# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL DE COMPILATION
# ─────────────────────────────────────────────
def process_images(rows, out_base, target_size, crop_mode="none", rois=None, super_tensor=False):
    saved = []
    errors = 0
    fallbacks = 0
    desc_str = os.path.basename(out_base)
    
    print(f"\n🚀 Démarrage du traitement vers : {out_base} ({len(rows)} images à traiter)")

    for entry in tqdm(rows, desc=f"{desc_str:20s}", dynamic_ncols=True, leave=False):
        name   = entry["name"]
        label  = entry["label"]
        domain = entry["domain"]
        lname = LABEL_NAMES_MULTI.get(label, str(label)) if IS_MULTICLASS else LABEL_NAMES_BIN.get(label, str(label))
        dst_dir  = os.path.join(out_base, lname)
        os.makedirs(dst_dir, exist_ok=True)
        dst_path = os.path.join(dst_dir, name)
        
        final_name = name
        if super_tensor:
            dst_path = os.path.splitext(dst_path)[0] + ".png"
            final_name = os.path.splitext(name)[0] + ".png"

        if os.path.isfile(dst_path):
            saved.append((final_name, label, domain))
            continue

        try:
            final_img = Image.open(entry["path"]).convert("RGB")
            
            # SUPPRESSION DE LA BANNIÈRE NOIRE RECONYX (7% de la hauteur totale)
            w_orig, h_orig = final_img.size
            banner_height = int(h_orig * 0.07)
            final_img = final_img.crop((0, 0, w_orig, h_orig - banner_height))
            
            # 1. DÉCOUPAGE DE L'EAU (CROP)
            if crop_mode != "none":
                bbox = None
                river = name.split("_")[-1].split(".")[0]
                
                if crop_mode == "manual" and rois and river in rois:
                    r = rois[river]
                    bbox = (r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"])
                elif crop_mode == "segformer":
                    bbox = get_water_bbox(final_img)
                    
                if bbox:
                    x1, y1, x2, y2 = bbox
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_orig, x2), min(h_orig - banner_height, y2)
                    final_img = final_img.crop((x1, y1, x2, y2))
                else:
                    fallbacks += 1
                    w_crop, h_crop = final_img.size
                    new_dim = min(w_crop, h_crop)
                    left = (w_crop - new_dim) / 2
                    top = (h_crop - new_dim) / 2
                    final_img = final_img.crop((left, top, left + new_dim, top + new_dim))
                    
            # 2. FILTRAGE CHIMIQUE (SUPER TENSEUR) -> Appliqué SUR L'EAU uniquement
            if super_tensor:
                final_img = create_super_tensor(final_img)

            # 3. LETTERBOXING (PAD) -> Ajoute du Vrai Noir (0,0,0) protégé des filtres
            final_img = pad_to_square(final_img)
                    
            # 4. REDIMENSIONNEMENT ET SAUVEGARDE
            if target_size:
                final_img = final_img.resize((target_size, target_size), Image.LANCZOS)
                
            if super_tensor:
                final_img.save(dst_path, format="PNG")
            else:
                final_img.save(dst_path, quality=95)
                
            saved.append((final_name, label, domain))
            
        except Exception:
            errors += 1

    if errors:
        print(f"  ⚠ {errors} erreurs ignorées (fichiers corrompus)")
    if fallbacks:
        print(f"  ⚠ {fallbacks} fallbacks exécutés (pas de zone détectée)")
    return saved

def save_csv(saved_rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nom_Image", "Label", "Domain"])
        for name, label, domain in saved_rows:
            writer.writerow([name, label, domain])

def save_domain_map(domain_map, out_dir):
    path = os.path.join(out_dir, "domain_map.json")
    with open(path, "w", encoding="utf-8") as f:
        inv_map = {v: k for k, v in domain_map.items()}
        json.dump(inv_map, f, indent=4)

def print_report(saved, version):
    counts = {}
    for _, lbl, _ in saved:
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"  ✅ {version} terminé : {sum(counts.values())} images sauvegardées.")

# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline — Water Pollution")
    parser.add_argument("--csv",        default=CSV_IN, help="Fichier CSV de ground_truth")
    parser.add_argument("--size",       type=int, default=512, help="Redimensionner les images (défaut: 512)")
    parser.add_argument("--seed",       type=int, default=RANDOM_SEED)
    parser.add_argument("--multiclass", action="store_true", help="Dataset multi-classes")
    parser.add_argument("--smart-crop", action="store_true", help="Utilise l'IA SegFormer pour détourer")
    parser.add_argument("--manual-crop",action="store_true", help="Ouvre OpenCV pour détourer manuellement chaque rivière")
    args = parser.parse_args()

    global IS_MULTICLASS, OUT_DIR
    if args.multiclass:
        IS_MULTICLASS = True
        OUT_DIR = os.path.join(BASE_DIR, "data_preprocessed_multiclass")

    random.seed(args.seed)
    print(f"\n{'='*60}")
    print(f"  PREPROCESSING PIPELINE {'(MULTI-CLASS)' if IS_MULTICLASS else '(BINARY)'}")
    print(f"{'='*60}")
    
    rows_by_class, domain_map = load_csv(args.csv)
    
    all_rows = []
    for lst in rows_by_class.values():
        all_rows.extend(lst)
    random.shuffle(all_rows)

    # Version SANS MASQUE
    print("\n2. Génération version: SANS MASQUE (Baseline)...")
    out_nm  = os.path.join(OUT_DIR, "no_mask")
    saved_nm = process_images(all_rows, out_nm, args.size, crop_mode="none")
    csv_nm   = os.path.join(OUT_DIR, "dataset_no_mask.csv")
    save_csv(saved_nm, csv_nm)
    print_report(saved_nm, "no_mask")

    # Version MANUELLE (OpenCV)
    if args.manual_crop:
        print("\n3. Génération version: MANUAL CROP (Détourage Manuel + Letterboxing)...")
        rois = setup_manual_rois(all_rows)

        out_mc  = os.path.join(OUT_DIR, "manual_crop")
        saved_mc = process_images(all_rows, out_mc, args.size, crop_mode="manual", rois=rois, super_tensor=False)
        csv_mc   = os.path.join(OUT_DIR, "dataset_manual_crop.csv")
        save_csv(saved_mc, csv_mc)
        print_report(saved_mc, "manual_crop")
        
        print("\n4. Génération version: MANUAL SUPER-TENSEUR (CLAHE, NDTI, HSV)...")
        out_tensor = os.path.join(OUT_DIR, "manual_crop_tensor")
        saved_tensor = process_images(all_rows, out_tensor, args.size, crop_mode="manual", rois=rois, super_tensor=True)
        csv_tensor = os.path.join(OUT_DIR, "dataset_manual_crop_tensor.csv")
        save_csv(saved_tensor, csv_tensor)
        print_report(saved_tensor, "manual_crop_tensor")

    # Version SMART CROP (IA SegFormer)
    elif args.smart_crop:
        print("\n3. Génération version: SMART CROP (IA SegFormer + Letterboxing)...")

        out_sc  = os.path.join(OUT_DIR, "smart_crop")
        saved_sc = process_images(all_rows, out_sc, args.size, crop_mode="segformer", super_tensor=False)
        csv_sc   = os.path.join(OUT_DIR, "dataset_smart_crop.csv")
        save_csv(saved_sc, csv_sc)
        print_report(saved_sc, "smart_crop")
        
        print("\n4. Génération version: SMART SUPER-TENSEUR (CLAHE, NDTI, HSV)...")
        out_tensor = os.path.join(OUT_DIR, "smart_crop_tensor")
        saved_tensor = process_images(all_rows, out_tensor, args.size, crop_mode="segformer", super_tensor=True)
        csv_tensor = os.path.join(OUT_DIR, "dataset_smart_crop_tensor.csv")
        save_csv(saved_tensor, csv_tensor)
        print_report(saved_tensor, "smart_crop_tensor")

    save_domain_map(domain_map, OUT_DIR)
    print(f"\n{'='*60}\n  ✅ TERMINÉ\n{'='*60}")

if __name__ == "__main__":
    main()