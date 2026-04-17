import os
from PIL import Image, ImageOps
from tqdm import tqdm

def preparer_images_masquees(input_dir='dataset/val/4', output_dir='dataset/data_masked/val/4', 
                             mask_aire_path='mask/mask_aire.png', mask_ziplo_path='mask/mask_ziplo.png'):
    
    print("1. Chargement des masques...")
    if not os.path.exists(mask_aire_path) or not os.path.exists(mask_ziplo_path):
        print("🚨 ERREUR : Les fichiers masque_aire.png ou masque_ziplo.png sont introuvables !")
        return

    # On force les masques en Noir et Blanc strict (Blanc=255, Noir=0)
    mask_aire = Image.open(mask_aire_path).convert("L").point(lambda p: 255 if p > 128 else 0)
    mask_ziplo = Image.open(mask_ziplo_path).convert("L").point(lambda p: 255 if p > 128 else 0)

    # On calcule la Bounding Box (la boîte qui englobe tout le BLANC)
    bbox_aire = mask_aire.getbbox()
    bbox_ziplo = mask_ziplo.getbbox()
    
    os.makedirs(output_dir, exist_ok=True)
    images = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    
    print(f"2. Découpage de {len(images)} images en cours...")
    for file in tqdm(images):
        filepath = os.path.join(input_dir, file)
        img = Image.open(filepath).convert("RGB")
        nom_minuscule = file.lower()

        try:
            if 'ziplo' in nom_minuscule:
                # 1. Redimensionner le masque à la taille de l'image
                mask = mask_ziplo.resize(img.size)
                # 2. Appliquer le masque (fond noir)
                black_bg = Image.new("RGB", img.size, (0, 0, 0))
                img_masked = Image.composite(img, black_bg, mask)
                # 3. Couper tout le noir inutile autour du blanc
                if bbox_ziplo: img_masked = img_masked.crop(bbox_ziplo)

            elif 'aire' in nom_minuscule:
                mask = mask_aire.resize(img.size)
                black_bg = Image.new("RGB", img.size, (0, 0, 0))
                img_masked = Image.composite(img, black_bg, mask)
                if bbox_aire: img_masked = img_masked.crop(bbox_aire)
            else:
                continue

            # 4. Ajouter des petites bandes noires pour faire un carré parfait (évite l'écrasement)
            max_dim = max(img_masked.size)
            img_final = ImageOps.pad(img_masked, size=(max_dim, max_dim), color=(0, 0, 0))

            # 5. Sauvegarder dans le nouveau dossier
            img_final.save(os.path.join(output_dir, file))
            
        except Exception as e:
            print(f"Erreur avec {file}: {e}")
            
    print(f"\n✅ Terminé ! Allez vérifier le dossier '{output_dir}'.")

# --- LANCEMENT ---
preparer_images_masquees()