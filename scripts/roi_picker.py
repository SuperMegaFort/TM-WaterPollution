import cv2
import json
import os
import glob

def main():
    print("\n---------------------------------------------------------")
    print(" 🎯 TM-WaterPollution : Configuration ROI (Static Crop)")
    print("---------------------------------------------------------")
    print(" Instructions OpenCV :")
    print(" 1. Dessinez un carré/rectangle avec la souris.")
    print(" 2. Appuyez sur ENTRÉE ou ESPACE pour sauvegarder la zone.")
    print(" 3. Appuyez sur 'c' pour ignorer cette rivière (pas de crop).")
    print("---------------------------------------------------------\n")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gt_dir = os.path.join(base_dir, "ground_truth")
    roi_file = os.path.join(base_dir, "rois.json")
    
    # 1. Scanner toutes les rivières existantes (Extraction depuis Nom_Image)
    images_paths = glob.glob(os.path.join(gt_dir, "*", "*.jpg")) + glob.glob(os.path.join(gt_dir, "*", "*.jpeg"))
    
    if not images_paths:
        print("❌ Aucune image trouvée dans ground_truth/ ! Avez-vous exécuté la copie ?")
        return

    rivers = {}
    for path in images_paths:
        name = os.path.basename(path)
        # 08012026_121000_RCNX0004_Ziplo.jpg -> Ziplo
        river_name = name.split("_")[-1].split(".")[0]
        # On ne garde qu'une seule image de référence par rivière
        if river_name not in rivers:
            rivers[river_name] = path

    print(f"🌊 Rivières uniques détectées : {list(rivers.keys())}")

    # 2. Charger l'ancien JSON s'il existe
    rois = {}
    if os.path.exists(roi_file):
        with open(roi_file, "r") as f:
            rois = json.load(f)
            print("📁 Fichier 'rois.json' existant chargé.")

    # 3. Boucle sur chaque rivière pour afficher l'interface
    cv2.namedWindow('Select ROI', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Select ROI', 1280, 720)

    for river, img_path in sorted(rivers.items()):
        print(f"\n=> 📸 Chargement de l'image de référence pour la rivière : [ {river} ]")
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # Si un ROI existe déjà, on le dessine en rouge pour information
        if river in rois:
            r = rois[river]
            cv2.rectangle(img, (r["x"], r["y"]), (r["x"] + r["width"], r["y"] + r["height"]), (0, 0, 255), 4)
            print(f"   (Un cadre existe déjà. Redessinez par-dessus pour écraser, ou appuyez sur Space/Enter pour valider)")

        # Lancement de l'outil interactif natif de CV2
        roi = cv2.selectROI('Select ROI', img, showCrosshair=True, fromCenter=False)
        
        # roi = (x, y, w, h)
        x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
        
        # Si l'utilisateur a réellement dessiné une box
        if w > 0 and h > 0:
            rois[river] = {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            }
            print(f"   ✅ ROI Sauvegardé pour {river} : {rois[river]}")
        else:
            print(f"   ⏩ Ignoré (Aucune zone valide sélectionnée).")

    cv2.destroyAllWindows()

    # 4. Sauvegarde Finale
    with open(roi_file, "w") as f:
        json.dump(rois, f, indent=4)
    print(f"\n🎉 Terminé ! Les {len(rois)} coordonnées ont été sauvegardées dans 'rois.json' !")
    print("🚀 Vous pouvez maintenant relancer process_pipeline.py avec --smart-crop !")

if __name__ == "__main__":
    main()
