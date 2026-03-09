import os
import re
import pandas as pd

def associer_plages_images_csv(dossier_images, fichier_csv_meta, fichier_sortie):
    print(f"1. Lecture du fichier CSV des métadonnées : {fichier_csv_meta}")
    
    # Lire directement le fichier CSV (plus d'Excel)
    df_meta = pd.read_csv(fichier_csv_meta)
         
    # Nettoyer les noms de colonnes (enlever les espaces cachés au cas où)
    df_meta.columns = df_meta.columns.str.strip()
    
    # Ce dictionnaire stockera nos clés doubles : (nom_riviere, RCNX) -> classe
    mapping_classes = {}
    
    # On va garder en mémoire la liste de toutes les rivières connues
    rivieres_connues = set()
    
    print("2. Analyse des plages de numéros et rivières...")
    for index, row in df_meta.iterrows():
        # Utilisation des nouveaux noms de colonnes
        riviere = str(row.get("river", '')).strip().lower()
        debut = str(row.get('start', '')).strip()
        fin = str(row.get('end', '')).strip()
        classe = row.get('class')
        
        # Ignorer les lignes vides ou mal formatées
        if pd.isna(classe) or not debut.upper().startswith('RCNX') or not riviere:
            continue
            
        # Ajouter la rivière à notre liste connue (aire, arve, bloch, avril...)
        rivieres_connues.add(riviere)
            
        match_d = re.match(r"([A-Za-z]+)(\d+)", debut)
        match_f = re.match(r"([A-Za-z]+)(\d+)", fin)
        
        # Dérouler la plage d'images de "start" à "end"
        if match_d and match_f and match_d.group(1).upper() == match_f.group(1).upper():
            prefixe = match_d.group(1).upper() # "RCNX"
            num_d_str = match_d.group(2) # "1771"
            num_f_str = match_f.group(2) # "1774"
            
            d_val = int(num_d_str)
            f_val = int(num_f_str)
            
            # Sécurité : vérifier que start est bien <= end
            if d_val <= f_val:
                for i in range(d_val, f_val + 1):
                    # zfill permet de garder les zéros, ex: "0021" au lieu de "21"
                    nom_genere = f"{prefixe}{str(i).zfill(len(num_d_str))}"
                    mapping_classes[(riviere, nom_genere)] = int(classe)
        else:
            # Si "end" est vide ou bizarre, on stocke juste le "start"
            mapping_classes[(riviere, debut.upper())] = int(classe)

    print(f"   -> {len(mapping_classes)} images virtuelles créées en mémoire à partir des plages.")
    
    print(f"\n3. Parcours du dossier d'images : {dossier_images}")
    donnees_finales = []
    images_trouvees = 0
    images_sans_classe = 0
    
    # Parcourir les vraies images
    if not os.path.exists(dossier_images):
        print(f"Erreur : Le dossier {dossier_images} est introuvable.")
        return

    for nom_fichier in os.listdir(dossier_images):
        nom_minuscule = nom_fichier.lower()
        
        if nom_minuscule.endswith(('.jpg', '.jpeg', '.png')):
            # Trouver le RCNX dans le nom de l'image (ex: RCNX1773)
            match_nom = re.search(r"RCNX\d+", nom_fichier, re.IGNORECASE)
            
            if match_nom:
                ancien_nom = match_nom.group(0).upper()
                classe_trouvee = None
                
                # Chercher le nom d'une de nos rivières connues dans le nom de l'image
                for riviere in rivieres_connues:
                    if riviere in nom_minuscule:
                        cle = (riviere, ancien_nom)
                        # Vérification instantanée dans notre mémoire
                        if cle in mapping_classes:
                            classe_trouvee = mapping_classes[cle]
                            break # On a trouvé, on arrête de chercher la rivière
                
                # Si on a trouvé une classe valide
                if classe_trouvee is not None:
                    donnees_finales.append({
                        "Nom_Image": nom_fichier,
                        "Classe": classe_trouvee
                    })
                    images_trouvees += 1
                else:
                    images_sans_classe += 1
                    
    print("\n4. Création du fichier CSV final...")
    df_final = pd.DataFrame(donnees_finales)
    
    if not df_final.empty:
        df_final.to_csv(fichier_sortie, index=False)
        print(f"Succès ! {images_trouvees} images ont été associées à leur classe.")
        print(f"Fichier généré : {fichier_sortie}")
    else:
        print("Attention : Aucune correspondance n'a été trouvée. Vérifie les noms.")
        
    if images_sans_classe > 0:
        print(f"Note : {images_sans_classe} images ont été ignorées (elles ne sont pas dans les plages du fichier {fichier_csv_meta}).")

# --- PARAMÈTRES (à modifier avec tes propres chemins) ---
dossier_des_images = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/data"
# Modifie ce chemin pour qu'il pointe bien vers ton nouveau metadata_photo.csv
fichier_csv_source = "/Users/cyriltelley/Desktop/MSE/Third_semester/TM-WaterPollution/Photos/Sélection de photos événement pollution/metadata_photo.csv"
fichier_csv_final = "dataset_ml_final.csv"

# --- Lancement ---
associer_plages_images_csv(dossier_des_images, fichier_csv_source, fichier_csv_final)