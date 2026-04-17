import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image

st.set_page_config(page_title="TM-WaterPollution Analyst", layout="wide")

# Configuration globale et chemins
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eval_dir = os.path.join(base_dir, "evaluation_results")
gt_csv_path = os.path.join(base_dir, "ground_truth", "ground_truth.csv")

st.sidebar.title("🛠️ Outils Analyste")
mode = st.sidebar.radio("Sélectionnez le mode :", ["Analyse d'Erreurs", "Éditeur de Labels (GT)"])

if mode == "Analyse d'Erreurs":
    st.title("🔬 Débogueur de Modèles (Analyse d'Erreurs)")
    st.markdown("Identifiez exactement sur quelles textures ou météos le modèle commet un *Faux Positif* ou un *Faux Négatif*.")
else:
    st.title("🏷️ Éditeur de Labels (Ground Truth)")
    st.markdown("Vérifiez et corrigez les labels de vos images directement. Les modifications sont sauvegardées dans `ground_truth.csv`.")

gt_map = {}

# 2. Scanner le Ground Truth pour faire la correspondance (Cache pour rapidité)
@st.cache_data
def get_gt_mapping():
    mapping = {}
    gt_images = glob.glob(os.path.join(base_dir, "ground_truth", "**", "*.jpg"), recursive=True)
    gt_images += glob.glob(os.path.join(base_dir, "ground_truth", "**", "*.jpeg"), recursive=True)
    gt_images += glob.glob(os.path.join(base_dir, "ground_truth", "**", "*.png"), recursive=True)
    for path in gt_images:
        clean_name = os.path.splitext(os.path.basename(path))[0]
        mapping[clean_name] = path
    return mapping

gt_map = get_gt_mapping()

# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 : ANALYSE D'ERREURS (COMPARAISON PRÉDICTIONS)
# ─────────────────────────────────────────────────────────────────────────────
if mode == "Analyse d'Erreurs":
    # Scanner tous les fichiers predictions.csv disponibles
    csv_files = glob.glob(os.path.join(eval_dir, "**", "predictions.csv"), recursive=True)
    
    if not csv_files:
        st.warning("Aucun fichier `predictions.csv` trouvé. Lancer une évaluation avec `evaluate_grl.py`.")
        st.stop()

    csv_dict = {os.path.relpath(f, eval_dir): f for f in csv_files}
    st.sidebar.header("Analyse de Performance")
    selected_csv_key = st.sidebar.selectbox("Fichier de prédictions :", list(csv_dict.keys()))
    df = pd.read_csv(csv_dict[selected_csv_key])

else:
    # ─────────────────────────────────────────────────────────────────────────────
    # MODE 2 : ÉDITEUR DE LABELS (SAUVEGARDE GT)
    # ─────────────────────────────────────────────────────────────────────────────
    if not os.path.exists(gt_csv_path):
        st.error(f"Fichier `ground_truth.csv` introuvable dans {gt_csv_path}")
        st.stop()

    # Utilisation du session_state pour stocker les modifications en cours
    if 'gt_df' not in st.session_state:
        st.session_state.gt_df = pd.read_csv(gt_csv_path)

    df_gt = st.session_state.gt_df

    # --- Filtres Sidebar ---
    st.sidebar.header("Filtres Ground Truth")
    
    # Extraire les rivières des noms de fichiers
    rivers = sorted(list(set(name.split("_")[-1].split(".")[0] for name in df_gt['Nom_Image'])))
    selected_river = st.sidebar.selectbox("Rivière :", ["Toutes"] + rivers)
    
    unique_labels = sorted(df_gt['Label'].unique().tolist())
    selected_lbl = st.sidebar.selectbox("Label actuel :", ["Tous"] + unique_labels)

    # Filtrer le dataframe pour l'affichage
    df_filtered = df_gt.copy()
    if selected_river != "Toutes":
        df_filtered = df_filtered[df_filtered['Nom_Image'].str.contains(selected_river, case=False)]
    if selected_lbl != "Tous":
        df_filtered = df_filtered[df_filtered['Label'] == int(selected_lbl)]

    # Pagination simple car 2100 images c'est trop pour Streamlit
    limit = 40
    total_found = len(df_filtered)
    page = st.sidebar.number_input(f"Page (Max {total_found//limit})", min_value=0, max_value=max(0, total_found//limit), step=1)
    df_page = df_filtered.iloc[page*limit : (page+1)*limit]

    st.subheader(f"Affichage de {len(df_page)} images sur {total_found}")

    # --- Enregistrement ---
    if st.sidebar.button("💾 SAUVEGARDER LES MODIFICATIONS", type="primary"):
        st.session_state.gt_df.to_csv(gt_csv_path, index=False)
        st.sidebar.success("Fichier `ground_truth.csv` mis à jour !")

    # --- Grille d'affichage ---
    cols_per_row = 4
    rows_list = [df_page.iloc[i:i+cols_per_row] for i in range(0, len(df_page), cols_per_row)]
    
    label_options = [0, 1, 2, 3, 4, 6] # Labels standards
    label_help = {0: "Propre", 1: "Coloré", 2: "Limon", 3: "Mousse", 4: "Trouble", 6: "Inclassable"}

    for r_df in rows_list:
        cols = st.columns(cols_per_row)
        for idx, (index_original, data) in enumerate(r_df.iterrows()):
            with cols[idx]:
                img_name = data['Nom_Image']
                clean_name = os.path.splitext(img_name)[0]
                img_path = gt_map.get(clean_name)

                if img_path and os.path.exists(img_path):
                    st.image(Image.open(img_path), use_container_width=True)
                else:
                    st.error(f"Image non trouvée : {img_name}")
                
                new_lbl = st.selectbox(f"Label: {img_name}", 
                                      options=label_options, 
                                      index=label_options.index(data['Label']) if data['Label'] in label_options else 0,
                                      key=f"label_{index_original}",
                                      help=label_help.get(data['Label'], ""))
                
                # Mise à jour immédiate dans le session_state
                if new_lbl != data['Label']:
                    st.session_state.gt_df.at[index_original, 'Label'] = new_lbl
                    st.toast(f"Modifié : {img_name} -> {new_lbl}")
        st.divider()

# Nettoyer et définir les catégories visuelles
# True_Label == 0 -> Propre, True_Label == 1 -> Pollué
def categorize(row):
    if row['True_Label'] == row['Pred_Label']:
        if row['True_Label'] == 1:
            return "Vrai Positif (Pollué trouvé)"
        else:
            return "Vrai Négatif (Propre validé)"
    else:
        if row['Pred_Label'] == 1:
            return "FAUX POSITIF 🔴 (Propre confondu avec Pollué)"
        else:
            return "FAUX NÉGATIF 🟡 (Pollution ratée)"

df['Category'] = df.apply(categorize, axis=1)

# Options de filtrage
st.sidebar.markdown("---")
show_ground_truth = st.sidebar.checkbox("Afficher le Ground Truth (Image complète)", value=True)

filter_cat = st.sidebar.radio("Afficher les images :", 
    ["Toutes", 
     "FAUX POSITIF 🔴 (Propre confondu avec Pollué)", 
     "FAUX NÉGATIF 🟡 (Pollution ratée)", 
     "Vrai Positif (Pollué trouvé)", 
     "Vrai Négatif (Propre validé)"])

if filter_cat != "Toutes":
    df_filtered = df[df['Category'] == filter_cat]
else:
    df_filtered = df

st.subheader(f"Résultats : {len(df_filtered)} images trouvées")

# 3. Affichage en grille (Row/Cols)
cols_per_row = 4
rows = [df_filtered.iloc[i:i+cols_per_row] for i in range(0, len(df_filtered), cols_per_row)]

for row_df in rows:
    cols = st.columns(cols_per_row)
    for idx, (index, data) in enumerate(row_df.iterrows()):
        with cols[idx]:
            try:
                # Recherche par nom de base (sans extension) pour matcher .png (tensor) vs .jpg (ground_truth)
                img_name_no_ext = os.path.splitext(data['Image'])[0]
                img_path = gt_map.get(img_name_no_ext, data['Path']) if show_ground_truth else data['Path']
                
                # Gestion robuste des extensions (.jpg vs .png) entre le CSV et le disque
                if not os.path.exists(img_path):
                    alt_ext = ".png" if img_path.lower().endswith(".jpg") or img_path.lower().endswith(".jpeg") else ".jpg"
                    alt_path = os.path.splitext(img_path)[0] + alt_ext
                    if os.path.exists(alt_path):
                        img_path = alt_path

                img = Image.open(img_path)
                # Correction suite au message de dépréciation Streamlit
                st.image(img, width='stretch')
            except Exception:
                st.error(f"Image introuvable : {os.path.basename(img_path)}")
                
            color = "red" if "FAUX" in data['Category'] else "green"
            st.markdown(f"**Image**: `{data['Image']}`")
            st.markdown(f"**Catégorie**: :{color}[{data['Category']}]")
            # N'afficher le Score/Distance que si pertinent (Siamois != 0)
            if data['Score'] != 0.0:
                 st.write(f"**Distance**: {data['Score']:.3f}")
            st.divider()

st.sidebar.markdown("---")
st.sidebar.info("💡 Astuce : Les Faux Positifs vous indiqueront si votre modèle est trompé par des reflets verts d'arbres, tandis que les Faux Négatifs vous montreront s'il ignore le limon léger.")
