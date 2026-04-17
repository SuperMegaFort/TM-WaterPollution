import streamlit as st
import os
import pandas as pd
import csv
import random
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_IN   = os.path.join(BASE_DIR, "ground_truth", "dataset_complet.csv")
OUT_DIR  = os.path.join(BASE_DIR, "ground_truth")
CSV_OUT  = os.path.join(OUT_DIR, "ground_truth.csv")
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_LABELS = {0: "Propre", 1: "Coloration", 2: "Limon/Turbidité", 3: "Mousse", 4: "Irisation/Autres", 6: "Inclassable"}
CLASS_EMOJI = {0: "💧", 1: "🟠", 2: "🟫", 3: "🫧", 4: "🌈", 6: "❓"}

st.set_page_config(page_title="Water Pollution Labeler", layout="wide")

# --- FONCTIONS DE DONNÉES ---
@st.cache_data
def load_all_rows():
    csv_hints = {}
    if os.path.isfile(CSV_IN):
        with open(CSV_IN, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Classe", "").isdigit():
                    csv_hints[row["Nom_Image"]] = int(row["Classe"])
                    
    rows = []
    if os.path.isdir(DATA_DIR):
        for name in os.listdir(DATA_DIR):
            path = os.path.join(DATA_DIR, name)
            if os.path.isfile(path) and name.lower().endswith(('.png', '.jpg', '.jpeg')):
                rows.append({"name": name, "path": path, "orig_class": csv_hints.get(name, -1)})
    return rows

def load_labeled():
    labeled = {}
    if os.path.isfile(CSV_OUT):
        with open(CSV_OUT, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val = row.get("Label", "").strip()
                if val.isdigit():
                    labeled[row["Nom_Image"]] = int(val)
                else:
                    labeled[row["Nom_Image"]] = ""
    return labeled

def save_labels(labeled_dict):
    os.makedirs(OUT_DIR, exist_ok=True)
    sorted_items = sorted(labeled_dict.items())
    df = pd.DataFrame(sorted_items, columns=["Nom_Image", "Label"])
    df.to_csv(CSV_OUT, index=False)

def build_queue(target_class, all_rows, labeled_dict, task_mode):
    if task_mode == "À faire (Nouvelles)":
        # Images non labellisées
        candidates_prio = [r for r in all_rows if r["orig_class"] == target_class and r["name"] not in labeled_dict]
        candidates_other = [r for r in all_rows if r["orig_class"] != target_class and r["name"] not in labeled_dict]
        random.shuffle(candidates_prio)
        random.shuffle(candidates_other)
        return candidates_prio + candidates_other
    else:
        # Images DÉJÀ labellisées dans cette classe
        candidates = [r for r in all_rows if labeled_dict.get(r["name"]) == target_class]
        return candidates

# Callback pour la sauvegarde auto en mode Grille
def grid_auto_save(img_name, widget_key):
    st.session_state.labels[img_name] = st.session_state[widget_key]
    save_labels(st.session_state.labels)

# --- INITIALISATION SESSION STATE ---
if 'all_rows' not in st.session_state:
    st.session_state.all_rows = load_all_rows()
if 'labels' not in st.session_state:
    st.session_state.labels = load_labeled()
if 'screen' not in st.session_state:
    st.session_state.screen = "selector"
if 'target_class' not in st.session_state:
    st.session_state.target_class = None
if 'queue' not in st.session_state:
    st.session_state.queue = []
if 'queue_idx' not in st.session_state:
    st.session_state.queue_idx = 0
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Individuelle"
if 'task_mode' not in st.session_state:
    st.session_state.task_mode = "À faire (Nouvelles)"

# --- ROUTAGE DES ÉCRANS ---

if st.session_state.screen == "selector":
    st.title("💧 Water Pollution Labeler - Sélection")
    
    st.sidebar.title("⚙️ Paramètres")
    st.session_state.task_mode = st.sidebar.radio("Tâche actuelle :", ["À faire (Nouvelles)", "Vérification (Déjà annotées)"])
    st.session_state.view_mode = st.sidebar.radio("Mode d'affichage :", ["Individuelle", "Grille (Tri rapide)"])
    
    st.write(f"### Mode : {st.session_state.task_mode}")
    st.write("Choisissez la classe cible :")
    
    for c in sorted(CLASS_LABELS.keys()):
        already = sum(1 for v in st.session_state.labels.values() if v == c)
        done_str = f"{already} annotées"
        btn_text = f"{CLASS_EMOJI[c]} [{c}] {CLASS_LABELS[c].upper()}  —  ({done_str})"
        
        if st.button(btn_text, key=f"btn_{c}", use_container_width=True):
            st.session_state.target_class = c
            st.session_state.queue = build_queue(c, st.session_state.all_rows, st.session_state.labels, st.session_state.task_mode)
            st.session_state.queue_idx = 0
            st.session_state.screen = "annotation"
            st.rerun()
            
    st.write("---")
    st.write(f"**Total annoté :** {len(st.session_state.labels)} / {len(st.session_state.all_rows)} images.")

elif st.session_state.screen == "annotation":
    tgt_class = st.session_state.target_class
    st.title(f"Triage : {CLASS_EMOJI[tgt_class]} {CLASS_LABELS[tgt_class]} ({st.session_state.task_mode})")
    
    if st.sidebar.button("⬅️ Retour au Menu", use_container_width=True):
        st.session_state.screen = "selector"
        st.rerun()

    # --- MODE GRILLE ---
    if st.session_state.view_mode == "Grille (Tri rapide)":
        candidates = st.session_state.queue
        if not candidates:
            st.warning("Aucune image à afficher ici.")
        else:
            limit = 24
            total_pages = max(0, (len(candidates) - 1) // limit)
            page = st.number_input(f"Page (0 à {total_pages})", min_value=0, max_value=total_pages, step=1)
            df_page = candidates[page*limit : (page+1)*limit]
            
            cols_per_row = 4
            for i in range(0, len(df_page), cols_per_row):
                batch = df_page[i:i+cols_per_row]
                cols = st.columns(cols_per_row)
                for j, item in enumerate(batch):
                    with cols[j]:
                        try:
                            st.image(Image.open(item["path"]), use_container_width=True)
                            
                            # Récupération de la valeur actuelle pour l'index du selectbox
                            current_val = st.session_state.labels.get(item["name"], tgt_class)
                            opts = list(CLASS_LABELS.keys())
                            idx = opts.index(current_val) if current_val in opts else 0
                            
                            w_key = f"grid_{item['name']}"
                            st.selectbox(
                                "Label",
                                options=opts,
                                index=idx,
                                format_func=lambda x: f"{CLASS_EMOJI.get(x,'')} {CLASS_LABELS.get(x,'')}",
                                key=w_key,
                                on_change=grid_auto_save,
                                args=(item["name"], w_key),
                                label_visibility="collapsed"
                            )
                        except Exception as e:
                            st.error(f"Image corrompue: {item['name']}")

    # --- MODE INDIVIDUEL ---
    else:
        if st.session_state.queue_idx >= len(st.session_state.queue):
            st.success("Tâche terminée pour cette file !")
            if st.button("⬅️ Menu Principal", use_container_width=True):
                st.session_state.screen = "selector"
                st.rerun()
        else:
            current_img = st.session_state.queue[st.session_state.queue_idx]
            col_img, col_act = st.columns([2, 1])
            
            with col_img:
                try:
                    st.image(Image.open(current_img["path"]), use_container_width=True)
                except:
                    st.error("Impossible d'afficher l'image.")
                st.caption(f"Fichier: {current_img['name']} | Progression: {st.session_state.queue_idx + 1} / {len(st.session_state.queue)}")
            
            with col_act:
                st.write("### Assigner :")
                for cid, name in CLASS_LABELS.items():
                    btype = "primary" if cid == tgt_class else "secondary"
                    if st.button(f"{CLASS_EMOJI[cid]} {name}", key=f"bi_{cid}", type=btype, use_container_width=True):
                        st.session_state.labels[current_img["name"]] = cid
                        save_labels(st.session_state.labels) # Sauvegarde sûre à chaque clic
                        st.session_state.queue_idx += 1
                        st.rerun()
                
                st.write("---")
                col_skip, col_del = st.columns(2)
                with col_skip:
                    if st.button("⏩ Ignorer", use_container_width=True):
                        st.session_state.queue_idx += 1
                        st.rerun()
                with col_del:
                    if st.button("🗑️ Effacer label", use_container_width=True):
                        st.session_state.labels[current_img["name"]] = ""
                        save_labels(st.session_state.labels)
                        st.session_state.queue_idx += 1
                        st.rerun()