"""
==============================================================================
 GROUND TRUTH LABELER — Water Pollution Detection
==============================================================================
 Mode séquentiel : validez toutes les images d'une classe avant la suivante.

 RACCOURCIS :
   Entrée / Espace  → Accepter la classe proposée
   0, 1, 2, 3, 4    → Assigner cette classe et avancer
   ← (Flèche G.)   → Revenir à l'image précédente
   S                → Sauvegarder immédiatement
   Q / Echap        → Sauvegarder et quitter

 CLASSES :
   0 = Propre       – Eau claire, situation normale
   1 = Coloration   – Eau anormalement colorée par un rejet
   2 = Limon        – Eau trouble, boue ou terre en suspension
   3 = Mousse       – Présence d'écume blanche en surface
   4 = Irisation    – Taches d'huile / hydrocarbures (arc-en-ciel)

 SORTIE :
   CSV     → ground_truth/ground_truth.csv
   Images  → ground_truth/<classe>/
==============================================================================
"""

import os
import csv
import shutil
import random
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data")
CSV_IN           = os.path.join(BASE_DIR, "dataset_complet.csv")
OUT_DIR          = os.path.join(BASE_DIR, "ground_truth")
CSV_OUT          = os.path.join(OUT_DIR, "ground_truth.csv")
TARGET_PER_CLASS = 200
MAX_IMG_SIZE     = 1000

CLASS_LABELS = {
    0: "Propre",
    1: "Coloration",
    2: "Limon",
    3: "Mousse",
    4: "Irisation",
}
CLASS_DESC = {
    0: "Eau claire, situation normale",
    1: "Eau anormalement colorée par un rejet",
    2: "Eau trouble, boue ou terre en suspension",
    3: "Présence d'écume blanche en surface",
    4: "Taches d'huile / hydrocarbures (reflets arc-en-ciel)",
}
CLASS_EMOJI = {0: "💧", 1: "🟠", 2: "🟫", 3: "🫧", 4: "🌈"}

# Couleurs très saturées pour les boutons (fond foncé + texte blanc)
COLOR_BG  = {0: "#1a7a4a", 1: "#c0580a", 2: "#7a3e18", 3: "#1055b0", 4: "#6a1b9a"}
COLOR_HOV = {0: "#22a060", 1: "#e06a12", 2: "#a05020", 3: "#1a6ee0", 4: "#8e2cb8"}


# ─────────────────────────────────────────────
# HELPERS — données globales
# ─────────────────────────────────────────────
def load_dataset():
    # Load hints from CSV if they exist to prioritize them
    csv_hints = {}
    if os.path.isfile(CSV_IN):
        with open(CSV_IN, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Classe", "").isdigit():
                    csv_hints[row["Nom_Image"]] = int(row["Classe"])
                    
    rows = []
    for name in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, name)
        if os.path.isfile(path) and name.lower().endswith(('.png', '.jpg', '.jpeg')):
            rows.append({
                "name": name, 
                "path": path, 
                "orig_class": csv_hints.get(name, -1)
            })
    return rows


def load_labeled():
    labeled = {}
    if os.path.isfile(CSV_OUT):
        with open(CSV_OUT, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labeled[row["Nom_Image"]] = int(row["Label"])
    return labeled


def build_queue_for_class(rows, labeled, target_class, target=TARGET_PER_CLASS):
    """Retourne (pool_candidats, nb_déjà_validées). Priorise les orig_class = target_class, puis le reste."""
    already    = sum(1 for v in labeled.values() if v == target_class)
    
    candidates_prio = [r for r in rows if r["orig_class"] == target_class and r["name"] not in labeled]
    candidates_other = [r for r in rows if r["orig_class"] != target_class and r["name"] not in labeled]
    
    random.shuffle(candidates_prio)
    random.shuffle(candidates_other)
    
    candidates = candidates_prio + candidates_other
    return candidates, already


def make_btn(parent, text, bg, hov, cmd, fill_x=False, width=None):
    """Bouton via tk.Label — seul moyen d'avoir des couleurs sur macOS."""
    kw = dict(text=text, bg=bg, fg="white",
              font=("Helvetica", 13, "bold"),
              relief="flat", cursor="hand2",
              padx=16, pady=10, anchor="center")
    if width:
        kw["width"] = width
    lbl = tk.Label(parent, **kw)
    lbl.bind("<Button-1>", lambda e: cmd())
    lbl.bind("<Enter>",    lambda e: lbl.config(bg=hov))
    lbl.bind("<Leave>",    lambda e: lbl.config(bg=bg))
    if fill_x:
        lbl.pack(fill="x", ipady=2)
    return lbl


# ─────────────────────────────────────────────
# ÉCRAN 1 — Sélection de classe
# ─────────────────────────────────────────────
class SelectorScreen(tk.Frame):
    def __init__(self, master):
        super().__init__(master, bg="#1a1a2e")
        self.pack(fill="both", expand=True)
        self._build()

    def _build(self):
        tk.Label(self, text="GROUND TRUTH LABELER",
                 font=("Helvetica", 22, "bold"), fg="#e94560", bg="#1a1a2e"
                 ).pack(pady=(36, 6))
        tk.Label(self, text="Sélectionnez une classe à annoter",
                 font=("Helvetica", 13), fg="#a0a0c0", bg="#1a1a2e"
                 ).pack(pady=(0, 24))

        for c in sorted(CLASS_LABELS.keys()):
            already  = sum(1 for v in LABELED.values() if v == c)
            
            # On affiche (déjà validés / cible)
            tgt      = TARGET_PER_CLASS
            done_str = f"✓ {already}/{tgt}" if already >= tgt else f"{already}/{tgt}"

            frame = tk.Frame(self, bg="#1a1a2e")
            frame.pack(fill="x", padx=80, pady=5)

            label_text = (
                f"  {CLASS_EMOJI[c]}  [{c}]  {CLASS_LABELS[c].upper()}"
                f"  —  {CLASS_DESC[c]}  ({done_str})  "
            )
            btn = make_btn(frame, label_text,
                           COLOR_BG[c], COLOR_HOV[c],
                           cmd=lambda cl=c: self.master._start(cl),
                           fill_x=True)

        tk.Label(self, text="Q / Echap = quitter",
                 font=("Helvetica", 9), fg="#505070", bg="#1a1a2e"
                 ).pack(pady=(28, 10))


# ─────────────────────────────────────────────
# ÉCRAN 2 — Annotation
# ─────────────────────────────────────────────
class AnnotationScreen(tk.Frame):
    def __init__(self, master, target_class):
        super().__init__(master, bg="#1a1a2e")
        self.pack(fill="both", expand=True)

        self.cls     = target_class
        self.pool, self.already = build_queue_for_class(ALL_ROWS, LABELED, target_class)
        # queue = images actuellement présentées, pool = réserve restante
        self.queue   = [self.pool.pop(0)] if self.pool else []
        self.validated_in_target = self.already  # nb réellement assignées à target_class
        self.idx     = 0
        self.history = []
        self.col     = COLOR_BG[target_class]
        self.col_hov = COLOR_HOV[target_class]

        self._build()
        self._show()
        self._bind()

    def _build(self):
        # ── Barre du haut
        top = tk.Frame(self, bg="#16213e")
        top.pack(fill="x", padx=6, pady=(8, 0))

        back = tk.Label(top, text="← Liste des classes",
                        font=("Helvetica", 11), fg="#a0c4ff", bg="#16213e",
                        cursor="hand2")
        back.pack(side="left", padx=12)
        back.bind("<Button-1>", lambda e: self._back_to_selector())
        back.bind("<Enter>",    lambda e: back.config(fg="white"))
        back.bind("<Leave>",    lambda e: back.config(fg="#a0c4ff"))

        self.lbl_prog = tk.Label(top, text="", font=("Helvetica", 11),
                                  fg="#c0c0d0", bg="#16213e")
        self.lbl_prog.pack(side="right", padx=12)

        # ── Bandeau classe courante
        bar = tk.Frame(self, bg=self.col)
        bar.pack(fill="x", padx=6, pady=4)
        tk.Label(bar,
                 text=f"{CLASS_EMOJI[self.cls]}  [{self.cls}] {CLASS_LABELS[self.cls].upper()}  —  {CLASS_DESC[self.cls]}",
                 font=("Helvetica", 13, "bold"), fg="white", bg=self.col, pady=7
                 ).pack()

        # ── Canvas image
        img_wrap = tk.Frame(self, bg="#0a0a14", bd=0)
        img_wrap.pack(padx=10, pady=6, fill="both", expand=True)
        self.canvas = tk.Canvas(img_wrap, bg="#0a0a14", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        # ── Info fichier + classe CSV
        self.lbl_name  = tk.Label(self, text="", font=("Courier", 10),
                                   fg="#8080a0", bg="#1a1a2e", anchor="w")
        self.lbl_name.pack(fill="x", padx=14, pady=(2, 0))

        self.lbl_class = tk.Label(self, text="", font=("Helvetica", 13, "bold"),
                                   fg="white", bg="#1a1a2e")
        self.lbl_class.pack(pady=2)

        # ── Titre section boutons
        sep = tk.Frame(self, bg="#2a2a4e", height=2)
        sep.pack(fill="x", padx=10, pady=(4, 2))
        tk.Label(self,
                 text="CORRIGER LA CLASSE SI NÉCESSAIRE :",
                 font=("Helvetica", 10, "bold"), fg="#d0d0e0", bg="#1a1a2e"
                 ).pack(pady=(2, 4))

        # ── Boutons d'assignation — GRAND et COLORÉS
        btn_row = tk.Frame(self, bg="#1a1a2e")
        btn_row.pack(padx=10, pady=4, fill="x")
        for c in sorted(CLASS_LABELS.keys()):
            b = make_btn(btn_row,
                         f"{CLASS_EMOJI[c]}\n[{c}] {CLASS_LABELS[c]}",
                         COLOR_BG[c], COLOR_HOV[c],
                         cmd=lambda cl=c: self._assign(cl),
                         width=12)
            b.pack(side="left", padx=5, pady=4, fill="x", expand=True)

        # ── Barre basse : précédent + accepter
        bot = tk.Frame(self, bg="#14142a")
        bot.pack(fill="x", padx=6, pady=(4, 10))

        make_btn(bot, "← Précédent", "#2d2d55", "#3d3d70",
                 self._go_back).pack(side="left", padx=8)

        make_btn(bot,
                 f"✔  Accepter classe CSV  [{self.cls} = {CLASS_LABELS[self.cls]}]  [Entrée]",
                 self.col, self.col_hov,
                 self._accept).pack(side="left", padx=8)

        tk.Label(bot, text="0-4 = assigner | S = sauver | Q = quitter",
                 font=("Helvetica", 9), fg="#505070", bg="#14142a"
                 ).pack(side="right", padx=10)

    def _show(self):
        tgt_avail = self.already + len(self.pool) + len(self.queue) - self.idx
        need      = TARGET_PER_CLASS - self.validated_in_target

        # Cas : objectif atteint
        if self.validated_in_target >= TARGET_PER_CLASS:
            self.master._save()
            messagebox.showinfo("Classe terminée ✓",
                f"Classe [{self.cls}] {CLASS_LABELS[self.cls]} \u2014 {TARGET_PER_CLASS} images validées !\n"
                f"Retour à la sélection.")
            self._back_to_selector()
            return

        # Cas : plus d'images disponibles dans le pool
        if self.idx >= len(self.queue) and not self.pool:
            self.master._save()
            messagebox.showwarning("Pool épuisé",
                f"Plus d'images disponibles pour la classe [{self.cls}].\n"
                f"{self.validated_in_target}/{TARGET_PER_CLASS} validées.\nRetour à la sélection.")
            self._back_to_selector()
            return

        # Si la queue est épuisée mais le pool a encore des images, compléter
        while self.idx >= len(self.queue) and self.pool:
            self.queue.append(self.pool.pop(0))

        # Charger la prochaine image depuis la queue
        item = self.queue[self.idx]
        img  = Image.open(item["path"]).convert("RGB")
        w, h = img.size
        scale = min(MAX_IMG_SIZE / w, MAX_IMG_SIZE / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        disp  = img.resize((nw, nh), Image.LANCZOS)

        self._photo = ImageTk.PhotoImage(disp)
        self.canvas.config(width=nw, height=nh)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # Bandeau classe CSV sur l'image
        orig = item["orig_class"]
        cbg  = COLOR_BG[orig]
        self.canvas.create_rectangle(0, nh - 46, nw, nh, fill=cbg, outline="")
        self.canvas.create_text(nw // 2, nh - 30,
                                text=f"CSV : {CLASS_EMOJI[orig]} [{orig}] {CLASS_LABELS[orig]}",
                                font=("Helvetica", 12, "bold"), fill="white")
        self.canvas.create_text(nw // 2, nh - 12,
                                text=CLASS_DESC[orig],
                                font=("Helvetica", 9), fill="#e0e0e0")

        self.lbl_name.config(text=f"\U0001f4c1  {item['name']}")
        self.lbl_class.config(
            text=f"Classe CSV : {CLASS_EMOJI[orig]} [{orig}] {CLASS_LABELS[orig]}  \u2014  {CLASS_DESC[orig]}",
            fg=cbg)
        self.lbl_prog.config(
            text=f"Validées : {self.validated_in_target}/{TARGET_PER_CLASS}   •   "
                 f"Image {self.idx + 1}/{len(self.queue)}   •   Réserve : {len(self.pool)}")

    def _assign(self, label):
        # Guard : ne rien faire si on est hors queue (ex : clic après fin de session)
        if self.idx >= len(self.queue):
            return
        item = self.queue[self.idx]
        LABELED[item["name"]] = label
        dst = os.path.join(OUT_DIR, str(label))
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(item["path"], os.path.join(dst, item["name"]))
        self.history.append((self.idx, item["name"], label))

        # Si l'image est assignée à la bonne classe, on compte
        if label == self.cls:
            self.validated_in_target += 1
        else:
            # Image redirigée vers une autre classe → on compense depuis la réserve
            if self.pool:
                self.queue.append(self.pool.pop(0))

        self.idx += 1
        self._show()

    def _accept(self):
        if self.idx < len(self.queue):
            self._assign(self.queue[self.idx]["orig_class"])

    def _go_back(self):
        if not self.history:
            return
        prev_idx, name, label = self.history.pop()

        # Annuler le compteur si l'image était bien dans la classe cible
        if label == self.cls:
            self.validated_in_target = max(0, self.validated_in_target - 1)
        else:
            # On avait ajouté une image depuis la réserve, on la remet
            if len(self.queue) > prev_idx + 1:
                self.pool.insert(0, self.queue.pop())

        if name in LABELED:
            del LABELED[name]
        dst = os.path.join(OUT_DIR, str(label), name)
        if os.path.isfile(dst):
            os.remove(dst)
        self.idx = prev_idx
        self._show()

    def _back_to_selector(self):
        self.master._save()
        self.destroy()
        self.master._show_selector()

    def _bind(self):
        w = self.master
        w.bind("<Return>", lambda e: self._accept())
        w.bind("<space>",  lambda e: self._accept())
        w.bind("<Left>",   lambda e: self._go_back())
        w.bind("s",        lambda e: self.master._save())
        w.bind("S",        lambda e: self.master._save())
        w.bind("q",        lambda e: self.master._quit())
        w.bind("Q",        lambda e: self.master._quit())
        w.bind("<Escape>", lambda e: self.master._quit())
        for c in CLASS_LABELS:
            w.bind(str(c), lambda e, cl=c: self._assign(cl))


# ─────────────────────────────────────────────
# FENÊTRE PRINCIPALE
# ─────────────────────────────────────────────
class LabelerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Ground Truth Labeler — Water Pollution")
        self.configure(bg="#1a1a2e")
        self.resizable(True, True)
        self.minsize(880, 680)
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self._screen = None
        self._show_selector()

    def _show_selector(self):
        if self._screen:
            self._screen.destroy()
        # Reset bindings
        for k in ("<Return>","<space>","<Left>","s","S","q","Q","<Escape>",
                  "0","1","2","3","4"):
            self.unbind(k)
        self.bind("q",        lambda e: self._quit())
        self.bind("Q",        lambda e: self._quit())
        self.bind("<Escape>", lambda e: self._quit())
        self._screen = SelectorScreen(self)

    def _start(self, target_class):
        queue, already = build_queue_for_class(ALL_ROWS, LABELED, target_class)
        tgt = min(TARGET_PER_CLASS,
                  already + sum(1 for r in ALL_ROWS if r["orig_class"] == target_class))
        if already >= tgt:
            messagebox.showinfo("Classe complète ✓",
                f"[{target_class}] {CLASS_LABELS[target_class]} déjà complète ({already}/{tgt}).")
            return
        if not queue:
            messagebox.showwarning("Aucune image",
                f"Aucune image dispo pour la classe [{target_class}].")
            return
        if self._screen:
            self._screen.destroy()
        self._screen = AnnotationScreen(self, target_class)

    def _save(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Nom_Image", "Label"])
            for name, label in LABELED.items():
                writer.writerow([name, label])
        print(f"💾 Sauvegardé → {CSV_OUT}  ({len(LABELED)} images)")

    def _quit(self):
        self._save()
        self.destroy()


# ─────────────────────────────────────────────
# POINT D'ENTRÉE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ground Truth Labeler")
    parser.add_argument("--target", type=int, default=TARGET_PER_CLASS, help=f"Nombre d'images cible à labelliser par classe (défaut: {TARGET_PER_CLASS})")
    args = parser.parse_args()
    
    TARGET_PER_CLASS = args.target
    
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"📂 Base dir : {BASE_DIR}")
    print(f"🎯 Cible configurée à : {TARGET_PER_CLASS} images / classe")
    print("⏳ Chargement du dataset...")
    ALL_ROWS = load_dataset()
    LABELED  = load_labeled()
    print(f"✅ {len(ALL_ROWS)} images — {len(LABELED)} déjà validées")
    LabelerApp().mainloop()
