import os
import sys
import warnings
import threading
import csv

# --- SUPPRESSION DES AVERTISSEMENTS FLET 0.80+ ---
warnings.filterwarnings("ignore", category=DeprecationWarning)

import flet as ft
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat

# --- Empêche le crash macOS ---
if sys.stdout is None: sys.stdout = open(os.devnull, "w")
if sys.stderr is None: sys.stderr = open(os.devnull, "w")

# --- Détection du dossier racine ---
if getattr(sys, 'frozen', False):
    exe_path = sys.argv[0]
    BASE_DIR = os.path.abspath(os.path.join(exe_path, "../../../../..")) if "Contents/MacOS" in exe_path else os.path.abspath(os.path.dirname(exe_path))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)

try:
    from pipeline.train_grl import WaterPollutionGRL, get_transforms
except ImportError:
    print("Erreur : Impossible de trouver pipeline/train_grl.py")

MODEL_PATH = os.path.join(BASE_DIR, "models", "grl", "no_mask", "no_grl", "train_all", "best_grl_model.pth")

# --- PALETTE DE COULEURS HEX ---
COLOR_BG = "#020617"
COLOR_PANEL = "#801e293b"      # #1e293b avec 50% opacité
COLOR_BORDER = "#1Affffff"     # Blanc avec 10% opacité
COLOR_TEXT_MAIN = "#f8fafc"
COLOR_TEXT_MUTED = "#94a3b8"
COLOR_CLEAN = "#06b6d4"        # Cyan
COLOR_POLLUTED = "#f97316"     # Orange

class WaterWatcherApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.datapoints = []
        self.model = None
        self.val_transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    def initialize(self):
        # 1. Configuration de la page
        self.page.title = "WaterWatcher - Prototype"
        self.page.bgcolor = COLOR_BG
        self.page.theme_mode = ft.ThemeMode.DARK  # <-- Force le mode sombre natif
        self.page.padding = 20
        self.page.window_width = 1200
        self.page.window_height = 900
        self.page.scroll = ft.ScrollMode.AUTO
        self.page.fonts = {"Inter": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"}
        self.page.theme = ft.Theme(font_family="Inter")

        # --- Boutons Custom ---
        self.btn_load = ft.Container(
            content=ft.Row([ft.Icon("folder_open", color="#0f172a", size=20), ft.Text("Importer un dossier", color="#0f172a", weight=ft.FontWeight.W_600)]),
            bgcolor=COLOR_CLEAN, padding=12, border_radius=8,
            on_click=self.on_load_clicked, ink=True
        )
        
        self.btn_export = ft.Container(
            content=ft.Row([ft.Icon("save", color=COLOR_TEXT_MAIN, size=20), ft.Text("Sauvegarder", color=COLOR_TEXT_MAIN, weight=ft.FontWeight.W_600)]),
            bgcolor="#1Affffff", border=ft.border.all(1, COLOR_BORDER),
            padding=12, border_radius=8,
            on_click=self.on_export_clicked, opacity=0.5, ink=True
        )

        # 2. HEADER (Plus de blur ici !)
        self.header = ft.Container(
            bgcolor=COLOR_PANEL, border=ft.border.all(1, COLOR_BORDER), border_radius=12,
            padding=20,
            content=ft.Row(
                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                controls=[
                    ft.Row([
                        ft.Text("💧", size=24),
                        ft.Text("Classification des eaux", size=20, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MAIN),
                        ft.Container(content=ft.Text("PROTO", size=10, color="#0f172a", weight=ft.FontWeight.BOLD), bgcolor=COLOR_CLEAN, padding=4, border_radius=4)
                    ]),
                    ft.Row([self.btn_load, self.btn_export], spacing=12)
                ]
            )
        )

        # 3. EMPTY STATE
        self.empty_state = ft.Container(
            content=ft.Text("Aucune séquence chargée. Importez vos images pour lancer l'analyse.", size=18, color=COLOR_TEXT_MUTED, weight=ft.FontWeight.W_500),
            alignment=ft.Alignment(0, 0), padding=80
        )

        # --- UI DE CHARGEMENT ---
        self.loading_ui = ft.Container(
            content=ft.Column([
                ft.ProgressRing(color=COLOR_CLEAN),
                ft.Text("Évaluation IA en cours...", size=20, color=COLOR_TEXT_MAIN, weight=ft.FontWeight.BOLD),
                ft.Text("Veuillez patienter.", color=COLOR_TEXT_MUTED)
            ], horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            alignment=ft.Alignment(0, 0), visible=False, expand=True
        )

        # 4. CHART SECTION (Plus de blur ici !)
        self.chart_wrapper = ft.Container(height=300)
        self.range_slider_container = ft.Container(padding=20, visible=False)
        
        self.chart_section = ft.Container(
            bgcolor=COLOR_PANEL, border=ft.border.all(1, COLOR_BORDER), border_radius=12,
            padding=20, visible=False,
            content=ft.Column([
                ft.Row([
                    ft.Text("Confiance du Modèle & Événements", size=18, weight=ft.FontWeight.BOLD, color=COLOR_TEXT_MAIN),
                    ft.Container(content=ft.Text("Fond Orange = Pollution Détectée", size=12, color=COLOR_POLLUTED, weight=ft.FontWeight.BOLD), bgcolor="#33f97316", padding=6, border_radius=4)
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                self.chart_wrapper,
                self.range_slider_container
            ])
        )

        # 5. GRID SECTION (Plus de blur ici !)
        self.grid_title = ft.Text("Zone sélectionnée", size=18, weight=ft.FontWeight.BOLD, color=COLOR_CLEAN)
        self.images_grid = ft.GridView(expand=1, runs_count=2, max_extent=600, child_aspect_ratio=16/9, spacing=24, run_spacing=24)
        
        self.grid_section = ft.Container(
            bgcolor=COLOR_PANEL, border=ft.border.all(1, COLOR_BORDER), border_radius=12,
            padding=20, visible=False, expand=True,
            content=ft.Column([
                self.grid_title,
                ft.Text("💡 Cliquez sur n'importe quelle image ci-dessous pour forcer ou annuler une pollution.", size=13, color=COLOR_TEXT_MUTED),
                self.images_grid
            ], expand=True)
        )

        # Assemblage final
        self.page.add(
            self.header,
            self.empty_state,
            self.loading_ui,
            self.chart_section,
            self.grid_section
        )
        
        threading.Thread(target=self.load_model_worker, daemon=True).start()

    def load_model_worker(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = WaterPollutionGRL(num_domains=1, num_classes=2, backbone='efficientnet_v2_m', use_grl=False)
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
                self.model.to(self.device)
                self.model.eval()
                _, self.val_transform = get_transforms(scope="no_mask")
                print("Modèle chargé avec succès.")
        except Exception as e:
            print(f"Erreur chargement IA : {e}")

    async def on_load_clicked(self, e):
        files = await ft.FilePicker().pick_files(allow_multiple=True)
        if not files: return
            
        self.datapoints.clear()
        self.empty_state.visible = False
        self.chart_section.visible = False
        self.grid_section.visible = False
        self.loading_ui.visible = True
        self.btn_load.opacity = 0.5
        self.page.update()
        
        threading.Thread(target=self.inference_worker, args=(files,), daemon=True).start()

    def inference_worker(self, files):
        for idx, ffile in enumerate(files):
            try:
                img = Image.open(ffile.path).convert("RGB")
                if ImageStat.Stat(img.convert("L")).mean[0] >= 40:
                    input_tensor = self.val_transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        score = F.softmax(outputs, dim=1)[0][1].item()
                        label = 1 if score >= 0.5 else 0
                    
                    self.datapoints.append({"name": ffile.name, "path": ffile.path, "score": score, "label": label})
            except Exception as ex:
                pass
        
        self.datapoints.sort(key=lambda d: d["path"])
        self.loading_ui.visible = False
        self.btn_load.opacity = 1
        self.btn_export.opacity = 1
        self.btn_export.disabled = False
        
        if len(self.datapoints) > 0:
            self.chart_section.visible = True
            self.build_chart_and_slider()
            self.update_grid(0, max(0, len(self.datapoints) - 1))
        else:
            self.empty_state.visible = True
            
        self.page.update()

    async def on_export_clicked(self, e):
        if len(self.datapoints) == 0: return
        path = await ft.FilePicker().save_file(allowed_extensions=['csv'])
        if path:
            if not path.endswith('.csv'): path += ".csv"
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Fichier", "Pollué (1/0)", "Confiance"])
                for dp in self.datapoints:
                    writer.writerow([dp['name'], dp['label'], f"{dp['score']:.4f}"])
            print("Export terminé.")

    def build_chart_and_slider(self):
        max_idx = len(self.datapoints) - 1
        points_clean, points_polluted = [], []
        
        for i, dp in enumerate(self.datapoints):
            score = dp['score'] * 100
            points_clean.append(ft.LineChartDataPoint(i, score))
            if dp['label'] == 1:
                points_polluted.append(ft.LineChartDataPoint(i, score))

        chart = ft.LineChart(
            data_series=[
                ft.LineChartData(data_points=points_clean, color=COLOR_CLEAN, stroke_width=2, curved=True),
                ft.LineChartData(data_points=points_polluted, color=COLOR_POLLUTED, stroke_width=0, point=True)
            ],
            border=ft.border.all(1, COLOR_BORDER),
            min_y=0, max_y=105, expand=True,
            bottom_axis=ft.ChartAxis(labels=[]),
            left_axis=ft.ChartAxis(labels_size=40)
        )
        
        self.chart_wrapper.content = chart
        self.slider = ft.RangeSlider(
            min=0, max=max_idx, start_value=0, end_value=max_idx, active_color=COLOR_CLEAN, inactive_color=COLOR_BORDER,
            on_change=lambda e: self.update_grid(int(e.control.start_value), int(e.control.end_value))
        )
        self.range_slider_container.content = self.slider
        self.range_slider_container.visible = True

    def update_grid(self, start, end):
        subset = self.datapoints[start : end + 1]
        self.grid_title.value = f"Zone de {start} à {end} : {len(subset)} Image(s)"
        self.images_grid.controls.clear()
        
        for i, dp in enumerate(subset):
            is_polluted = dp['label'] == 1
            color_border = COLOR_POLLUTED if is_polluted else COLOR_CLEAN
            lbl_text = "🟠 POLLUÉ" if is_polluted else "💧 PROPRE"
            
            def make_toggle(idx):
                def toggle(e):
                    self.datapoints[start + idx]['label'] = 1 if self.datapoints[start + idx]['label'] == 0 else 0
                    self.build_chart_and_slider()
                    self.update_grid(int(self.slider.start_value), int(self.slider.end_value))
                return toggle

            card = ft.Container(
                content=ft.Stack([
                    ft.Image(src=dp['path'], fit=ft.ImageFit.COVER, expand=True),
                    ft.Container(
                        content=ft.Text(f"{lbl_text} ({(dp['score']*100):.1f}%)", color=color_border, weight=ft.FontWeight.BOLD, size=12),
                        bgcolor="#b3000000", padding=6,
                        border_radius=6, top=10, left=10
                    )
                ], expand=True),
                border=ft.border.all(4, color_border), border_radius=12, bgcolor="#1e293b",
                on_click=make_toggle(i), ink=True, aspect_ratio=16/9
            )
            self.images_grid.controls.append(card)
        
        self.grid_section.visible = True
        self.page.update()

def main(page: ft.Page):
    app = WaterWatcherApp(page)
    app.initialize()

if __name__ == "__main__":
    ft.run(main)