import os
import sys

# Empêche le crash silencieux sous MacOS (Mode Windowed / Double-clic)
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import flet as ft
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat
import datetime
import threading
import math
import csv

# --- Détection Robuste du BASE_DIR pour PyInstaller MacOS ---
if getattr(sys, 'frozen', False):
    exe_path = sys.argv[0]
    if "Contents/MacOS" in exe_path:
        # dist/WaterWatcher.app/Contents/MacOS/WaterWatcher -> root
        BASE_DIR = os.path.abspath(os.path.join(exe_path, "../../../../.."))
    else:
        BASE_DIR = os.path.abspath(os.path.dirname(exe_path))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)

print(f"DEBUG: BASE_DIR est résolu à -> {BASE_DIR}")

try:
    from pipeline.train_grl import WaterPollutionGRL, get_transforms

except ImportError:
    print("Attention: le script backend 'train_grl' n'est pas dans le path. Exécution du Flet restreinte.")

# --- Variables Globales ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "grl", "no_mask", "no_grl", "train_all", "best_grl_model.pth")

class WaterWatcherApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "WaterWatcher - Standalone App"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 20
        self.page.scroll = ft.ScrollMode.HIDDEN
        
        # Données chargées
        self.datapoints = []
        self.filtered_datapoints = []
        
        # Modèle IA
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.model = None
        self.val_transform = None
        
        # UI Elements
        self.loading_ring = ft.ProgressRing()
        self.loading_text = ft.Text("Prêt.", size=16, color=ft.Colors.GREY_400)
        
        self.file_picker = ft.FilePicker()
        self.file_picker.on_result = self.on_files_picked
        self.page.overlay.append(self.file_picker)
        
        self.export_save_picker = ft.FilePicker()
        self.export_save_picker.on_result = self.on_save_picked
        self.page.overlay.append(self.export_save_picker)

        # -- COMPOSANTS VISUELS --
        self.btn_load = ft.ElevatedButton("Charger Images", icon="IMAGE_SEARCH", on_click=lambda _: self.file_picker.pick_files(allow_multiple=True))
        self.btn_export = ft.ElevatedButton("Exporter Labels", disabled=True, icon="SAVE_ALT", on_click=lambda _: self.export_save_picker.save_file(allowed_extensions=['csv']))
        
        self.header = ft.Row(
            controls=[
                ft.Row([ft.Icon("WATER_DROP", size=30, color=ft.Colors.CYAN), ft.Text("Classification des Eaux", size=24, weight=ft.FontWeight.BOLD)]),
                ft.Row([self.btn_load, self.btn_export])
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )

        self.chart_container = ft.Container(padding=10, height=350, border_radius=10, bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE))
        
        self.range_slider_container = ft.Container(
            padding=ft.padding.symmetric(horizontal=30, vertical=10),
            visible=False
        )
        
        self.grid_title = ft.Text("", size=20, weight=ft.FontWeight.W_600, color=ft.Colors.CYAN_200)
        self.images_grid = ft.GridView(
            expand=1,
            runs_count=5,
            max_extent=250,
            child_aspect_ratio=0.8,
            spacing=15,
            run_spacing=15,
        )
        self.grid_container = ft.Container(
            content=ft.Column([self.grid_title, self.images_grid], expand=True),
            visible=False,
            expand=True
        )

        # Layout Principal
        self.page.add(
            self.header,
            ft.Row([self.loading_ring, self.loading_text], alignment=ft.MainAxisAlignment.CENTER),
            self.chart_container,
            self.range_slider_container,
            self.grid_container
        )
        
        # Lancer le chargement du modèle en arrière-plan
        self.loading_ring.visible = True
        self.loading_text.value = "Chargement du moteur IA côté processeur..."
        self.page.update()
        threading.Thread(target=self.load_model).start()

    def load_model(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = WaterPollutionGRL(num_domains=1, num_classes=2, backbone='efficientnet_v2_m', use_grl=False)
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
                self.model.to(self.device)
                self.model.eval()
                
                from pipeline.train_grl import get_transforms
                _, self.val_transform = get_transforms(scope="no_mask")
                
                self.loading_text.value = f"IA Prête ({self.device})"
                self.loading_text.color = ft.Colors.GREEN_400
            else:
                self.loading_text.value = "Fichier modèle introuvable :("
                self.loading_text.color = ft.Colors.RED_400
        except Exception as e:
            self.loading_text.value = f"Erreur IA: {e}"
        finally:
            self.loading_ring.visible = False
            self.page.update()

    def extract_datetime(self, filename: str):
        # Ex: "12032024_081520_image.jpg"
        parts = filename.split('_')
        if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
            try:
                date_str = f"{parts[0][:2]}/{parts[0][2:4]}/{parts[0][4:]}"
                time_str = f"{parts[1][:2]}:{parts[1][2:4]}"
                return date_str, time_str
            except:
                pass
        return "Inconnue", "Inconnue"

    def on_files_picked(self, e):
        if not e.files:
            return
            
        self.datapoints.clear()
        self.loading_ring.visible = True
        self.btn_load.disabled = True
        
        def process_files():
            total = len(e.files)
            for idx, ffile in enumerate(e.files):
                self.loading_text.value = f"Inférence IA : {idx + 1} / {total}"
                self.page.update()
                
                name = ffile.name
                path = ffile.path
                date_str, time_str = self.extract_datetime(name)
                
                try:
                    img = Image.open(path).convert("RGB")
                    # Filtre nuit
                    gray = img.convert("L")
                    avg_brightness = ImageStat.Stat(gray).mean[0]
                    
                    if avg_brightness < 40:
                        continue # Saute les images de nuit
                        
                    input_tensor = self.val_transform(img).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_tensor)
                        probs = F.softmax(outputs, dim=1)
                        score_polluted = probs[0][1].item()
                        label = 1 if score_polluted >= 0.5 else 0
                        
                    self.datapoints.append({
                        "name": name,
                        "path": path,
                        "date": date_str,
                        "time": time_str,
                        "score": score_polluted,
                        "label": label
                    })
                except Exception as ex:
                    print(f"Skipped {name}: {ex}")
            
            # Trie chronologique auto
            self.datapoints.sort(key=lambda d: d["path"]) 
            
            self.loading_ring.visible = False
            self.loading_text.value = f"{len(self.datapoints)} images validées."
            self.btn_load.disabled = False
            self.btn_export.disabled = False
            
            # Setup UI components
            self.build_chart_and_slider()
            self.update_grid(0, len(self.datapoints) - 1)
            self.page.update()
        
        threading.Thread(target=process_files).start()

    def build_chart_and_slider(self):
        if not self.datapoints:
            return
            
        max_idx = len(self.datapoints) - 1
        
        # ---- CONSTRUCTION CHART ----
        points_clean = []
        points_polluted = []
        
        for i, dp in enumerate(self.datapoints):
            score = dp['score'] * 100
            if dp['label'] == 1:
                points_polluted.append(ft.LineChartDataPoint(i, score))
                points_clean.append(ft.LineChartDataPoint(i, score)) # relier la ligne
            else:
                points_clean.append(ft.LineChartDataPoint(i, score))

        data_series = [
            ft.LineChartData(
                data_points=points_clean,
                stroke_width=2,
                color=ft.Colors.CYAN_400,
                curved=True,
                below_line_bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.CYAN_400)
            ),
            ft.LineChartData(
                data_points=points_polluted,
                stroke_width=0, # pas de ligne, que des points
                point=True,
                color=ft.Colors.ORANGE_500,
            )
        ]

        chart = ft.LineChart(
            data_series=data_series,
            border=ft.Border(bottom=ft.BorderSide(1, ft.Colors.WHITE24), left=ft.BorderSide(1, ft.Colors.WHITE24)),
            min_y=0, max_y=105, min_x=0, max_x=max_idx,
            expand=True,
            tooltip_bgcolor=ft.Colors.BLUE_GREY_900
        )
        
        self.chart_container.content = chart

        # ---- SLIDER RANGE ----
        self.slider = ft.RangeSlider(
            min=0, max=max_idx, 
            start_value=0, end_value=max_idx, 
            divisions=max_idx if max_idx > 0 else 1,
            label_position=ft.SliderLabelPosition.TOP,
            on_change=lambda e: self.update_grid(int(e.control.start_value), int(e.control.end_value))
        )
        self.range_slider_container.content = self.slider
        self.range_slider_container.visible = True

    def update_grid(self, start_idx, end_idx):
        if not self.datapoints:
            return
            
        subset = self.datapoints[start_idx : end_idx + 1]
        self.grid_title.value = f"Zone de {start_idx} à {end_idx} : {len(subset)} Image(s)"
        
        self.images_grid.controls.clear()
        
        for i, dp in enumerate(subset):
            is_polluted = dp['label'] == 1
            color_bg = ft.Colors.ORANGE_900 if is_polluted else ft.Colors.BLUE_GREY_900
            color_text = ft.Colors.ORANGE_200 if is_polluted else ft.Colors.CYAN_200
            icon_status = "🟠 POLLUÉ" if is_polluted else "💧 PROPRE"
            
            # Fonction locale pour capturer la variable
            def make_toggle(true_index):
                def toggle(e):
                    # Inverser le statut
                    self.datapoints[true_index]['label'] = 1 if self.datapoints[true_index]['label'] == 0 else 0
                    # Redessiner le range actuel
                    self.update_grid(int(self.slider.start_value), int(self.slider.end_value))
                    self.build_chart_and_slider() # maj graphique
                    self.page.update()
                return toggle
                
            true_idx = start_idx + i
            
            card = ft.Container(
                content=ft.Column([
                    ft.Image(src=dp['path'], fit=ft.ImageFit.COVER, expand=True, border_radius=5),
                    ft.Container(
                        ft.Text(f"{icon_status}  {(dp['score']*100):.1f}%", color=color_text, weight=ft.FontWeight.BOLD),
                        padding=5, bgcolor=color_bg, border_radius=5, alignment=ft.alignment.center
                    )
                ], spacing=5),
                bgcolor=ft.Colors.with_opacity(0.1, color_bg),
                border=ft.border.all(2, color_bg),
                border_radius=8,
                padding=5,
                on_click=make_toggle(true_idx),
                tooltip=dp['name']
            )
            self.images_grid.controls.append(card)
            
        self.grid_container.visible = True
        self.page.update()

    def on_save_picked(self, e):
        if e.path:
            with open(e.path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Nom Fichier", "Label", "Confiance Modèle", "Validation Humaine"])
                for dp in self.datapoints:
                    writer.writerow([dp['name'], dp['label'], f"{dp['score']:.4f}", dp['label']])
            self.loading_text.value = f"Sauvegardé avec succès dans: {e.path}"
            self.loading_text.color = ft.Colors.GREEN_400
            self.page.update()

def main(page: ft.Page):
    app = WaterWatcherApp(page)

if __name__ == "__main__":
    ft.app(target=main)
