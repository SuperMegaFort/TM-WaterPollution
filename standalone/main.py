import os
import sys
import asyncio
import flet as ft
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat
import threading
import csv

# Empêche le crash silencieux sous MacOS (Mode Windowed / Double-clic)
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

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

try:
    from pipeline.train_grl import WaterPollutionGRL, get_transforms
except ImportError:
    print("Attention: le script backend 'train_grl' n'est pas dans le path.")

# --- Variables Globales ---
MODEL_PATH = os.path.join(BASE_DIR, "models", "grl", "no_mask", "no_grl", "train_all", "best_grl_model.pth")

class WaterWatcherApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.datapoints = []
        self.model = None
        self.val_transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    async def initialize(self):
        # Configuration Page
        self.page.title = "WaterWatcher - Standalone App (Async)"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.padding = 20
        self.page.scroll = ft.ScrollMode.HIDDEN
        
        # File Pickers (Services - Ne pas ajouter à l'overlay pour éviter l'erreur "Unknown control")
        self.file_picker = ft.FilePicker()
        self.export_save_picker = ft.FilePicker()
        
        # UI Elements
        self.loading_ring = ft.ProgressRing(visible=False)
        self.loading_text = ft.Text("Prêt.", size=16, color=ft.Colors.GREY_400)
        
        self.btn_load = ft.ElevatedButton(
            "Charger Images", 
            icon="IMAGE_SEARCH", 
            on_click=self.on_load_clicked
        )
        self.btn_export = ft.ElevatedButton(
            "Exporter Labels", 
            disabled=True, 
            icon="SAVE_ALT", 
            on_click=self.on_export_clicked
        )
        
        self.header = ft.Row(
            controls=[
                ft.Row([ft.Icon("WATER_DROP", size=30, color=ft.Colors.CYAN), ft.Text("Classification des Eaux", size=24, weight=ft.FontWeight.BOLD)]),
                ft.Row([self.btn_load, self.btn_export])
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN
        )

        self.chart_container = ft.Container(padding=10, height=350, border_radius=10, bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE))
        self.range_slider_container = ft.Container(padding=ft.padding.symmetric(horizontal=30, vertical=10), visible=False)
        
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

        # Ajout du header et loading à la page
        await self.page.add_async(
            self.header,
            ft.Row([self.loading_ring, self.loading_text], alignment=ft.MainAxisAlignment.CENTER),
            self.chart_container,
            self.range_slider_container,
            self.grid_container
        )
        
        # Lancement du chargement du modèle
        self.loading_ring.visible = True
        self.loading_text.value = "Chargement du moteur IA..."
        await self.page.update_async()
        
        # Chargement modèle dans un thread pour ne pas bloquer l'event loop (AI initialization can be heavy)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.load_model_sync)
        
        self.loading_ring.visible = False
        await self.page.update_async()

    def load_model_sync(self):
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

    async def on_load_clicked(self, _):
        # Dans Flet 0.84+, pick_files est asynchrone et renvoie les résultats
        res = await self.file_picker.pick_files(allow_multiple=True)
        if not res or not res.files:
            return
            
        self.datapoints.clear()
        self.loading_ring.visible = True
        self.btn_load.disabled = True
        await self.page.update_async()
        
        total = len(res.files)
        for idx, ffile in enumerate(res.files):
            self.loading_text.value = f"Inférence IA : {idx + 1} / {total}"
            await self.page.update_async()
            
            name = ffile.name
            path = ffile.path
            
            # Extraction Date/Heure
            date_str, time_str = "Inconnue", "Inconnue"
            parts = name.split('_')
            if len(parts) >= 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
                date_str = f"{parts[0][:2]}/{parts[0][2:4]}/{parts[0][4:]}"
                time_str = f"{parts[1][:2]}:{parts[1][2:4]}"
            
            try:
                # Inférence (dans l'executor pour ne pas figer l'UI)
                loop = asyncio.get_running_loop()
                score, label = await loop.run_in_executor(None, self.predict_sync, path)
                
                if score is not None:
                    self.datapoints.append({
                        "name": name,
                        "path": path,
                        "date": date_str,
                        "time": time_str,
                        "score": score,
                        "label": label
                    })
            except Exception as ex:
                print(f"Skipped {name}: {ex}")
        
        self.datapoints.sort(key=lambda d: d["path"])
        self.loading_ring.visible = False
        self.loading_text.value = f"{len(self.datapoints)} images validées."
        self.btn_load.disabled = False
        self.btn_export.disabled = len(self.datapoints) == 0
        
        await self.build_chart_and_slider()
        await self.update_grid(0, len(self.datapoints) - 1)
        await self.page.update_async()

    def predict_sync(self, path):
        try:
            img = Image.open(path).convert("RGB")
            # Filtre nuit
            gray = img.convert("L")
            avg_brightness = ImageStat.Stat(gray).mean[0]
            if avg_brightness < 40:
                return None, None
                
            input_tensor = self.val_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = F.softmax(outputs, dim=1)
                score = probs[0][1].item()
                label = 1 if score >= 0.5 else 0
            return score, label
        except:
            return None, None

    async def on_export_clicked(self, _):
        path = await self.export_save_picker.save_file(allowed_extensions=['csv'])
        if path:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Nom Fichier", "Label", "Confiance Modèle", "Validation Humaine"])
                for dp in self.datapoints:
                    writer.writerow([dp['name'], dp['label'], f"{dp['score']:.4f}", dp['label']])
            self.loading_text.value = f"Sauvegardé avec succès."
            self.loading_text.color = ft.Colors.GREEN_400
            await self.page.update_async()

    async def build_chart_and_slider(self):
        if not self.datapoints:
            return
        
        max_idx = len(self.datapoints) - 1
        points_clean = []
        points_polluted = []
        
        for i, dp in enumerate(self.datapoints):
            score = dp['score'] * 100
            if dp['label'] == 1:
                points_polluted.append(ft.LineChartDataPoint(i, score))
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
                stroke_width=0,
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
        self.slider = ft.RangeSlider(
            min=0, max=max_idx, 
            start_value=0, end_value=max_idx, 
            divisions=max_idx if max_idx > 0 else 1,
            label_position=ft.SliderLabelPosition.TOP,
            on_change=lambda e: self.page.run_task(self.update_grid, int(e.control.start_value), int(e.control.end_value))
        )
        self.range_slider_container.content = self.slider
        self.range_slider_container.visible = True

    async def update_grid(self, start_idx, end_idx):
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
            
            true_idx = start_idx + i
            
            async def toggle_label(e, idx=true_idx):
                self.datapoints[idx]['label'] = 1 if self.datapoints[idx]['label'] == 0 else 0
                await self.build_chart_and_slider()
                await self.update_grid(int(self.slider.start_value), int(self.slider.end_value))
                await self.page.update_async()

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
                on_click=toggle_label,
                tooltip=dp['name']
            )
            self.images_grid.controls.append(card)
            
        self.grid_container.visible = True
        await self.page.update_async()

async def main(page: ft.Page):
    app = WaterWatcherApp(page)
    await app.initialize()

if __name__ == "__main__":
    ft.app(target=main)
