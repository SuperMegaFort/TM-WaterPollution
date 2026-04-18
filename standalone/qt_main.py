import os
import sys
import csv
import threading
from datetime import datetime

# Qt Imports
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QFileDialog, QProgressBar, QScrollArea, 
    QGridLayout, QFrame, QSizePolicy, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSize
from PySide6.QtGui import QPixmap, QIcon, QColor, QPalette

# Matplotlib Imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Deep Learning / Image
import torch
import torch.nn.functional as F
from PIL import Image, ImageStat

# Détection du BASE_DIR
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.abspath(os.path.join(sys.argv[0], "../../../../.."))
else:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(BASE_DIR)

try:
    from pipeline.train_grl import WaterPollutionGRL, get_transforms
except ImportError:
    print("Erreur: Impossible d'importer le pipeline d'IA.")

MODEL_PATH = os.path.join(BASE_DIR, "models", "grl", "no_mask", "no_grl", "train_all", "best_grl_model.pth")

# --- WORKER POUR L'INFÉRENCE IA (Thread séparé) ---
class InferenceWorker(QObject):
    progress = Signal(int, int) # Current, Total
    finished = Signal(list)
    error = Signal(str)

    def __init__(self, file_paths, model, transform, device):
        super().__init__()
        self.file_paths = file_paths
        self.model = model
        self.transform = transform
        self.device = device

    def run(self):
        results = []
        total = len(self.file_paths)
        try:
            for idx, path in enumerate(self.file_paths):
                name = os.path.basename(path)
                
                # Filtre nuit
                img_pil = Image.open(path).convert("RGB")
                gray = img_pil.convert("L")
                brightness = ImageStat.Stat(gray).mean[0]
                
                if brightness < 40:
                    self.progress.emit(idx + 1, total)
                    continue
                
                # Inference
                input_tensor = self.transform(img_pil).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    score = probs[0][1].item()
                    label = 1 if score >= 0.5 else 0
                
                results.append({
                    "name": name,
                    "path": path,
                    "score": score,
                    "label": label
                })
                self.progress.emit(idx + 1, total)
            
            # Tri séquentiel par nom de fichier
            results.sort(key=lambda x: x["path"])
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

# --- WIDGET D'IMAGE INDIVIDUELLE ---
class ImageCard(QFrame):
    clicked = Signal(int)

    def __init__(self, index, data):
        super().__init__()
        self.index = index
        self.data = data
        self.init_ui()

    def init_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)

        # Image
        self.img_label = QLabel()
        self.img_label.setFixedSize(200, 150)
        self.img_label.setScaledContents(True)
        pixmap = QPixmap(self.data["path"])
        self.img_label.setPixmap(pixmap.scaled(200, 150, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        self.layout.addWidget(self.img_label)

        # Labels
        self.status_label = QLabel()
        self.update_style()
        self.layout.addWidget(self.status_label)

    def update_style(self):
        is_polluted = self.data["label"] == 1
        txt = f"POLLUÉ ({(self.data['score']*100):.1f}%)" if is_polluted else f"PROPRE ({(self.data['score']*100):.1f}%)"
        self.status_label.setText(txt)
        
        color = "#e67e22" if is_polluted else "#2ecc71" # Orange vs Green
        self.status_label.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; padding: 5px; border-radius: 3px;")
        self.setStyleSheet(f"ImageCard {{ border: 2px solid {color}; background-color: #2c3e50; }}")

    def mousePressEvent(self, event):
        self.clicked.emit(self.index)

# --- FENÊTRE PRINCIPALE ---
class WaterWatcherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WaterWatcher - Industry Standard (Qt/Matplotlib)")
        self.resize(1200, 900)
        
        self.datapoints = []
        self.model = None
        self.val_transform = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.init_ui()
        self.load_ia_engine()

    def init_ui(self):
        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Header / ToolBar pseudo-style
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        
        title_lib = QLabel("💧 Classification des Eaux")
        title_lib.setStyleSheet("font-size: 20px; font-weight: bold; color: #3498db;")
        header_layout.addWidget(title_lib)
        
        header_layout.addStretch()
        
        self.btn_load = QPushButton(" Charger Images")
        self.btn_load.setIcon(QIcon.fromTheme("image-x-generic"))
        self.btn_load.clicked.connect(self.on_load_images)
        header_layout.addWidget(self.btn_load)
        
        self.btn_export = QPushButton(" Exporter CSV")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.on_export_csv)
        header_layout.addWidget(self.btn_export)
        
        self.main_layout.addWidget(header_widget)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.main_layout.addWidget(self.progress_bar)
        
        # Chart Section (Matplotlib)
        self.canvas = FigureCanvas(Figure(figsize=(5, 3), facecolor='#1e1e1e'))
        self.ax = self.canvas.figure.add_subplot(111)
        self.ax.set_facecolor('#1e1e1e')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_edgecolor('white')
            
        self.main_layout.addWidget(self.canvas, stretch=1)

        # Range Slider Wrapper
        slider_layout = QHBoxLayout()
        self.lbl_range = QLabel("Zone : -")
        self.lbl_range.setStyleSheet("color: #ecf0f1;")
        
        self.slider_min = QSlider(Qt.Horizontal)
        self.slider_max = QSlider(Qt.Horizontal)
        self.slider_min.valueChanged.connect(self.on_slider_change)
        self.slider_max.valueChanged.connect(self.on_slider_change)
        
        slider_layout.addWidget(QLabel("Min:"))
        slider_layout.addWidget(self.slider_min)
        slider_layout.addWidget(QLabel("Max:"))
        slider_layout.addWidget(self.slider_max)
        slider_layout.addWidget(self.lbl_range)
        
        self.main_layout.addLayout(slider_layout)

        # Grid Section
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.grid_content = QWidget()
        self.grid_layout = QGridLayout(self.grid_content)
        self.scroll_area.setWidget(self.grid_content)
        
        self.main_layout.addWidget(self.scroll_area, stretch=2)

    def load_ia_engine(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = WaterPollutionGRL(num_domains=1, num_classes=2, backbone='efficientnet_v2_m', use_grl=False)
                # Utilisation de weights_only par sécurité
                self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
                self.model.to(self.device)
                self.model.eval()
                
                from pipeline.train_grl import get_transforms
                _, self.val_transform = get_transforms(scope="no_mask")
                self.statusBar().showMessage(f"IA Chargée sur {self.device}")
            else:
                self.statusBar().showMessage("Erreur: Modèle introuvable.")
        except Exception as e:
            self.statusBar().showMessage(f"Erreur IA: {e}")

    def on_load_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Sélectionner des images", "", "Images (*.png *.jpg *.jpeg)")
        if not files:
            return
            
        self.btn_load.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(len(files))
        self.progress_bar.setValue(0)
        
        # Threading IA
        self.thread = QThread()
        self.worker = InferenceWorker(files, self.model, self.val_transform, self.device)
        self.worker.moveToThread(self.thread)
        
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(lambda cur, tot: self.progress_bar.setValue(cur))
        self.worker.finished.connect(self.on_inference_finished)
        self.worker.error.connect(lambda err: self.statusBar().showMessage(f"Erreur: {err}"))
        
        self.thread.start()

    def on_inference_finished(self, results):
        self.datapoints = results
        self.thread.quit()
        self.progress_bar.setVisible(False)
        self.btn_load.setEnabled(True)
        self.btn_export.setEnabled(True)
        
        # Setup Sliders
        self.slider_min.setMaximum(len(results) - 1)
        self.slider_max.setMaximum(len(results) - 1)
        self.slider_min.setValue(0)
        self.slider_max.setValue(len(results) - 1)
        
        self.update_chart()
        self.refresh_grid()

    def update_chart(self):
        if not self.datapoints: return
        self.ax.clear()
        
        scores = [d["score"] * 100 for d in self.datapoints]
        indices = list(range(len(scores)))
        
        # Ligne de base
        self.ax.plot(indices, scores, color='#3498db', linewidth=2, label="Confiance")
        
        # Points pollués
        polluted_idx = [i for i, d in enumerate(self.datapoints) if d["label"] == 1]
        polluted_scores = [scores[i] for i in polluted_idx]
        if polluted_idx:
            self.ax.scatter(polluted_idx, polluted_scores, color='#e67e22', zorder=5, label="Pollué")
        
        # Stylisation
        self.ax.set_ylim(0, 105)
        self.ax.set_title("Profil de Pollution IA", color="white")
        self.ax.set_ylabel("Confiance (%)", color="white")
        self.ax.legend()
        
        self.canvas.draw()

    def on_slider_change(self):
        low = min(self.slider_min.value(), self.slider_max.value())
        high = max(self.slider_min.value(), self.slider_max.value())
        self.lbl_range.setText(f"Zone : {low} à {high} ({high - low + 1} images)")
        self.refresh_grid()

    def refresh_grid(self):
        if not self.datapoints: return
        
        # Clear layout
        for i in reversed(range(self.grid_layout.count())): 
            self.grid_layout.itemAt(i).widget().setParent(None)

        low = min(self.slider_min.value(), self.slider_max.value())
        high = max(self.slider_min.value(), self.slider_max.value())
        
        subset = self.datapoints[low : high + 1]
        cols = 5
        for i, data in enumerate(subset):
            true_idx = low + i
            card = ImageCard(true_idx, data)
            card.clicked.connect(self.toggle_label)
            self.grid_layout.addWidget(card, i // cols, i % cols)

    def toggle_label(self, index):
        # Inversion du label
        self.datapoints[index]["label"] = 1 if self.datapoints[index]["label"] == 0 else 0
        self.update_chart()
        self.refresh_grid()

    def on_export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Exporter CSV", "label_export.csv", "CSV (*.csv)")
        if path:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Fichier", "Score_IA", "Label_Final"])
                for d in self.datapoints:
                    writer.writerow([d["name"], f"{d['score']:.4f}", d["label"]])
            self.statusBar().showMessage(f"Exporté: {path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Dark Theme Palette
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(44, 62, 80))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = WaterWatcherWindow()
    window.show()
    sys.exit(app.exec())
