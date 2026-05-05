@echo off
echo Construction de l'executable WaterWatcher pour Windows...

:: Activer l'environnement Python avant de lancer :
:: Si vous utilisez venv standard : call venv\Scripts\activate
:: Si vous utilisez conda : call conda activate TM-WP

pyinstaller --noconfirm --windowed --name "WaterWatcher" ^
    --paths "." ^
    --hidden-import "UI_V2.server" ^
    --add-data "UI_V2/index.html;UI_V2" ^
    --add-data "UI_V2/style.css;UI_V2" ^
    --add-data "UI_V2/app.js;UI_V2" ^
    --add-data "pipeline;pipeline" ^
    --add-data "standalone/best_model.pth;standalone" ^
    --hidden-import "torch" ^
    --hidden-import "torchvision" ^
    --hidden-import "piexif" ^
    --hidden-import "scipy.signal" ^
    standalone/web_wrapper.py

echo Termine! L'executable se trouve dans le dossier "dist".
pause
