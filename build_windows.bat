@echo off
echo Construction de l'executable WaterWatcher pour Windows...

:: Activer l'environnement Python avant de lancer :
:: Si vous utilisez venv standard : call venv\Scripts\activate
:: Si vous utilisez conda : call conda activate TM-WP

pyinstaller --noconfirm --windowed --name "WaterWatcher" ^
    --paths "." ^
    --hidden-import "UI.server_2" ^
    --hidden-import "UI.server" ^
    --add-data "UI/index_2.html;UI" ^
    --add-data "UI/style_2.css;UI" ^
    --add-data "UI/app_2.js;UI" ^
    --add-data "pipeline;pipeline" ^
    --add-data "models;models" ^
    --hidden-import "torch" ^
    --hidden-import "torchvision" ^
    --hidden-import "piexif" ^
    --hidden-import "scipy.signal" ^
    standalone/web_wrapper.py

echo Termine! L'executable se trouve dans le dossier "dist".
pause
