import os
import sys
import threading
import time
import webview

# Assurer que le dossier racine est dans le path pour trouver 'UI'
sys.path.append(os.getcwd())
from UI.server import app

class Api:
    def open_folder_dialog(self, title):
        try:
            active_window = webview.windows[0]
            result = active_window.create_file_dialog(
                webview.FOLDER_DIALOG, allow_multiple=False, directory='', save_filename='', file_types=()
            )
            if result and len(result) > 0:
                return result[0]
            return None
        except Exception as e:
            print("Erreur dialogue dossier:", e)
            return None

def start_flask():
    # On lance Flask sur le port 5000 sans le mode debug pour la stabilité
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':
    # 1. Lancement du serveur Flask dans un thread séparé
    t = threading.Thread(target=start_flask)
    t.daemon = True
    t.start()

    # 2. On attend une seconde que le serveur démarre
    time.sleep(1)

    # 3. Création de la fenêtre native
    # On lui donne un titre, l'URL de localhost, et on active la sélection de fichiers
    window = webview.create_window(
        'WaterWatcher - Inférence IA', 
        'http://127.0.0.1:5000',
        width=1200,
        height=900,
        min_size=(1000, 700),
        confirm_close=True,
        js_api=Api()
    )

    # 4. Démarrage de la boucle d'interface native (WebKit sur Mac)
    webview.start()
