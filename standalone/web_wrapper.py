import os
import sys
import threading
import time
import webview


sys.path.append(os.getcwd())

USE_V2 = True 

if USE_V2:
    from UI_V2.server import app
    APP_TITLE = "WaterPollution V2"
else:
    from UI_V1.server import app
    APP_TITLE = "WaterPollution V1"

class Api:
    def open_folder_dialog(self, title, directory=''):
        try:
            active_window = webview.windows[0]
            result = active_window.create_file_dialog(
                webview.FileDialog.FOLDER, allow_multiple=False, directory=directory, save_filename='', file_types=()
            )
            if result and len(result) > 0:
                return result[0]
            return None
        except Exception as e:
            print("Erreur dialogue dossier:", e)
            return None

def start_flask():
   
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)

if __name__ == '__main__':

    t = threading.Thread(target=start_flask)
    t.daemon = True
    t.start()

    time.sleep(1)

    window = webview.create_window(
        APP_TITLE, 
        'http://127.0.0.1:5000',
        width=1200,
        height=900,
        min_size=(1000, 700),
        confirm_close=True,
        js_api=Api()
    )

    # erreur sur linux entre pywebview et gtk3
    # solution : forcer l'utilisation de qt
    if sys.platform == "linux":
        webview.start(gui='qt')
    else:
        webview.start()
