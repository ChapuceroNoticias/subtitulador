"""
Arranque del Subtitulador (lo abre el acceso directo del escritorio, sin ventana negra).
1) Prepara el entorno (ffmpeg, modelos, tarjeta gráfica).
2) Revisa si hay versión nueva (si se configuró una dirección de actualización).
3) Se asegura de que Ollama esté corriendo.
4) Inicia el servidor (si ya estaba abierto, solo abre el navegador) y abre el navegador.
"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

APP = Path(__file__).resolve().parent
RAIZ = APP.parent
CONFIG = json.loads((RAIZ / "config.json").read_text(encoding="utf-8-sig")) if (RAIZ / "config.json").exists() else {}
PUERTO = int(CONFIG.get("puerto", 7870))
URL = f"http://127.0.0.1:{PUERTO}"
DATOS = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "SubtituladorChapucero"
DATOS.mkdir(parents=True, exist_ok=True)
LOG = DATOS / "arranque.log"
SIN_VENTANA = 0x08000000 if os.name == "nt" else 0


def log(txt):
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + txt + "\n")
    except OSError:
        pass


def aviso(texto, titulo="Subtitulador"):
    log("AVISO: " + texto)
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, texto, titulo, 0x40)
        except Exception:
            pass


def vivo(url, timeout=1.5):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def preparar_entorno():
    ffbin = RAIZ / "ffmpeg" / "bin"
    sr = os.environ.get("SystemRoot", r"C:\Windows")
    sistema = [os.path.join(sr, "System32"), sr, os.path.join(sr, "System32", "Wbem")]  # por si el PATH de la PC está dañado
    os.environ["PATH"] = os.pathsep.join([str(ffbin)] + sistema + [os.environ.get("PATH", "")])
    os.environ["HF_HOME"] = str(RAIZ / "modelos_hf")
    os.environ["HF_HUB_OFFLINE"] = "1"  # el modelo ya se descargó al instalar
    os.environ["PUERTO"] = str(PUERTO)
    if CONFIG.get("whisper_compute"):
        os.environ["WHISPER_COMPUTE"] = CONFIG["whisper_compute"]
    if CONFIG.get("gpu_cuda", True) and CONFIG.get("gpu_nvenc", True):
        os.environ["EDITOR_SOLO_GPU"] = "1"
    if not CONFIG.get("gpu_cuda", True):
        os.environ["WHISPER_COMPUTE"] = ""


def actualizar():
    """Si config.json trae 'url_actualizacion' (carpeta web con version.txt y app.zip), baja la versión nueva."""
    base = (CONFIG.get("url_actualizacion") or "").rstrip("/")
    if not base:
        return
    try:
        actual = (APP / "version.txt").read_text(encoding="utf-8").strip()
        with urllib.request.urlopen(base + "/version.txt", timeout=6) as r:
            nueva = r.read().decode("utf-8").strip()
        if not nueva or nueva == actual:
            return
        log(f"Actualizando {actual} -> {nueva}")
        import zipfile, shutil, tempfile
        tmp = Path(tempfile.mkdtemp())
        urllib.request.urlretrieve(base + "/app.zip", tmp / "app.zip")
        with zipfile.ZipFile(tmp / "app.zip") as z:
            z.extractall(tmp / "nuevo")
        origen = tmp / "nuevo" / "app" if (tmp / "nuevo" / "app").is_dir() else tmp / "nuevo"
        for p in origen.rglob("*"):
            if p.is_file():
                dest = APP / p.relative_to(origen)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        shutil.rmtree(tmp, ignore_errors=True)
        log("Actualización lista")
    except Exception as e:
        log(f"No se pudo actualizar (sigo con la versión instalada): {e}")


def asegurar_ollama():
    if vivo("http://127.0.0.1:11434/api/tags"):
        return
    candidatos = [Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
                  Path(os.environ.get("ProgramFiles", "")) / "Ollama" / "ollama.exe"]
    exe = next((c for c in candidatos if c.exists()), None)
    try:
        subprocess.Popen([str(exe) if exe else "ollama", "serve"], creationflags=SIN_VENTANA,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log("Ollama iniciado")
    except Exception as e:
        log(f"No pude iniciar Ollama: {e} (los subtítulos se hacen igual, sin corrección de nombres)")


def abrir_navegador():
    for _ in range(120):
        if vivo(URL + "/api/vivo"):
            webbrowser.open(URL)
            return
        time.sleep(0.5)
    aviso("El Subtitulador tardó demasiado en arrancar. Revisa el archivo:\n" + str(LOG))


def main():
    if vivo(URL + "/api/vivo"):
        webbrowser.open(URL)  # ya estaba abierto
        return
    preparar_entorno()
    actualizar()
    threading.Thread(target=asegurar_ollama, daemon=True).start()
    threading.Thread(target=abrir_navegador, daemon=True).start()
    # salida de errores a un archivo (pythonw no tiene consola)
    sys.stdout = sys.stderr = open(DATOS / "servidor.log", "a", encoding="utf-8", buffering=1)
    log("Arrancando servidor en " + URL)
    sys.path.insert(0, str(APP))
    try:
        import subtitulador
        subtitulador.main()
    except Exception as e:
        import traceback
        log(traceback.format_exc())
        aviso(f"El Subtitulador no pudo arrancar:\n{e}\n\nDetalles en:\n{LOG}")


if __name__ == "__main__":
    main()
