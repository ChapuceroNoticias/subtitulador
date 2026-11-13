"""
Subtitulador · El Chapucero
Pone subtítulos estilo Chapucero a clips ya editados (Clips Chapuceros y Clips Fuera).

- Se abre en el navegador (http://127.0.0.1:PUERTO).
- El usuario elige el clip con la ventana de Windows (la plataforma recuerda su ruta; no hace copia).
- Paso 1: transcribe con Whisper (tarjeta NVIDIA), corrige nombres y puntuación con Ollama y arma los subtítulos
  (en la franja azul cuando el clip tiene pantalla dividida). Se revisan en vivo sobre el clip original.
- Paso 2: «Renderizar y Exportar» hace el video final y lo copia a  Escritorio\\Subtitulados.
- Si el clip cambia en su ruta (se reexportó con ajustes), al abrirlo se usa la versión nueva con los mismos
  subtítulos, sin volver a transcribir.
- Cualquier trabajo se puede cancelar.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

APP = Path(__file__).resolve().parent
RAIZ = APP.parent
sys.path.insert(0, str(APP))

import interfaz as I  # noqa: E402  (rutas /sub/* del editor de subtítulos en vivo)
import editor as ed  # noqa: E402
from fastapi import Body, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

VERSION = (APP / "version.txt").read_text(encoding="utf-8").strip() if (APP / "version.txt").exists() else "dev"
CONFIG = json.loads((RAIZ / "config.json").read_text(encoding="utf-8-sig")) if (RAIZ / "config.json").exists() else {}
DATOS = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "SubtituladorChapucero"
TRABAJOS = DATOS / "trabajos"
TRABAJOS.mkdir(parents=True, exist_ok=True)
SIN_VENTANA = 0x08000000 if os.name == "nt" else 0
EXT_VIDEO = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".mts", ".wmv", ".flv")

# python.exe (no pythonw) con la ventana oculta: así ffmpeg y los demás procesos tampoco abren ventanas negras
PY = Path(sys.executable).with_name("python.exe") if Path(sys.executable).name.lower() == "pythonw.exe" else Path(sys.executable)

ESTILOS = {
    "chapuceros": ("Clips Chapuceros", APP / "plantillas" / "solo_subtitulos_chapucero.json"),
    "fuera": ("Clips Fuera", APP / "plantillas" / "solo_subtitulos_fuera.json"),
}


def escritorio():
    """Carpeta del Escritorio real (también si está en OneDrive)."""
    if os.name == "nt":
        try:
            import ctypes
            buf = ctypes.create_unicode_buffer(260)
            if ctypes.windll.shell32.SHGetFolderPathW(None, 0x10, None, 0, buf) == 0 and buf.value:
                return Path(buf.value)
        except Exception:
            pass
    return Path.home() / "Desktop"


def carpeta_salida():
    c = escritorio() / CONFIG.get("carpeta_salida", "Subtitulados")
    c.mkdir(parents=True, exist_ok=True)
    return c


def leer_json(p, defecto=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8-sig"))
    except Exception:
        return {} if defecto is None else defecto


def escribir_json(p, datos):
    Path(p).write_text(json.dumps(datos, ensure_ascii=False, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------
# Elegir el video con la ventana de Windows (así se conoce su ruta)
# ----------------------------------------------------------------------------
PREFS = DATOS / "preferencias.json"


def elegir_video_windows(titulo="Elige el clip ya editado"):
    """Abre la ventana «Abrir» de Windows y devuelve la ruta elegida ('' si se cancela)."""
    if os.environ.get("SUBT_PRUEBA_ELEGIR") is not None:  # pruebas fuera de Windows
        return os.environ["SUBT_PRUEBA_ELEGIR"]
    inicial = leer_json(PREFS).get("ultima_carpeta", "")
    filtro = "Videos|" + ";".join("*" + e for e in EXT_VIDEO) + "|Todos los archivos|*.*"
    ps = (
        "[Console]::OutputEncoding = [Text.Encoding]::UTF8;"
        "Add-Type -AssemblyName System.Windows.Forms;"
        "$f = New-Object System.Windows.Forms.Form; $f.TopMost = $true; $f.ShowInTaskbar = $false;"
        "$f.StartPosition = 'CenterScreen'; $f.Size = New-Object System.Drawing.Size(1,1); $f.Show(); $f.Activate();"
        "$d = New-Object System.Windows.Forms.OpenFileDialog;"
        f"$d.Title = '{titulo}'; $d.Filter = '{filtro}';"
        + (f"$d.InitialDirectory = '{inicial.replace(chr(39), chr(39) * 2)}';" if inicial else "")
        + "if ($d.ShowDialog($f) -eq 'OK') { [Console]::Out.Write($d.FileName) }; $f.Close()"
    )
    exe = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    r = subprocess.run([str(exe), "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", ps],
                       capture_output=True, creationflags=SIN_VENTANA)
    ruta = r.stdout.decode("utf-8", errors="replace").strip().lstrip("\ufeff")
    if ruta:
        prefs = leer_json(PREFS)
        prefs["ultima_carpeta"] = str(Path(ruta).parent)
        escribir_json(PREFS, prefs)
    return ruta


def huella(ruta):
    """Tamaño y fecha del archivo: si cambian, el video se reexportó."""
    st = Path(ruta).stat()
    return {"tam": st.st_size, "mtime": round(st.st_mtime, 2)}


def mismo_archivo(a, b):
    try:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
    except Exception:
        return False


def trabajo_de_ruta(ruta):
    """El trabajo más reciente hecho con ese mismo video (misma ruta y nombre), si existe."""
    mejores = []
    for t in TRABAJOS.iterdir():
        meta = leer_json(t / "meta.json")
        if meta.get("ruta") and mismo_archivo(meta["ruta"], ruta) and (t / "plan.json").exists():
            mejores.append(t)
    return max(mejores, key=lambda p: p.stat().st_mtime) if mejores else None


# ----------------------------------------------------------------------------
# Trabajos en segundo plano (uno a la vez: la tarjeta gráfica no se reparte) y cancelables
# ----------------------------------------------------------------------------
JOBS = {}
_turno = threading.Lock()
REGISTRO = DATOS / "registro.log"


def anotar(linea):
    try:
        with open(REGISTRO, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S  ") + linea + "\n")
    except OSError:
        pass


def opciones_gpu():
    """Con NVENC: todo en la tarjeta (sin caer al procesador). Sin NVENC: el video se codifica con el procesador."""
    return ["--gpu", "--solo-gpu"] if CONFIG.get("gpu_nvenc", True) and CONFIG.get("gpu_cuda", True) else []


def matar(proc):
    """Termina el proceso y todo lo que abrió (ffmpeg, whisper...)."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True,
                           creationflags=SIN_VENTANA)
        else:
            proc.kill()
    except Exception:
        pass


def correr_job(jid, cmd, trabajo, env_extra=None):
    job = JOBS[jid]
    with _turno:
        if job.get("cancelar"):
            job.update(estado="cancelado", etapa="Cancelado")
            return
        job.update(estado="trabajando", etapa="Empezando", progreso=0.0)
        anotar("▶ " + " ".join(str(c) for c in cmd))
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", **(env_extra or {}))
        try:
            proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    cwd=str(APP), env=env, creationflags=SIN_VENTANA)
        except Exception as e:
            job.update(estado="error", etapa=f"No pude iniciar: {e}")
            return
        job["_proc"] = proc
        ultimas = []
        for raw in iter(proc.stdout.readline, b""):
            linea = raw.decode("utf-8", errors="replace").rstrip()
            if linea.startswith("@@PROGRESO@@"):
                try:
                    v, etapa = linea.split(" ", 1)[1].split("|", 1)
                    job.update(progreso=float(v), etapa=etapa)
                except ValueError:
                    pass
                continue
            if not linea.strip() or ("WARN" in linea and "dnn" in linea):
                continue
            anotar("   " + linea)
            ultimas = (ultimas + [linea])[-12:]
        proc.wait()
        job.pop("_proc", None)
        if job.get("cancelar"):
            despues_de_cancelar(job, Path(trabajo))
            return
        final = None
        for linea in reversed(ultimas):
            if linea.startswith("{"):
                try:
                    final = json.loads(linea)
                except json.JSONDecodeError:
                    pass
                break
        if proc.returncode == 0 and final and final.get("ok") and job["tipo"] == "procesar":
            # solo subtítulos (sin render): el editor muestra el clip original con los subtítulos encima
            job.update(etapa="Preparando la vista previa", progreso=0.97)
            try:
                preparar_vista(Path(trabajo))
            except Exception as e:
                anotar(f"No pude preparar la vista previa: {e}")
            job.update(estado="listo", progreso=1.0, etapa="Subtítulos listos", trabajo=str(trabajo), salida="")
            anotar("✔ Subtítulos listos para revisar")
        elif proc.returncode == 0 and final and final.get("ok"):
            salida = Path(final.get("salida", ""))
            copia = None
            if salida.exists():
                copia = carpeta_salida() / salida.name
                try:
                    shutil.copy2(salida, copia)
                except OSError as e:
                    anotar(f"No pude copiar al escritorio: {e}")
                    copia = None
            job.update(estado="listo", progreso=1.0, etapa="Listo", trabajo=str(trabajo),
                       salida=str(copia or salida))
            anotar(f"✔ Listo: {copia or salida}")
        else:
            utiles = [x for x in ultimas if not x.startswith("{")][-5:]
            job.update(estado="error", etapa="Algo falló", detalle="\n".join(utiles))
            anotar("✖ Falló")


def despues_de_cancelar(job, t):
    """Deja todo limpio: un clip nuevo sin terminar se borra; un render a medias no se queda como «exportado»."""
    anotar(f"■ Cancelado por el usuario ({job['tipo']})")
    if job["tipo"] == "procesar" and not (t / "plan.json").exists():
        shutil.rmtree(t, ignore_errors=True)
    elif job["tipo"] == "render":
        plan = leer_json(t / "plan.json")
        if plan.get("video"):
            parcial = t / (Path(plan["video"]).stem + "_subtitulado.mp4")
            if parcial.exists() and parcial.stat().st_mtime >= job.get("inicio", 0):
                parcial.unlink(missing_ok=True)
        for tmp in t.glob("vista_previa_tmp*"):
            tmp.unlink(missing_ok=True)
    job.update(estado="cancelado", etapa="Cancelado", progreso=0.0)


def lanzar(cmd, trabajo, tipo, env_extra=None):
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "tipo": tipo, "estado": "en espera", "etapa": "En espera", "progreso": 0.0,
                 "trabajo": str(trabajo), "inicio": time.time()}
    threading.Thread(target=correr_job, args=(jid, cmd, trabajo, env_extra), daemon=True).start()
    return jid


def lanzar_funcion(fn, trabajo, tipo):
    """Trabajo corto dentro del servidor (revisar un video que cambió)."""
    jid = uuid.uuid4().hex[:10]
    JOBS[jid] = {"id": jid, "tipo": tipo, "estado": "en espera", "etapa": "En espera", "progreso": 0.0,
                 "trabajo": str(trabajo), "inicio": time.time()}

    def correr():
        job = JOBS[jid]
        with _turno:
            if job.get("cancelar"):
                job.update(estado="cancelado", etapa="Cancelado")
                return
            job.update(estado="trabajando", etapa="Revisando el video", progreso=0.05)
            try:
                aviso = fn(job)
                if job.get("cancelar"):
                    job.update(estado="cancelado", etapa="Cancelado")
                else:
                    job.update(estado="listo", progreso=1.0, etapa="Listo", aviso=aviso or "", salida="")
            except Exception as e:
                anotar(f"✖ Falló al revisar el video: {e}")
                job.update(estado="error", etapa="Algo falló", detalle=str(e))

    threading.Thread(target=correr, daemon=True).start()
    return jid


# ----------------------------------------------------------------------------
# Vista previa y revisión de un video que cambió
# ----------------------------------------------------------------------------
def preparar_vista(t):
    """Si el navegador puede reproducir el clip tal cual (H.264), el editor usa el original; si no (HEVC, .mov raro…),
    se hace una copia ligera con la tarjeta (unos segundos)."""
    plan_p = t / "plan.json"
    plan = leer_json(plan_p)
    video = Path(plan["video"])
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt",
                        "-of", "json", str(video)], capture_output=True, text=True, creationflags=SIN_VENTANA)
    try:
        st = json.loads(r.stdout)["streams"][0]
    except Exception:
        st = {}
    directo = st.get("codec_name") == "h264" and st.get("pix_fmt") in ("yuv420p", "yuvj420p") and \
        video.suffix.lower() in (".mp4", ".m4v", ".mov")
    ligera = t / "vista_clip.mp4"
    ligera.unlink(missing_ok=True)
    if not directo:
        usa_gpu = CONFIG.get("gpu_nvenc", True) and CONFIG.get("gpu_cuda", True)
        venc = ["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "30"] if usa_gpu else \
            ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30"]
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0?",
                            "-vf", "scale=-2:960", *venc, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                            "-movflags", "+faststart", str(ligera)], capture_output=True, text=True,
                           creationflags=SIN_VENTANA)
        if r.returncode != 0:
            anotar("No pude hacer la copia ligera para la vista previa: " + r.stderr[-300:])
            ligera.unlink(missing_ok=True)
    plan["vista_original"] = True
    escribir_json(plan_p, plan)


def revisar_video(t, job):
    """El video de su ruta cambió: se usa con los mismos subtítulos. Se vuelve a buscar la franja azul y
    se avisa si cambió la duración (los tiempos de los subtítulos podrían no coincidir)."""
    import solo_subtitulos as so
    so.ed = ed
    meta = leer_json(t / "meta.json")
    plan = leer_json(t / "plan.json")
    ruta = Path(meta["ruta"])
    video = ruta
    if ruta.suffix.lower() != ".mp4":
        job.update(etapa="Convirtiendo el video", progreso=0.1)
        os.environ["CONVERTIR_EN"] = str(t)
        try:
            video = ed.asegurar_mp4(ruta)
        finally:
            os.environ.pop("CONVERTIR_EN", None)
    if job.get("cancelar"):
        return ""
    exportado = t / (Path(plan.get("video", "x")).stem + "_subtitulado.mp4")
    fps, n, W, H = ed.info_video(video)
    dur_vieja = plan.get("info", {}).get("duracion", 0)
    dur = n / fps if fps else 0
    plan["video"] = str(video)
    plan.setdefault("info", {}).update({"fps": fps, "cuadros": n, "ancho": W, "alto": H, "duracion": dur})
    pl = ed.cargar_json(plan["plantilla"])
    if pl.get("franja_azul", {}).get("activar", True):
        job.update(etapa="Buscando la franja azul", progreso=0.3)
        plan["franjas"] = so.detectar_franjas(video, plan["info"], pl.get("franja_azul", {}))
    if job.get("cancelar"):
        return ""
    escribir_json(t / "plan.json", plan)
    # la vista previa y el último render ya no corresponden al video nuevo
    for viejo in ("vista_previa.mp4", "vista_previa.json"):
        (t / viejo).unlink(missing_ok=True)
    exportado.unlink(missing_ok=True)  # era del video anterior: hay que volver a exportar
    job.update(etapa="Preparando la vista previa", progreso=0.85)
    preparar_vista(t)
    meta.update(huella(ruta))
    escribir_json(t / "meta.json", meta)
    anotar(f"↻ El video cambió y se actualizó: {ruta}")
    if abs(dur - dur_vieja) > 0.5:
        return (f"El video cambió de duración ({dur_vieja:.1f} s → {dur:.1f} s). Revisa que los subtítulos sigan "
                "en su lugar; si no, vuelve a elegir el video y genera subtítulos nuevos.")
    return "El video cambió: se usa la versión nueva con los mismos subtítulos."


def nombre_limpio(nombre):
    base = Path(nombre).stem
    base = re.sub(r"[^\w\-. áéíóúÁÉÍÓÚñÑüÜ]", "_", base).strip() or "clip"
    return base[:80]


def info_trabajo(t):
    t = Path(t)
    plan = leer_json(t / "plan.json")
    meta = leer_json(t / "meta.json")
    salida = I.video_salida(str(t)) if plan else None
    return {"trabajo": str(t), "nombre": meta.get("nombre", t.name), "estilo": meta.get("estilo", "chapuceros"),
            "fecha": meta.get("fecha", ""), "listo": bool(salida), "subtitulos": bool(plan),
            "ruta": meta.get("ruta", ""),
            "salida": str(carpeta_salida() / Path(salida).name) if salida else ""}


def comando_procesar(video, t, estilo):
    cmd = [PY, APP / "editor.py", "analizar", video, "--titulo", "", "--plantilla", ESTILOS[estilo][1],
           "--trabajo", t, "--whisper-modelo", CONFIG.get("whisper_modelo", "large-v3"),
           "--dispositivo", "cuda" if CONFIG.get("gpu_cuda", True) else "cpu"]
    if "--solo-gpu" in opciones_gpu():
        cmd.append("--solo-gpu")
    if CONFIG.get("ollama_modelo"):
        cmd += ["--modelo-ollama", CONFIG["ollama_modelo"]]
    if os.environ.get("SUBT_PRUEBA_TRANSCRIPCION"):  # solo para pruebas sin whisper
        cmd += ["--transcripcion", os.environ["SUBT_PRUEBA_TRANSCRIPCION"], "--sin-ia"]
    return cmd


# ----------------------------------------------------------------------------
# Servidor
# ----------------------------------------------------------------------------
app = I.crear_app()


@app.get("/")
def inicio():
    return FileResponse(APP / "web" / "subtitulador.html", headers={"Cache-Control": "no-store"})


@app.get("/api/sistema")
def sistema():
    return {"version": VERSION, "gpu": CONFIG.get("gpu_nombre", "¿?"), "vram": CONFIG.get("vram_gb"),
            "whisper": CONFIG.get("whisper_modelo", "large-v3"), "ollama": CONFIG.get("ollama_modelo", ""),
            "salida": str(carpeta_salida())}


@app.post("/api/elegir")
def elegir():
    """Ventana de Windows para elegir el clip. Dice si ese mismo video ya tiene subtítulos hechos."""
    ruta = elegir_video_windows()
    if not ruta:
        return {"ruta": ""}
    if not Path(ruta).is_file():
        raise HTTPException(404, "No encuentro ese archivo")
    previo = trabajo_de_ruta(ruta)
    return {"ruta": ruta, "nombre": Path(ruta).name,
            "existente": info_trabajo(previo) if previo else None}


@app.post("/api/procesar")
def procesar(d: dict = Body(...)):
    ruta = Path(str(d.get("ruta") or ""))
    estilo = d.get("estilo") if d.get("estilo") in ESTILOS else "chapuceros"
    if not ruta.is_file():
        raise HTTPException(404, "No encuentro ese video")
    nombre = nombre_limpio(ruta.name)
    t = TRABAJOS / f"{nombre}_{time.strftime('%Y%m%d_%H%M%S')}"
    t.mkdir(parents=True, exist_ok=True)
    escribir_json(t / "meta.json", {"nombre": nombre, "estilo": estilo, "fecha": time.strftime("%d/%m/%Y %H:%M"),
                                    "ruta": str(ruta), **huella(ruta)})
    # paso 1: solo subtítulos (sin render), directo del video en su ruta (sin copiarlo)
    cmd = comando_procesar(ruta, t, estilo)
    return {"job": lanzar(cmd, t, "procesar", {"CONVERTIR_EN": str(t)}), "trabajo": str(t)}


@app.post("/api/abrir_trabajo")
def abrir_trabajo(d: dict = Body(...)):
    """Al abrir un clip de la lista se busca su video en la ruta original.
    -> estado: 'ok' | 'cambio' (se lanza una revisión corta, viene 'job') | 'falta' (no está en su ruta)."""
    t = Path(str(d.get("trabajo") or ""))
    if not (t / "plan.json").exists():
        raise HTTPException(404, "Trabajo no válido")
    meta = leer_json(t / "meta.json")
    ruta = meta.get("ruta")
    if not ruta:  # clips de versiones anteriores: usan su copia
        return {"estado": "ok"}
    if not Path(ruta).is_file():
        return {"estado": "falta", "ruta": ruta}
    h = huella(ruta)
    if h["tam"] == meta.get("tam") and abs(h["mtime"] - meta.get("mtime", 0)) < 1:
        return {"estado": "ok", "ruta": ruta}
    jid = lanzar_funcion(lambda job: revisar_video(t, job), t, "revisar")
    return {"estado": "cambio", "ruta": ruta, "job": jid}


@app.post("/api/relocalizar")
def relocalizar(d: dict = Body(...)):
    """El video ya no estaba en su ruta: se elige de nuevo y se revisa."""
    t = Path(str(d.get("trabajo") or ""))
    if not (t / "plan.json").exists():
        raise HTTPException(404, "Trabajo no válido")
    ruta = elegir_video_windows("¿Dónde está ahora este video?")
    if not ruta:
        return {"ruta": ""}
    meta = leer_json(t / "meta.json")
    meta.update({"ruta": ruta, "tam": -1, "mtime": 0})  # fuerza la revisión
    escribir_json(t / "meta.json", meta)
    jid = lanzar_funcion(lambda job: revisar_video(t, job), t, "revisar")
    return {"ruta": ruta, "job": jid}


@app.post("/api/render")
def render(d: dict = Body(...)):
    t = str(d.get("trabajo") or "")
    if not t or not (Path(t) / "plan.json").exists():
        raise HTTPException(404, "Trabajo no válido")
    cmd = [PY, APP / "editor.py", "render", t] + opciones_gpu()
    return {"job": lanzar(cmd, t, "render")}


@app.post("/api/cancelar/{jid}")
def cancelar(jid: str):
    job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, "No existe")
    if job["estado"] in ("listo", "error", "cancelado"):
        return {"ok": True}
    job["cancelar"] = True
    job["etapa"] = "Cancelando…"
    matar(job.get("_proc"))
    return {"ok": True}


@app.get("/api/job/{jid}")
def estado_job(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "No existe")
    return {k: v for k, v in JOBS[jid].items() if not k.startswith("_")}


@app.get("/api/trabajos")
def trabajos():
    lista = sorted([p for p in TRABAJOS.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)[:30]
    return [info_trabajo(p) for p in lista if (p / "meta.json").exists()]


@app.post("/api/borrar")
def borrar(d: dict = Body(...)):
    t = Path(d.get("trabajo", ""))
    if t.parent.resolve() != TRABAJOS.resolve() or not t.exists():
        raise HTTPException(400, "No válido")
    shutil.rmtree(t, ignore_errors=True)
    return {"ok": True}


@app.post("/api/abrir")
def abrir(d: dict = Body(default={})):
    destino = Path(d.get("ruta") or carpeta_salida())
    if os.name == "nt":
        if destino.is_file():
            subprocess.Popen(["explorer", "/select,", str(destino)])
        else:
            os.startfile(str(destino))  # noqa
    return {"ok": True}


@app.get("/api/registro")
def registro():
    if not REGISTRO.exists():
        return JSONResponse({"texto": ""})
    lineas = REGISTRO.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]
    return JSONResponse({"texto": "\n".join(lineas), "ruta": str(REGISTRO)})


# ----------------------------------------------------------------------------
# Título de la barra roja (se edita junto a los subtítulos y sale en el video exportado)
# ----------------------------------------------------------------------------
import titulos  # noqa: E402


def trabajo_valido(t):
    t = Path(I.limpiar_ruta(str(t or "")))
    if not (t / "plan.json").exists():
        raise HTTPException(404, "Carpeta de trabajo no válida")
    return t


@app.get("/titulo/datos")
def titulo_datos(t: str):
    t = trabajo_valido(t)
    plan = leer_json(t / "plan.json")
    return {"titulo": titulos.cargar(t), "guardado": (t / "titulo.json").exists(),
            "salida": plan.get("subtitulos_render", {}).get("salida") or [plan["info"]["ancho"], plan["info"]["alto"]],
            "ref": [titulos.REF_W, titulos.REF_H], "caja": list(titulos.CAJA),
            "margen": [titulos.MARGEN_X, titulos.MARGEN_Y], "paso_renglon": titulos.PASO_RENGLON,
            "limites": titulos.LIMITES}


@app.post("/titulo/guardar")
def titulo_guardar(d: dict = Body(...)):
    t = trabajo_valido(d.get("t"))
    if not str(d.get("titulo", {}).get("texto", "")).strip():
        (t / "titulo.json").unlink(missing_ok=True)  # sin texto = sin título
        anotar(f"Título quitado: {t.name}")
        return {"ok": True, "titulo": titulos.normalizar({}), "guardado": False}
    g = titulos.guardar(t, d.get("titulo"))
    lineas, tam = titulos.acomodar(g)
    anotar(f"Título guardado: {' / '.join(lineas)} (letra {tam})")
    return {"ok": True, "titulo": g, "guardado": True, "lineas": lineas, "tam_final": tam}


@app.get("/titulo/fuente")
def titulo_fuente():
    return FileResponse(titulos.FUENTE, media_type="font/otf", headers={"Cache-Control": "max-age=86400"})


@app.get("/titulo/plantilla.png")
def titulo_plantilla():
    return FileResponse(titulos.PLANTILLA, media_type="image/png", headers={"Cache-Control": "max-age=86400"})


@app.get("/notificacion.mp3")
def notificacion():
    return FileResponse(APP / "web" / "notificacion.mp3", media_type="audio/mpeg")


def fecha_trabajo(t):
    """Fecha (local) en que se creó el trabajo: la de meta.json o, si no hay, la de la carpeta."""
    meta = leer_json(t / "meta.json")
    try:
        return time.strptime(meta.get("fecha", ""), "%d/%m/%Y %H:%M")
    except ValueError:
        return time.localtime(t.stat().st_mtime)


@app.post("/api/limpiar")
def limpiar():
    """Borra los trabajos de días anteriores (los de hoy se quedan). Los videos de Subtitulados no se tocan."""
    hoy = time.strftime("%Y-%m-%d")
    ocupados = {Path(j["trabajo"]) for j in JOBS.values() if j["estado"] in ("en espera", "trabajando")}
    borrados, liberado = 0, 0
    for t in TRABAJOS.iterdir():
        if not t.is_dir() or t in ocupados:
            continue
        if time.strftime("%Y-%m-%d", fecha_trabajo(t)) >= hoy:
            continue
        tam = sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
        shutil.rmtree(t, ignore_errors=True)
        if not t.exists():
            borrados += 1
            liberado += tam
    anotar(f"🗑 Limpieza: {borrados} trabajos de días anteriores ({liberado / 1e6:.0f} MB)")
    return {"borrados": borrados, "mb": round(liberado / 1e6)}


@app.post("/api/cerrar")
def cerrar():
    """Cierra el Subtitulador por completo: detiene lo que esté corriendo y termina el proceso de Python
    (como finalizarlo en el Administrador de tareas)."""
    for job in JOBS.values():
        if job["estado"] in ("en espera", "trabajando"):
            job["cancelar"] = True
            matar(job.get("_proc"))
    anotar("⏻ Subtitulador cerrado desde el botón «Cerrar»")

    def salir():
        time.sleep(0.8)  # deja que la respuesta llegue al navegador
        os._exit(0)

    threading.Thread(target=salir, daemon=True).start()
    return {"ok": True}


@app.get("/api/vivo")
def vivo():
    return {"ok": True, "app": "subtitulador"}


def main():
    import uvicorn
    puerto = int(os.environ.get("PUERTO", CONFIG.get("puerto", 7870)))
    uvicorn.run(app, host="127.0.0.1", port=puerto, log_level="warning")


if __name__ == "__main__":
    main()
