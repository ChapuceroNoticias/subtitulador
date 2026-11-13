#!/usr/bin/env python3
"""
Editor vertical automático: video 16:9 -> 9:16 con plantilla.

Uso:
  python editor.py analizar VIDEO --titulo "TEXTO" [--plantilla P] [--trabajo DIR] [--imagenes CARPETA] [--sin-web]
  python editor.py render DIR_TRABAJO [--salida final.mp4] [--gpu]
  python editor.py todo VIDEO --titulo "TEXTO" [...]          (analizar + render)
  python editor.py calibrar VIDEO --t 26 [--plantilla P]       (guarda un cuadro con los recuadros dibujados)

Al terminar imprime en stdout una línea JSON (para n8n). Los avisos van a stderr.
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

# Modo "solo tarjeta gráfica": el trabajo pesado va a la GPU y al procesador se le deja poco
SOLO_GPU = "--solo-gpu" in sys.argv or os.environ.get("EDITOR_SOLO_GPU") == "1"
HILOS_CPU = "2"
if SOLO_GPU:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[_v] = HILOS_CPU

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parent


def cuidar_procesador():
    """Prioridad baja y pocos hilos: la computadora sigue respondiendo mientras la plataforma trabaja."""
    if not SOLO_GPU:
        return
    cv2.setNumThreads(int(HILOS_CPU))
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)  # debajo de lo normal
        else:
            os.nice(10)
    except Exception:
        pass


def codificador(usar_gpu_pedido, crf):
    """Parámetros de ffmpeg para codificar. En modo solo GPU, sin NVENC no se codifica (no se usa el procesador)."""
    if SOLO_GPU or usar_gpu_pedido:
        if nvenc_disponible():
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", str(crf + 1), "-b:v", "0"]
        if SOLO_GPU:
            raise SystemExit("La tarjeta gráfica no pudo codificar el video (NVENC). Actualiza el driver de NVIDIA "
                             "(GeForce Experience o nvidia.com) y reinicia. No lo hago con el procesador porque "
                             "está activado «Solo tarjeta gráfica».")
        log("NVENC no disponible (¿driver de NVIDIA desactualizado?). Codifico con CPU.")
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-threads", "4"]
MODELO_CARA = BASE / "modelos" / "face_detection_yunet_2023mar.onnx"
EXT_IMG = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def log(*a):
    print("[editor]", *a, file=sys.stderr, flush=True)


class Progreso:
    """Reporta avance general (0..1) en líneas '@@PROGRESO@@ fracción|etapa' que lee la interfaz."""
    a, b = 0.0, 1.0
    ultimo = -1.0

    @classmethod
    def rango(cls, a, b):
        cls.a, cls.b = a, b

    @classmethod
    def paso(cls, frac, etapa):
        total = cls.a + (cls.b - cls.a) * min(1.0, max(0.0, frac))
        if abs(total - cls.ultimo) >= 0.004 or frac >= 1.0 or frac <= 0.0:
            cls.ultimo = total
            print(f"@@PROGRESO@@ {total:.4f}|{etapa}", file=sys.stderr, flush=True)


def ruta(p):
    """Rutas de la plantilla relativas a la carpeta del script."""
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else BASE / p


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def cargar_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def guardar_json(p, data):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def leer_img(p):
    """Lee una imagen aunque la ruta tenga acentos o Ñ (en Windows OpenCV no puede abrirlas directo)."""
    try:
        datos = np.fromfile(str(p), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if datos.size == 0:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def guardar_img(p, img, calidad=92):
    """Guarda una imagen aunque la ruta tenga acentos o Ñ."""
    ext = Path(p).suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, calidad] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(str(p))
    return Path(p).exists()


def dentro(t, intervalos):
    for it in intervalos:
        if it["inicio"] <= t < it["fin"]:
            return it
    return None


# ----------------------------------------------------------------------------
# Detección de rostro
# ----------------------------------------------------------------------------
class DetectorCara:
    def __init__(self):
        self.yunet = None
        if hasattr(cv2, "FaceDetectorYN") and MODELO_CARA.exists():
            self.yunet = cv2.FaceDetectorYN.create(str(MODELO_CARA), "", (320, 320), 0.6)
        else:
            log("YuNet no disponible, uso Haar (menos preciso)")
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def mayor(self, img):
        """Devuelve [x, y, w, h] de la cara más grande o None."""
        if img is None or img.size == 0:
            return None
        if self.yunet is not None:
            self.yunet.setInputSize((img.shape[1], img.shape[0]))
            _, caras = self.yunet.detect(img)
            if caras is None or len(caras) == 0:
                return None
            c = max(caras, key=lambda r: r[2] * r[3])
            return [float(c[0]), float(c[1]), float(c[2]), float(c[3])]
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        caras = self.haar.detectMultiScale(g, 1.1, 5, minSize=(60, 60))
        if len(caras) == 0:
            return None
        c = max(caras, key=lambda r: r[2] * r[3])
        return [float(v) for v in c]


# ----------------------------------------------------------------------------
# Análisis de tomas
# ----------------------------------------------------------------------------
def clasificar_cuadro(frame, det_cfg):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    x, y, w, h = det_cfg["franja_superior"]
    sup = g[y:y + h, x:x + w].mean()
    x, y, w, h = det_cfg["columna_separacion"]
    col = g[y:y + h, x:x + w].mean()
    u = det_cfg["umbral_oscuro"]
    return "dividida" if (sup < u and col < u * 1.5) else "completa"


def suavizar_etiquetas(muestras, min_muestras):
    """Elimina segmentos más cortos que min_muestras (parpadeos de la detección)."""
    etiquetas = [m["layout"] for m in muestras]
    cambiado = True
    while cambiado:
        cambiado = False
        i = 0
        while i < len(etiquetas):
            j = i
            while j < len(etiquetas) and etiquetas[j] == etiquetas[i]:
                j += 1
            if (j - i) < min_muestras and (i > 0 or j < len(etiquetas)):
                vecino = etiquetas[i - 1] if i > 0 else etiquetas[j]
                if vecino != etiquetas[i]:
                    for k in range(i, j):
                        etiquetas[k] = vecino
                    cambiado = True
            i = j
    for m, e in zip(muestras, etiquetas):
        m["layout"] = e


def info_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h


def tiene_audio(path):
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
                            "stream=index", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
        return bool(r.stdout.strip())
    except FileNotFoundError:
        raise SystemExit("No encontré ffprobe/ffmpeg en el PATH")


def escalar_cajas(pl, w, h):
    """Si el video no es 1920x1080, escala las coordenadas de la plantilla."""
    sx, sy = w / 1920.0, h / 1080.0
    if abs(sx - 1) < 1e-3 and abs(sy - 1) < 1e-3:
        return
    for k in ("persona_dividida", "proyeccion"):
        b = pl["fuente_16x9"][k]
        pl["fuente_16x9"][k] = [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]
    for k in ("franja_superior", "columna_separacion"):
        b = pl["deteccion"][k]
        pl["deteccion"][k] = [int(b[0] * sx), int(b[1] * sy), max(1, int(b[2] * sx)), max(1, int(b[3] * sy))]


def caja_contenido(gris, x0, y0, umbral=20):
    """Rectángulo con contenido (no negro) dentro de la proyección, en coordenadas del video."""
    cols = np.where(gris.mean(0) > umbral)[0]
    filas = np.where(gris.mean(1) > umbral)[0]
    if len(cols) < 10 or len(filas) < 10:
        return None
    return [int(x0 + cols.min()), int(y0 + filas.min()), int(cols.max() - cols.min() + 1), int(filas.max() - filas.min() + 1)]


def analizar_tomas(video, pl):
    fps, n, w, h = info_video(video)
    escalar_cajas(pl, w, h)
    det_cfg = pl["deteccion"]
    paso = max(1, round(fps / det_cfg["muestreo_fps"]))
    caja_p = [int(v) for v in pl["fuente_16x9"]["persona_dividida"]]
    px, py, pw, ph = [int(v) for v in pl["fuente_16x9"]["proyeccion"]]
    detector = DetectorCara()

    cap = cv2.VideoCapture(str(video))
    muestras = []
    idx = 0
    proy_ant = None
    pers_ant = None  # recuadro de la persona del cuadro anterior (para saber si está hablando)
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % paso == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if n:
                Progreso.paso(0.20 * idx / n, "Analizando las tomas del video")
            lay = clasificar_cuadro(frame, det_cfg)
            m = {"t": idx / fps, "n": idx, "layout": lay, "cara": None, "mov": 0.0, "caja": None, "cara_p": None}
            if lay == "dividida":
                x, y, bw, bh = caja_p
                c = detector.mayor(frame[y:y + bh, x:x + bw])
                if c:
                    c[0] += x
                    c[1] += y
                m["cara"] = c
                if det_cfg.get("actividad_persona"):
                    # cuánto se mueven la boca y la cara de Nacho: si habla se mueve; si ve el video, casi no
                    gp = cv2.cvtColor(frame[y:y + bh, x:x + bw], cv2.COLOR_BGR2GRAY)
                    if pers_ant is not None and c:
                        fx, fy, fw, fh = [int(v) for v in c]
                        fx, fy = fx - x, fy - y
                        zonas = [(fy + int(fh * 0.62), fy + fh, fx + int(fw * 0.2), fx + int(fw * 0.8)),
                                 (fy, fy + int(fh * 0.5), fx, fx + fw)]
                        difs = []
                        for y0z, y1z, x0z, x1z in zonas:
                            y0z, x0z = max(0, y0z), max(0, x0z)
                            a_ = gp[y0z:y1z, x0z:x1z]
                            b_ = pers_ant[y0z:y1z, x0z:x1z]
                            if a_.size > 16 and a_.shape == b_.shape:
                                difs.append(float(np.abs(cv2.resize(a_, (32, 16)).astype(np.float32)
                                                         - cv2.resize(b_, (32, 16)).astype(np.float32)).mean()))
                        if len(difs) == 2:
                            m["act"] = round((difs[0] + difs[1]) / 2, 2)
                    pers_ant = gp
                gris = cv2.cvtColor(frame[py:py + ph, px:px + pw], cv2.COLOR_BGR2GRAY)
                m["caja"] = caja_contenido(gris, px, py)
                if det_cfg.get("movimiento_en_contenido") and m["caja"]:
                    # el movimiento se mide solo donde hay contenido (un video de celular sobre fondo negro)
                    peq = cv2.resize(gris, (pw // 8, ph // 8)).astype(np.float32)
                    if proy_ant is not None and proy_ant.shape == peq.shape:
                        cx, cy, cw, ch = m["caja"]
                        x0, y0 = (cx - px) // 8, (cy - py) // 8
                        x1, y1 = max(x0 + 1, (cx - px + cw) // 8), max(y0 + 1, (cy - py + ch) // 8)
                        dif = np.abs(peq[y0:y1, x0:x1] - proy_ant[y0:y1, x0:x1])
                        m["mov"] = float(dif.mean())
                        # dónde se concentra el movimiento (para encuadrar si no hay una cara)
                        pesos = np.where(dif > 12, dif, 0)
                        if pesos.sum() > 50 * 12:
                            ys_, xs_ = np.indices(pesos.shape)
                            m["foco_m"] = [float(px + (x0 + (xs_ * pesos).sum() / pesos.sum()) * 8),
                                           float(py + (y0 + (ys_ * pesos).sum() / pesos.sum()) * 8)]
                else:
                    peq = cv2.resize(gris, (96, 54)).astype(np.float32)
                    if proy_ant is not None and proy_ant.shape == peq.shape:
                        m["mov"] = float(np.abs(peq - proy_ant).mean())
                proy_ant = peq
                cp = detector.mayor(frame[py:py + ph, px:px + pw])
                if cp and cp[3] > 40:
                    m["cara_p"] = [cp[0] + px, cp[1] + py, cp[2], cp[3]]
            else:
                m["cara"] = detector.mayor(frame)
                proy_ant = None
                pers_ant = None
            muestras.append(m)
        idx += 1
    cap.release()
    duracion = idx / fps
    log(f"{len(muestras)} muestras analizadas, duración {duracion:.1f}s")

    suavizar_etiquetas(muestras, max(1, int(det_cfg["duracion_minima_seg"] * det_cfg["muestreo_fps"])))

    segs = []
    for m in muestras:
        if not segs or segs[-1]["layout"] != m["layout"]:
            segs.append({"layout": m["layout"], "inicio": m["t"], "muestras": []})
        segs[-1]["muestras"].append(m)

    # afinar cada corte al cuadro exacto
    cap = cv2.VideoCapture(str(video))
    for i in range(1, len(segs)):
        n_fin = segs[i]["muestras"][0]["n"]
        n_ini = max(0, n_fin - paso + 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, n_ini)
        for n_c in range(n_ini, n_fin):
            ok, fr = cap.read()
            if ok and clasificar_cuadro(fr, det_cfg) == segs[i]["layout"]:
                segs[i]["inicio"] = n_c / fps
                break
    cap.release()

    for i, s in enumerate(segs):
        s["fin"] = segs[i + 1]["inicio"] if i + 1 < len(segs) else duracion
        if s["layout"] == "dividida":
            region = caja_p
        else:
            region = [0, 0, w, h]
        s["cara"] = pista_cara(s["muestras"], "cara", region, pl["deteccion"]["muestreo_fps"])
    info = {"fps": fps, "cuadros": n, "ancho": w, "alto": h, "duracion": duracion, "audio": tiene_audio(video)}
    return segs, info


def pista_cara(muestras, campo, region, muestreo_fps):
    """Rellena huecos y suaviza la posición de la cara: lista [t, cx, cy, alto_cara]."""
    ts = [m["t"] for m in muestras]
    datos = [m[campo] for m in muestras]
    validos = [i for i, c in enumerate(datos) if c]
    if not validos:
        x, y, w, h = region
        return [[round(ts[0], 3), x + w / 2, y + h * 0.4, h * 0.35]]
    arr = np.full((len(datos), 3), np.nan)
    for i in validos:
        x, y, w, h = datos[i]
        arr[i] = [x + w / 2, y + h / 2, h]
    for k in range(3):
        col = arr[:, k]
        malos = np.isnan(col)
        if malos.any():
            col[malos] = np.interp(np.flatnonzero(malos), np.flatnonzero(~malos), col[~malos])
    v = max(1, int(round(muestreo_fps * 1.5)))
    pad = v // 2
    med = np.array([np.median(arr[max(0, i - pad):i + pad + 1], axis=0) for i in range(len(arr))])
    ker = np.ones(v) / v
    sua = np.stack([np.convolve(np.pad(med[:, k], pad, mode="edge"), ker, mode="valid")[:len(med)] for k in range(3)], 1)
    return [[round(t, 3), *[round(float(x), 1) for x in fila]] for t, fila in zip(ts, sua)]


def runs(flags):
    """[(i, j)] de tramos consecutivos True (j exclusivo)."""
    out, i = [], 0
    while i < len(flags):
        if flags[i]:
            j = i
            while j < len(flags) and flags[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def umbral_otsu(vals, lo, hi, defecto):
    """Umbral que separa mejor dos grupos de valores (p. ej. hablando / callado), acotado a [lo, hi]."""
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) < 20:
        return defecto
    mejor, um = -1, defecto
    for u in np.linspace(lo, hi, 25):
        a, b = v[v < u], v[v >= u]
        if len(a) < 5 or len(b) < 5:
            continue
        var = len(a) * len(b) * (a.mean() - b.mean()) ** 2
        if var > mejor:
            mejor, um = var, u
    return float(um)


def _mfcc(x, sr=16000):
    """Coeficientes MFCC (100 por segundo) y energía en dB, sin librerías extra."""
    from scipy.fft import dct
    N, H = 512, 160
    nf = (len(x) - N) // H
    if nf <= 0:
        return np.zeros((0, 13)), np.zeros(0)
    idx = np.arange(N)[None, :] + H * np.arange(nf)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hamming(N), axis=1)) ** 2
    f = np.fft.rfftfreq(N, 1 / sr)
    mel = lambda hz: 2595 * np.log10(1 + hz / 700)
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = imel(np.linspace(mel(80), mel(7600), 28))
    fb = np.zeros((26, len(f)))
    for k in range(26):
        l, c, r = pts[k], pts[k + 1], pts[k + 2]
        fb[k] = np.clip(np.minimum((f - l) / (c - l), (r - f) / (r - c)), 0, None)
    E = np.log(S @ fb.T + 1e-10)
    return dct(E, type=2, axis=1, norm="ortho")[:, 1:14], 10 * np.log10(S.sum(1) + 1e-10)


def voces(video, segs, cfg):
    """¿Quién habla en cada muestra de la pantalla partida? 1 = Nacho, -1 = otra persona (el video que puso),
    0 = nadie. Aprende la voz de Nacho de las tomas donde solo sale él y la compara con el resto. -> {t: int} o None."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        log("Para distinguir la voz de Nacho del audio de los videos instala: python -m pip install scikit-learn")
        return None
    solo = [(sg["inicio"], sg["fin"]) for sg in segs if sg["layout"] != "dividida"]
    if sum(b - a for a, b in solo) < cfg.get("voz_min_entrenamiento_seg", 4):
        return None
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.float32)
    if len(a) < 16000 * 5:
        return None
    M, en = _mfcc(a)
    t = np.arange(len(M)) * 0.01
    voz = en > np.percentile(en, 40)
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    sel = np.zeros(len(M), bool)
    for x, y in solo:
        sel |= (t >= x) & (t < y)
    if (sel & voz).sum() < 300:
        return None
    nacho = GaussianMixture(16, covariance_type="diag", random_state=0).fit(M[sel & voz])
    todos = GaussianMixture(16, covariance_type="diag", random_state=0).fit(M[voz])
    llr = nacho.score_samples(M) - todos.score_samples(M)
    u = cfg.get("voz_umbral", -2.5)
    medio = 0.5
    if cfg.get("voz_autoaprender"):
        # Umbral a la medida de este video: se aprende la voz de Nacho con la mitad de sus tomas y se mide qué tanto
        # se parece la otra mitad (bloques de 2 s alternados). Lo que queda muy por debajo de eso no es Nacho.
        bloque = (t // 2).astype(int) % 2 == 0
        ok_a, ok_b = sel & voz & bloque, sel & voz & ~bloque
        if ok_a.sum() > 150 and ok_b.sum() > 150:
            na = GaussianMixture(16, covariance_type="diag", random_state=0).fit(M[ok_a])
            nb = GaussianMixture(16, covariance_type="diag", random_state=0).fit(M[ok_b])
            base = todos.score_samples(M)
            fuera = np.where(bloque, nb.score_samples(M) - base, na.score_samples(M) - base)
            vals = []
            for s0 in np.arange(0, t[-1] if len(t) else 0, 2.0):
                q = (t >= s0) & (t < s0 + 2) & voz & sel
                if q.sum() > 30:
                    vals.append(float(fuera[q].mean()))
            if len(vals) >= 3:
                mu, sd = float(np.mean(vals)), float(np.std(vals))
                u = mu - max(3 * sd, cfg.get("voz_margen_min", 2.0))
                log(f"Voz de Nacho: parecido normal {mu:.1f} ± {sd:.1f}; menos de {u:.1f} = otra persona")
        medio = 1.0
    res = {}
    for sg in segs:
        if sg["layout"] != "dividida":
            continue
        for m in sg.get("muestras", []):
            k0, k1 = int((m["t"] - medio) * 100), int((m["t"] + medio) * 100)
            k0, k1 = max(0, k0), min(len(llr), k1)
            v = voz[k0:k1]
            if v.sum() <= 15:
                res[round(m["t"], 3)] = 0
            else:
                res[round(m["t"], 3)] = 1 if llr[k0:k1][v].mean() >= u else -1
    log("Voz de Nacho distinguida del audio de los videos")
    return res


def voz_de_nacho(video, segs, cfg):
    """{t: True si habla Nacho} en la pantalla partida, o None."""
    v = voces(video, segs, cfg)
    return None if v is None else {t: x == 1 for t, x in v.items()}


def mostrar_al_que_habla(segs, proy, pl, video):
    """Clips Chapuceros: si en la pantalla partida Nacho se calla y habla la persona del video que puso,
    ese rato va el video a pantalla completa (aunque el video casi no se mueva). Nacho no sale viendo la pantalla."""
    cfg = pl.get("proyeccion", {})
    if not cfg.get("mostrar_al_que_habla") or cfg.get("video_vertical_completo"):
        return proy
    fs = pl["deteccion"]["muestreo_fps"]
    v = voces(video, segs, dict(cfg, voz_autoaprender=True))
    if not v:
        return proy
    nuevos = []
    for sg in segs:
        if sg["layout"] != "dividida":
            continue
        ms = [m for m in sg.get("muestras", []) if round(m["t"], 3) in v]
        if not ms:
            continue
        # otra persona habla = voz que no es de Nacho; los silencios cortos no cortan el tramo
        otro = np.array([v[round(m["t"], 3)] == -1 for m in ms])
        nacho = np.array([v[round(m["t"], 3)] == 1 for m in ms])
        bandera = otro.copy()
        for a, b in runs(list(~otro)):
            if 0 < a and b < len(otro) and not nacho[a:b].any():
                bandera[a:b] = True  # pausa de la otra persona
        bandera = suavizar_bandera(bandera, int(cfg.get("otro_min_seg", 2.0) * fs),
                                   int(cfg.get("nacho_min_seg", 1.5) * fs))
        for a, b in runs(list(bandera)):
            sub = ms[a:b]
            ini = sub[0]["t"]
            fin = min(sub[-1]["t"] + 1 / fs, sg["fin"])
            if fin - ini < cfg.get("otro_min_seg", 2.0):
                continue
            cajas = [m["caja"] for m in sub if m.get("caja")]
            if len(cajas) < 0.5 * len(sub):
                continue  # no hay nada en la proyección
            caja = [int(np.median([c[k] for c in cajas])) for k in range(4)]
            nuevos.append({"tipo": "video", "inicio": round(ini, 3), "fin": round(fin, 3), "caja": caja,
                           "cara": pista_cara(sub, "cara_p", caja, fs), "habla_otro": True,
                           "con_cara": round(sum(1 for m in sub if m.get("cara_p")) / max(1, len(sub)), 2)})
    if not nuevos:
        return proy
    # lo nuevo manda: se recorta lo que se encime (salvo documentos que Nacho lee)
    salida = []
    for p in proy:
        if p.get("documento"):
            salida.append(p)
            continue
        piezas = [(p["inicio"], p["fin"])]
        for q in nuevos:
            otras = []
            for a, b in piezas:
                if q["fin"] <= a or q["inicio"] >= b:
                    otras.append((a, b))
                    continue
                if q["inicio"] - a >= 1.0:
                    otras.append((a, q["inicio"]))
                if b - q["fin"] >= 1.0:
                    otras.append((q["fin"], b))
            piezas = otras
        for a, b in piezas:
            salida.append(dict(p, inicio=round(a, 3), fin=round(b, 3)))
    docs = [p for p in proy if p.get("documento")]
    for q in nuevos:
        if any(d["inicio"] < q["fin"] and d["fin"] > q["inicio"] for d in docs):
            continue
        salida.append(q)
    salida.sort(key=lambda x: x["inicio"])
    # unir videos pegados (el que ya estaba con el nuevo)
    unidos = []
    for q in salida:
        u = unidos[-1] if unidos else None
        if u and u["tipo"] == "video" and q["tipo"] == "video" and not u.get("pantalla") and not q.get("pantalla") \
                and q["inicio"] - u["fin"] < 1.0 and u.get("caja") == q.get("caja"):
            u["fin"] = max(u["fin"], q["fin"])
            u["cara"] = sorted(u.get("cara", []) + q.get("cara", []))
            u["habla_otro"] = True
        else:
            unidos.append(q)
    log(f"La persona del video habla en {len(nuevos)} ratos: ahí va el video a pantalla completa")
    return unidos


def suavizar_bandera(flag, min_si, min_no):
    """Quita tramos True más cortos que min_si y False más cortos que min_no (en muestras)."""
    flag = np.array(flag, bool)
    cambiado = True
    while cambiado:
        cambiado = False
        for val, minimo in ((True, min_si), (False, min_no)):
            for a, b in runs(list(flag == val)):
                if b - a < minimo and 0 < a and b < len(flag):
                    flag[a:b] = not val
                    cambiado = True
                    break
            if cambiado:
                break
    return flag


def hablando_por_muestra(segs, cfg):
    """Para cada muestra de pantalla partida: ¿Nacho está hablando? (se mueve su boca/cara) -> {t: bool}."""
    muestras = [m for sg in segs if sg["layout"] == "dividida" for m in sg.get("muestras", [])]
    acts = [m.get("act") for m in muestras]
    if sum(a is not None for a in acts) < 8:
        return None
    u = umbral_otsu(acts, cfg.get("habla_umbral_min", 9), cfg.get("habla_umbral_max", 15), cfg.get("habla_umbral", 12))
    log(f"Umbral de 'Nacho hablando': {u:.1f}")
    fs = cfg.get("_fs", 4)
    res = {}
    for sg in segs:
        if sg["layout"] != "dividida":
            continue
        ms = sg.get("muestras", [])
        vals = np.array([m.get("act") if m.get("act") is not None else np.nan for m in ms], float)
        if np.isnan(vals).all():
            continue
        malos = np.isnan(vals)
        if malos.any():
            vals[malos] = np.interp(np.flatnonzero(malos), np.flatnonzero(~malos), vals[~malos])
        v = max(1, int(round(fs * cfg.get("habla_ventana_seg", 1.75))))
        med = np.array([np.median(vals[max(0, k - v // 2):k + v // 2 + 1]) for k in range(len(vals))])
        flag = suavizar_bandera(med >= u, int(cfg.get("habla_min_seg", 3.0) * fs),
                                int(cfg.get("callado_min_seg", 2.0) * fs))
        for m, f in zip(ms, flag):
            res[round(m["t"], 3)] = bool(f)
    return res


def presentar_videos(segs, proy, pl, video=None):
    """Clips Fuera: mientras Nacho habla, el video va en pantalla dividida (Nacho arriba); cuando se calla para
    ver el video, el video va a pantalla completa. Si el video está quieto mientras Nacho habla, va en dividida."""
    cfg = pl.get("proyeccion", {})
    if not cfg.get("video_vertical_completo"):
        return proy
    fs = pl["deteccion"]["muestreo_fps"]
    cfg = dict(cfg, _fs=fs)
    habla = None
    if cfg.get("dividir_cuando_habla", True):
        habla = voz_de_nacho(video, segs, cfg) if video else None
        if habla:
            # suavizar por tramo de pantalla partida
            for sg in segs:
                ms = [m for m in sg.get("muestras", []) if round(m["t"], 3) in habla]
                if not ms:
                    continue
                f = suavizar_bandera([habla[round(m["t"], 3)] for m in ms], int(cfg.get("habla_min_seg", 3.0) * fs),
                                     int(cfg.get("callado_min_seg", 2.0) * fs))
                for m, v in zip(ms, f):
                    habla[round(m["t"], 3)] = bool(v)
        else:
            habla = hablando_por_muestra(segs, cfg)  # respaldo: movimiento de la boca

    # 1) ratos de pantalla partida con un video de celular quieto y nada más: también cuentan
    nuevos = list(proy)
    for sg in segs:
        if sg["layout"] != "dividida":
            continue
        libres, act = [], None
        for m in sg.get("muestras", []):
            c = m.get("caja")
            ocupado = any(q["inicio"] - 0.1 <= m["t"] < q["fin"] + 0.1 for q in proy)
            if c and c[3] > c[2] * 1.2 and not ocupado:
                if act and m["t"] - act["fin"] <= 1.5 / fs:
                    act["fin"] = m["t"] + 1 / fs
                    act["cajas"].append(c)
                else:
                    act = {"inicio": m["t"], "fin": m["t"] + 1 / fs, "cajas": [c]}
                    libres.append(act)
        for a in libres:
            fin = min(a["fin"], sg["fin"])
            if fin - a["inicio"] >= cfg.get("quieto_min_seg", 1.5):
                caja = [int(np.median([c[q] for c in a["cajas"]])) for q in range(4)]
                nuevos.append({"tipo": "imagen", "vertical": True, "inicio": round(a["inicio"], 3),
                               "fin": round(fin, 3), "caja": caja})
    if not habla:
        return sorted(nuevos, key=lambda x: x["inicio"])

    # 2) partir cada video según si Nacho habla (dividida) o no (completa)
    tiempos = sorted(habla)
    salida = []
    for p in sorted(nuevos, key=lambda x: x["inicio"]):
        es_video = p["tipo"] == "video" or (p["tipo"] == "imagen" and not p.get("documento"))
        if not es_video or p.get("pantalla"):
            salida.append(p)
            continue
        pts = [(t, habla[t]) for t in tiempos if p["inicio"] - 1e-6 <= t < p["fin"]]
        if not pts:
            salida.append(p)
            continue
        tramos = []
        for t, h in pts:
            if tramos and tramos[-1][2] == h:
                continue
            tramos.append([max(t, p["inicio"]), None, h])
        for k in range(len(tramos)):
            tramos[k][1] = tramos[k + 1][0] if k + 1 < len(tramos) else p["fin"]
        tramos[0][0] = p["inicio"]
        for a, b, h in tramos:
            if b - a < 0.3:
                continue
            q = dict(p, inicio=round(a, 3), fin=round(b, 3), pantalla="dividida" if h else "completa",
                     auto=True)
            salida.append(q)
    salida.sort(key=lambda x: x["inicio"])
    # tramos muy cortos toman la pantalla del anterior (evita cambios de un segundo)
    for k in range(1, len(salida)):
        q, ant = salida[k], salida[k - 1]
        if q.get("auto") and q["fin"] - q["inicio"] < cfg.get("tramo_min_seg", 1.5) and \
                ant.get("pantalla") and abs(ant["fin"] - q["inicio"]) < 0.5:
            q["pantalla"] = ant["pantalla"]
    # unir tramos seguidos que quedaron iguales
    unidos = []
    for q in salida:
        u = unidos[-1] if unidos else None
        if u and u.get("auto") and q.get("auto") and u.get("pantalla") == q.get("pantalla") and \
                u["tipo"] == q["tipo"] and bool(u.get("vertical")) == bool(q.get("vertical")) and \
                abs(u["fin"] - q["inicio"]) < 0.3 and u.get("caja") == q.get("caja"):
            u["fin"] = q["fin"]
        else:
            unidos.append(q)
    salida = unidos
    n_div = sum(1 for q in salida if q.get("pantalla") == "dividida")
    n_com = sum(1 for q in salida if q.get("pantalla") == "completa")
    log(f"Videos: {n_div} tramos en pantalla dividida (Nacho hablando) y {n_com} a pantalla completa")
    return sorted(salida, key=lambda x: x["inicio"])


def analizar_proyeccion(segs, pl):
    """Detecta en la proyección: videos corriendo (van a pantalla completa) e imágenes fijas nuevas (van en dividida)."""
    cfg = pl.get("proyeccion", {})
    modo = cfg.get("modo", "auto")
    if modo == "ocultar":
        return []
    umbral = pl["deteccion"]["umbral_movimiento_proyeccion"]
    fs = pl["deteccion"]["muestreo_fps"]
    pw = pl["fuente_16x9"]["proyeccion"][2]
    dt = 1.0 / fs
    salida = []
    for s in segs:
        if s["layout"] != "dividida":
            continue
        ms = s["muestras"]
        if len(ms) < 2:
            continue
        mov = np.array([m["mov"] for m in ms])
        mov[0] = 0.0
        es_video = mov > umbral
        # rellenar huecos cortos y quitar ráfagas cortas (cambios de diapositiva)
        hueco = int(cfg.get("video_hueco_seg", 1.5) * fs)
        for i, j in runs(list(~es_video)):
            if 0 < i and j < len(es_video) and (j - i) <= hueco:
                es_video[i:j] = True
        minimo = int(cfg.get("video_min_seg", 1.5) * fs)
        for i, j in runs(list(es_video)):
            if (j - i) < minimo:
                es_video[i:j] = False
        if cfg.get("video_vertical_completo"):
            # videos de celular (contenido vertical): una vez que empieza a moverse, se muestra completo
            # aunque tenga ratos quietos, hasta que cambie el contenido de la pantalla
            def misma(c1, c2):
                return c1 and c2 and all(abs(c1[q] - c2[q]) <= 25 for q in range(3))
            i = 0
            while i < len(ms):
                j = i + 1
                while j < len(ms) and misma(ms[j]["caja"], ms[i]["caja"]):
                    j += 1
                c = ms[i]["caja"]
                se_mueve = (mov[i:j] > umbral)
                if c and c[3] > c[2] * 1.2 and se_mueve.any():
                    k0 = i + int(np.argmax(se_mueve))
                    if (k0 - i) * dt <= cfg.get("video_arranque_seg", 2.0):
                        k0 = i
                    es_video[k0:j] = True
                i = j

        for i, j in runs(list(es_video)):
            sub = ms[i:j]
            cajas = [m["caja"] for m in sub if m["caja"]]
            caja = [int(np.median([c[k] for c in cajas])) for k in range(4)] if cajas else list(map(int, pl["fuente_16x9"]["proyeccion"]))
            item = {"tipo": "video", "inicio": round(sub[0]["t"], 3), "fin": round(min(sub[-1]["t"] + dt, s["fin"]), 3),
                    "caja": caja, "cara": pista_cara(sub, "cara_p", caja, fs)}
            con_cara = sum(1 for m in sub if m.get("cara_p")) / max(1, len(sub))
            item["con_cara"] = round(con_cara, 2)
            if con_cara < 0.3 and any(m.get("foco_m") for m in sub):
                # sin una persona clara: seguir lo que se mueve (muy suavizado)
                for m in sub:
                    fm = m.get("foco_m")
                    m["_foco"] = [fm[0] - 20, fm[1] - 20, 40, 40] if fm else None
                item["cara"] = pista_cara(sub, "_foco", caja, fs * 3)
            salida.append(item)

        # imágenes fijas: tramos quietos largos y anchos (no un video en pausa vertical)
        quieto = (mov <= umbral) & ~es_video
        for i, j in runs(list(quieto)):
            dur = (j - i) * dt
            if dur < cfg.get("imagen_min_seg", 6):
                continue
            sub = ms[i:j]
            cajas = [m["caja"] for m in sub if m["caja"]]
            if not cajas:
                continue
            caja = [int(np.median([c[k] for c in cajas])) for k in range(4)]
            if caja[2] < cfg.get("imagen_ancho_min", 0.6) * pw:
                continue
            if caja[3] > caja[2] * 1.2:
                continue  # un video vertical en pausa no es una imagen
            ini = sub[0]["t"]
            fin = s["fin"] if modo == "mostrar" else min(ini + cfg.get("imagen_mostrar_seg", 8), sub[-1]["t"] + dt)
            salida.append({"tipo": "imagen", "inicio": round(ini, 3), "fin": round(fin, 3), "caja": caja})
    return sorted(salida, key=lambda x: x["inicio"])


# ----------------------------------------------------------------------------
# Lectura: cuando Nacho lee un texto de la proyección, la cámara sigue la palabra que va leyendo
# ----------------------------------------------------------------------------
_OCR = None


def ocr_lineas(img):
    """Renglones de texto [(caja 4x2, texto, confianza)] con RapidOCR (funciona sin internet)."""
    global _OCR
    if _OCR is None:
        try:
            from rapidocr import RapidOCR
            _OCR = ("v3", RapidOCR())
        except Exception:
            try:
                from rapidocr_onnxruntime import RapidOCR
                _OCR = ("v1", RapidOCR())
            except Exception as e:
                log(f"OCR no disponible ({e}). Instala: pip install rapidocr onnxruntime")
                print("@@AVISO@@ No se hizo el seguimiento de lectura: falta el OCR. "
                      "En la terminal: python -m pip install rapidocr onnxruntime", file=sys.stderr, flush=True)
                _OCR = ("no", None)
    tipo, eng = _OCR
    if tipo == "no":
        return []
    try:
        if tipo == "v3":
            out = eng(img)
            if out is None or out.boxes is None:
                return []
            return [(np.array(b, dtype=np.float32), str(t), float(c)) for b, t, c in zip(out.boxes, out.txts, out.scores)]
        res, _ = eng(img)
        return [(np.array(r[0], dtype=np.float32), str(r[1]), float(r[2])) for r in (res or [])]
    except Exception as e:
        log(f"  (OCR falló: {e})")
        return []


def solo_letras(txt):
    txt = txt.lower()
    for a, b in zip("áéíóúüàèìòùâêîôû", "aeiouuaeiouaeiou"):
        txt = txt.replace(a, b)
    return re.sub(r"[^a-z0-9ñ]", "", txt)


def leer_documento(frame, caja, escala=1.5):
    """OCR del contenido de la proyección. Devuelve renglones con posición en coordenadas del video."""
    x, y, w, h = [int(v) for v in caja]
    rec = frame[y:y + h, x:x + w]
    if rec.size == 0:
        return []
    grande = cv2.resize(rec, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    renglones = []
    for b, texto, conf in ocr_lineas(grande):
        letras = solo_letras(texto)
        if conf < 0.5 or len(letras) < 3:
            continue
        b = b / escala
        x0, x1 = float(b[:, 0].min()) + x, float(b[:, 0].max()) + x
        y0, y1 = float(b[:, 1].min()) + y, float(b[:, 1].max()) + y
        renglones.append({"x0": x0, "x1": x1, "y0": y0, "y1": y1, "letras": letras})
    renglones.sort(key=lambda r: (round(r["y0"] / 12), r["x0"]))
    return renglones


def unir_renglones(listas):
    """Junta los renglones de varias lecturas OCR (cada una se salta algunos)."""
    todos = []
    for ren in listas:
        for r in ren:
            igual = None
            for q in todos:
                inter = min(q["y1"], r["y1"]) - max(q["y0"], r["y0"])
                alto = min(q["y1"] - q["y0"], r["y1"] - r["y0"])
                cruza_x = min(q["x1"], r["x1"]) - max(q["x0"], r["x0"]) > 0
                if alto > 0 and inter > 0.5 * alto and cruza_x:
                    igual = q
                    break
            if igual is None:
                todos.append(dict(r))
            elif len(r["letras"]) > len(igual["letras"]):
                igual.update(r)
    todos.sort(key=lambda r: (round(r["y0"] / 12), r["x0"]))
    return todos


def es_documento(renglones):
    return sum(1 for r in renglones if len(r["letras"]) >= 15) >= 4


def alinear_lectura(renglones, palabras):
    """Busca cada palabra dicha dentro del texto leído por OCR (letra por letra, sin espacios ni acentos).
    Devuelve puntos [t, x, y] de las palabras encontradas, en orden de lectura."""
    S, mapa = "", []
    for k, r in enumerate(renglones):
        for j, _ in enumerate(r["letras"]):
            mapa.append((k, j))
        S += r["letras"]
    puntos, ptr = [], 0
    for p in palabras:
        w = solo_letras(p["p"])
        if len(w) < 4:
            continue
        ini, fin = max(0, ptr - 30), ptr + 260
        j = S.find(w, ini, fin)
        if j < 0 and len(w) >= 6:
            # tolera errores pequeños del OCR o de la transcripción
            for var in (w[:-1], w[1:], w[:-2], w[2:]):
                j = S.find(var, ini, fin)
                if j >= 0:
                    break
        if j < 0:
            continue
        k, c = mapa[j + len(w) // 2]
        r = renglones[k]
        frac = (c + 0.5) / max(1, len(r["letras"]))
        x = r["x0"] + frac * (r["x1"] - r["x0"])
        y = (r["y0"] + r["y1"]) / 2
        puntos.append([round((p["i"] + p["f"]) / 2, 3), round(x, 1), round(y, 1), k])
        ptr = j + len(w)
    return puntos


def camino_camara(puntos, inicio, fin, alto_linea, cfg):
    """Recorrido suave de la cámara: avanza por el renglón mientras lee y baja al siguiente al cambiar."""
    paso = 1 / 15.0
    ts = np.arange(inicio, fin + paso, paso)
    tp = [p[0] for p in puntos]
    xs, ys = [], []
    for t in ts:
        k = int(np.searchsorted(tp, t, side="right")) - 1
        if k < 0:
            x, y = puntos[0][1], puntos[0][2]
        elif k >= len(puntos) - 1:
            x, y = puntos[-1][1], puntos[-1][2]
        else:
            a, b = puntos[k], puntos[k + 1]
            if abs(a[2] - b[2]) < 0.5 * alto_linea and b[1] >= a[1]:
                f = (t - a[0]) / max(1e-3, b[0] - a[0])
                x, y = a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f
            else:
                # cambio de renglón: se queda y salta poco antes de la siguiente palabra
                cambio = max(a[0], b[0] - 0.3)
                if t < cambio:
                    x, y = a[1], a[2]
                else:
                    f = (t - cambio) / max(1e-3, b[0] - cambio)
                    x, y = a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f
        xs.append(x)
        ys.append(y)
    # suavizado ida y vuelta (sin retraso)
    tau = cfg.get("suavizado_seg", 0.25)
    alfa = 1 - math.exp(-paso / max(1e-3, tau))
    def suave(v):
        ida = np.array(v, dtype=float)
        for i in range(1, len(ida)):
            ida[i] = ida[i - 1] + alfa * (ida[i] - ida[i - 1])
        vuelta = ida.copy()
        for i in range(len(vuelta) - 2, -1, -1):
            vuelta[i] = vuelta[i + 1] + alfa * (vuelta[i] - vuelta[i + 1])
        return vuelta
    xs, ys = suave(xs), suave(ys)
    return [[round(float(t), 3), round(float(x), 1), round(float(y), 1)] for t, x, y in zip(ts, xs, ys)]


def analizar_lectura(video, segs, proy, palabras, pl):
    """Detecta tramos en que Nacho lee un documento de la proyección y arma el recorrido de la cámara.
    También quita los falsos 'video' que en realidad son scroll de una página con texto."""
    cfg = pl.get("lectura", {})
    if not cfg.get("activar", True) or not palabras:
        return proy
    umbral = pl["deteccion"]["umbral_movimiento_proyeccion"]
    fs = pl["deteccion"]["muestreo_fps"]
    caja_proy = [int(v) for v in pl["fuente_16x9"]["proyeccion"]]
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    def cuadro(t):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, fr = cap.read()
        return fr if ok else None

    # 1) vistas quietas de la proyección
    vistas = []
    for s in segs:
        if s["layout"] != "dividida":
            continue
        ms = s["muestras"]
        quieto = [m["mov"] <= umbral for m in ms]
        quieto[0] = True
        for i, j in runs(quieto):
            ini, fin = ms[i]["t"], min(ms[j - 1]["t"] + 1 / fs, s["fin"])
            if fin - ini >= cfg.get("vista_min_seg", 1.0):
                cajas = [m["caja"] for m in ms[i:j] if m["caja"]]
                caja = [int(np.median([c[q] for c in cajas])) for q in range(4)] if cajas else caja_proy
                vistas.append({"inicio": ini, "fin": fin, "caja": caja})
    log(f"Leyendo el texto de {len(vistas)} vistas de la proyección...")

    # 2) OCR de cada vista y alineación con lo que se dice
    docs = []
    for n, v in enumerate(vistas):
        Progreso.paso(0.46 + 0.03 * n / max(1, len(vistas)), "Leyendo el texto de la pantalla")
        esc = cfg.get("ocr_escala", 1.5)
        dur = v["fin"] - v["inicio"]
        lecturas_ocr = []
        for t_ocr, e_ocr in ((v["inicio"] + min(0.6, dur / 2), esc), (v["inicio"] + min(0.6, dur / 2), esc * 1.33),
                             (v["inicio"] + dur / 2, esc)):
            fr = cuadro(t_ocr)
            if fr is not None:
                lecturas_ocr.append(leer_documento(fr, v["caja"], e_ocr))
            if dur < 2.0 and len(lecturas_ocr) >= 2:
                break
        ren = unir_renglones(lecturas_ocr)
        if not es_documento(ren):
            continue
        dichas = [p for p in palabras if v["inicio"] - 0.3 <= p["i"] < v["fin"] + 0.3]
        pts = alinear_lectura(ren, dichas)
        altos = [r["y1"] - r["y0"] for r in ren]
        centros = sorted((r["y0"] + r["y1"]) / 2 for r in ren)
        saltos = [b - a for a, b in zip(centros, centros[1:]) if b - a > 5]
        v.update(doc=True, puntos=pts, alto_linea=float(np.median(saltos)) if saltos else float(np.median(altos)) * 1.5,
                 n=n)
        docs.append(v)
    cap.release()

    # 3) los 'video' que son scroll de un documento ya no van a pantalla completa
    def toca_doc(a, b):
        return any(d["inicio"] - 1.5 <= b and a <= d["fin"] + 1.5 for d in docs)
    def vertical(p):  # video de celular: nunca es el scroll de una página
        c = p.get("caja")
        return bool(c and c[3] > c[2] * 1.2)
    proy = [p for p in proy if not (p["tipo"] == "video" and not vertical(p) and toca_doc(p["inicio"], p["fin"]))]

    # 4) tramos de lectura: palabras encontradas seguidas (huecos < 3 s), unidas entre vistas cercanas
    puntos = sorted((pt + [d["alto_linea"], d["caja"], d["n"]] for d in docs for pt in d["puntos"]), key=lambda x: x[0])
    tramos = []
    for pt in puntos:
        ant = tramos[-1][-1] if tramos else None
        # sigue el mismo tramo si no hay hueco largo y no se saltó una página que no se leyó
        if ant and pt[0] - ant[0] <= cfg.get("hueco_max_seg", 3.0) and pt[6] - ant[6] <= 1:
            tramos[-1].append(pt)
        else:
            tramos.append([pt])
    lecturas = []
    for tr in tramos:
        if len(tr) < cfg.get("min_palabras", 4):
            continue
        ini = max(0.0, tr[0][0] - cfg.get("margen_inicio_seg", 0.5))
        fin = tr[-1][0] + cfg.get("margen_fin_seg", 1.2)
        alto = float(np.median([x[4] for x in tr]))
        cajas = [x[5] for x in tr]
        caja = [int(np.median([c[q] for c in cajas])) for q in range(4)]
        # mientras la página se desplaza (scroll) no se muestra el texto
        ocultar = []
        for m in (m for sg in segs for m in sg.get("muestras", [])):
            if ini <= m["t"] <= fin and m["mov"] > umbral:
                a, b = m["t"] - 0.5, m["t"] + 0.4
                if ocultar and a <= ocultar[-1][1] + 0.8:
                    ocultar[-1][1] = b
                else:
                    ocultar.append([a, b])
        lecturas.append({"tipo": "lectura", "inicio": round(ini, 3), "fin": round(fin, 3), "caja": caja,
                         "alto_linea": round(alto, 1), "ocultar": [[round(a, 2), round(b, 2)] for a, b in ocultar],
                         "camino": camino_camara([x[:3] for x in tr], ini, fin, alto, cfg)})
    log(f"{len(lecturas)} tramos de lectura: " + ", ".join(f"{l['inicio']:.1f}-{l['fin']:.1f}s" for l in lecturas))

    # 5) lo que se cruce con una lectura cede su lugar
    def choca(p):
        return any(p["inicio"] < l["fin"] and l["inicio"] < p["fin"] for l in lecturas)
    proy = [p for p in proy if not choca(p)] + lecturas
    # si el documento apareció poco antes de empezar a leerlo, se muestra completo desde que aparece
    for l in lecturas:
        for sg in segs:
            if sg["layout"] == "dividida" and sg["inicio"] < l["inicio"] <= sg["fin"]:
                a = sg["inicio"] + 0.2
                if 0.8 <= l["inicio"] - a <= cfg.get("doc_previo_max_seg", 5.0) and \
                        not any(p["inicio"] < l["inicio"] and a < p["fin"] for p in proy if p is not l):
                    proy.append({"tipo": "imagen", "documento": True, "inicio": round(a, 3),
                                 "fin": l["inicio"], "caja": l["caja"]})
    # imágenes fijas que son documentos (no leídos): se muestran a todo lo ancho, sin cortar el texto
    for p in proy:
        if p["tipo"] == "imagen" and any(p["inicio"] < d["fin"] and d["inicio"] < p["fin"] for d in docs):
            p["documento"] = True
    return sorted(proy, key=lambda x: x["inicio"])


# ----------------------------------------------------------------------------
# Recorte de pausas
# ----------------------------------------------------------------------------
def calcular_cortes(video, info, proy, cfg):
    """Acorta las pausas (silencios relativos al nivel de la voz). No toca los videos de la proyección."""
    if not cfg.get("activar", True) or not info.get("audio"):
        return []
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", "8000",
                        "-f", "s16le", "-"], capture_output=True)
    a = np.frombuffer(r.stdout, np.int16).astype(np.float32)
    if len(a) < 8000:
        return []
    n = len(a) // 80
    rms = np.sqrt((a[:n * 80].reshape(n, 80) ** 2).mean(1))
    db = 20 * np.log10(rms / 32768 + 1e-9)
    db = np.convolve(db, np.ones(3) / 3, "same")
    ref = np.percentile(db, 70)
    bajo = db < ref - cfg.get("umbral_rel_db", 16)
    fps = info["fps"]
    dejar = cfg.get("dejar_seg", 0.12)
    cortes = []
    for i, j in runs(list(bajo)):
        ini, fin = i / 100, j / 100
        if fin - ini < cfg.get("pausa_min_seg", 0.2):
            continue
        a_c = ini + dejar / 2
        b_c = fin - dejar / 2
        if any(p["tipo"] == "video" and a_c < p["fin"] and b_c > p["inicio"] for p in proy):
            continue
        a_c = math.ceil(a_c * fps) / fps   # alinear al cuadro para no perder sincronía
        b_c = math.ceil(b_c * fps) / fps
        if b_c - a_c >= 1.5 / fps:
            cortes.append({"inicio": round(a_c, 4), "fin": round(b_c, 4)})
    log(f"{len(cortes)} pausas recortadas, {sum(c['fin'] - c['inicio'] for c in cortes):.1f}s en total")
    return cortes


# ----------------------------------------------------------------------------
# Transcripción (faster-whisper) con tiempos por palabra
# ----------------------------------------------------------------------------
def preparar_dlls_cuda():
    """En Windows, agrega al PATH las DLL de CUDA instaladas con pip (nvidia-cublas-cu12, nvidia-cudnn-cu12)."""
    if os.name != "nt":
        return
    import site
    rutas = []
    candidatos = site.getsitepackages() + [site.getusersitepackages()] + [p for p in sys.path if p.endswith("site-packages")]
    for sp in dict.fromkeys(candidatos):
        base = Path(sp) / "nvidia"
        if base.is_dir():
            rutas += [str(b) for b in base.glob("*/bin") if b.is_dir()]
    for r in rutas:
        try:
            os.add_dll_directory(r)
        except Exception:
            pass
    if rutas:
        os.environ["PATH"] = os.pathsep.join(rutas) + os.pathsep + os.environ.get("PATH", "")


def transcribir(video, modelo, dispositivo):
    preparar_dlls_cuda()
    from faster_whisper import WhisperModel
    if dispositivo == "auto":
        try:
            import ctranslate2
            dispositivo = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            dispositivo = "cpu"
    if SOLO_GPU and dispositivo != "cuda":
        raise SystemExit("Whisper no encuentra la tarjeta gráfica (CUDA). Instala: python -m pip install "
                         "nvidia-cublas-cu12 nvidia-cudnn-cu12==9.* y actualiza el driver de NVIDIA. No transcribo "
                         "con el procesador porque está activado «Solo tarjeta gráfica».")
    # el audio lo extrae ffmpeg (evita incompatibilidades de PyAV)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-map", "0:a:0", "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        raise SystemExit("No pude extraer el audio con ffmpeg: " + r.stderr.decode(errors="ignore")[-300:])
    audio = np.frombuffer(r.stdout, np.float32)

    def correr(disp):
        tipo = "float16" if disp == "cuda" else "int8"
        if disp == "cuda" and os.environ.get("WHISPER_COMPUTE"):
            tipo = os.environ["WHISPER_COMPUTE"]  # tarjetas con poca memoria: int8_float16
        log(f"Transcribiendo con whisper '{modelo}' en {disp}...")
        wm = WhisperModel(modelo, device=disp, compute_type=tipo, cpu_threads=int(HILOS_CPU) if SOLO_GPU else 0)
        # condition_on_previous_text=False y hallucination_silence_threshold evitan que repita frases ya dichas
        prompt = "Transcripción en español con puntuación: puntos, comas, ¿preguntas? y ¡exclamaciones!"
        # VAD más sensible: con música o ruido de fondo (entrevistas, videos de redes) no debe saltarse voces.
        # Sin hallucination_silence_threshold: en audio con ruido se brincaba frases reales; las repeticiones
        # inventadas las quita limpiar_transcripcion.
        segmentos, _ = wm.transcribe(audio, language="es", word_timestamps=True, vad_filter=True,
                                     vad_parameters=dict(threshold=0.35, min_silence_duration_ms=600,
                                                         speech_pad_ms=300),
                                     condition_on_previous_text=False, initial_prompt=prompt)
        out = []
        total_seg = max(1.0, len(audio) / 16000)
        for s in segmentos:
            Progreso.paso(0.20 + 0.20 * min(1.0, s.end / total_seg), "Transcribiendo el audio")
            for p in s.words or []:
                texto = p.word.strip()
                if texto:
                    out.append({"p": texto, "i": round(p.start, 3), "f": round(p.end, 3)})
        # 2a pasada: ratos donde se oye voz pero no quedó ninguna palabra -> se transcriben aparte
        huecos = huecos_con_voz(audio, out)
        if huecos:
            log(f"Revisando {len(huecos)} ratos con voz que se quedaron sin texto")
        for k, (a, b) in enumerate(huecos):
            Progreso.paso(0.40 + 0.05 * k / len(huecos), "Revisando partes sin subtítulos")
            i0, i1 = int(max(0, a - 0.3) * 16000), int(min(total_seg, b + 0.3) * 16000)
            trozo, _ = wm.transcribe(audio[i0:i1], language="es", word_timestamps=True, vad_filter=False,
                                     condition_on_previous_text=False, initial_prompt=prompt)
            base = i0 / 16000
            for s in trozo:
                if s.no_speech_prob > 0.75 and s.avg_logprob < -1.0:
                    continue  # ruido, no voz
                for p in s.words or []:
                    texto = p.word.strip()
                    if texto and p.probability > 0.15:
                        out.append({"p": texto, "i": round(base + p.start, 3), "f": round(base + p.end, 3)})
        out.sort(key=lambda w: w["i"])
        return out

    try:
        palabras = correr(dispositivo)
    except RuntimeError as e:
        if dispositivo != "cuda":
            raise
        if SOLO_GPU:
            raise SystemExit(f"La tarjeta gráfica falló al transcribir ({e}). Instala: python -m pip install "
                             "nvidia-cublas-cu12 nvidia-cudnn-cu12==9.* (no uso el procesador: «Solo tarjeta gráfica»)")
        log(f"La GPU falló ({e}). Sigo en CPU (más lento). Para usar la GPU: "
            "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*")
        palabras = correr("cpu")
    palabras = limpiar_transcripcion(palabras)
    log(f"{len(palabras)} palabras transcritas")
    return palabras


def huecos_con_voz(audio, palabras, sr=16000):
    """Tramos (s) con voz de 1.5 s o más donde whisper no dejó palabras (se saltó a alguien hablando)."""
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
        voz = get_speech_timestamps(audio, VadOptions(threshold=0.3, min_speech_duration_ms=400,
                                                      min_silence_duration_ms=500, speech_pad_ms=200))
        tramos = [(v["start"] / sr, v["end"] / sr) for v in voz]
    except Exception:
        # respaldo: energía por ventanas de 0.1 s
        n = len(audio) // 1600
        if n == 0:
            return []
        e = 10 * np.log10((audio[:n * 1600].reshape(n, 1600) ** 2).mean(1) + 1e-10)
        hay = e > max(np.percentile(e, 30) + 8, -45)
        tramos = [(a * 0.1, b * 0.1) for a, b in runs(list(hay))]
    cubiertos = [(p["i"], p["f"]) for p in palabras]
    huecos = []
    for a, b in tramos:
        # partes del tramo sin palabras cerca (0.4 s de tolerancia)
        libres = [(a, b)]
        for x, y in cubiertos:
            x, y = x - 0.4, y + 0.4
            nuevos = []
            for c, d in libres:
                if y <= c or x >= d:
                    nuevos.append((c, d))
                    continue
                if x > c:
                    nuevos.append((c, x))
                if y < d:
                    nuevos.append((y, d))
            libres = nuevos
        for c, d in libres:
            while d - c >= 1.5:
                fin = min(d, c + 28)  # whisper trabaja en ventanas de 30 s
                huecos.append((round(c, 2), round(fin, 2)))
                c = fin
    return huecos


def limpiar_transcripcion(palabras):
    """Quita errores típicos de whisper: frases repetidas, palabras amontonadas en un instante
    (imposibles de pronunciar tan rápido) y palabras con el tiempo hacia atrás."""
    ws = sorted([dict(p) for p in palabras if str(p.get("p", "")).strip()], key=lambda p: (p["i"], p["f"]))
    n = n_original = len(ws)
    if n < 3:
        return ws
    malo = [False] * n
    # 1) rachas de 4+ palabras cortísimas (< 0.13 s cada una) a más de 35 letras por segundo:
    #    whisper las amontonó en un instante (el habla rápida real llega a ~25)
    k = 0
    while k < n:
        m = k
        while m < n and ws[m]["f"] - ws[m]["i"] < 0.13 and (m == k or ws[m]["i"] - ws[m - 1]["f"] < 0.08):
            m += 1
        if m - k >= 4:
            letras = sum(len(solo_letras(w["p"])) for w in ws[k:m])
            if letras / max(0.05, ws[m - 1]["f"] - ws[k]["i"]) > 35:
                # si hay espacio libre alrededor para decirlas a ritmo normal, solo se reacomodan los tiempos
                # (whisper a veces amontona palabras reales); si no, eran inventadas y se quitan
                lim0 = ws[k - 1]["f"] if k > 0 else max(0.0, ws[k]["i"] - 1.0)
                lim1 = ws[m]["i"] if m < n else ws[m - 1]["f"] + 1.0
                necesita = letras / 15.0  # ritmo normal de habla
                centro = (ws[k]["i"] + ws[m - 1]["f"]) / 2
                ini = max(lim0, min(centro - necesita / 2, lim1 - necesita))
                fin = min(lim1, ini + necesita)
                if fin > ini and letras / (fin - ini) <= 22:
                    paso = (fin - ini) / (m - k)
                    for y in range(k, m):
                        ws[y]["i"] = round(ini + (y - k) * paso + 0.02, 3)
                        ws[y]["f"] = round(ini + (y - k + 1) * paso - 0.02, 3)
                else:
                    for y in range(k, m):
                        malo[y] = True
        k = max(m, k + 1)
    ws = [w for w, b in zip(ws, malo) if not b]
    n = len(ws)
    malo = [False] * n
    # 2) frases de 5+ palabras repetidas (hace menos de 60 s): si una copia va amontonada o encimada,
    #    o las dos están a menos de 20 s, se queda solo la que se dijo a ritmo normal
    norm = [solo_letras(w["p"]) for w in ws]
    L = 5

    def ritmo(a, m):
        letras = sum(len(x) for x in norm[a:a + m])
        return letras / max(0.05, ws[a + m - 1]["f"] - ws[a]["i"])

    k = 0
    while k <= n - L:
        if malo[k]:
            k += 1
            continue
        clave = norm[k:k + L]
        j = None
        for x in range(k - 1, -1, -1):
            if ws[k]["i"] - ws[x]["i"] > 60:
                break
            if norm[x:x + L] == clave:
                j = x
                break
        if j is None:
            k += 1
            continue
        m = 0  # largo de la repetición (puede pasar de 5 palabras)
        while k + m < n and j + m < k and norm[k + m] == norm[j + m]:
            m += 1
        rk, rj = ritmo(k, m), ritmo(j, m)
        encimada = k > 0 and ws[k]["i"] < ws[k - 1]["f"] - 0.1
        cerca = ws[k]["i"] - ws[j]["i"] < 20
        if rk > 20 or rj > 20 or encimada or cerca:
            quitar = j if rj > rk * 1.25 else k  # la copia más amontonada; si son parecidas, la segunda
            for y in range(quitar, quitar + m):
                malo[y] = True
        k += max(m, 1)
    # 3) palabras que empiezan antes de que termine la anterior (tiempo hacia atrás)
    out = []
    for k, w in enumerate(ws):
        if malo[k]:
            continue
        if out and w["i"] < out[-1]["i"] + 0.01 and w["f"] <= out[-1]["f"]:
            continue
        out.append(w)
    quitadas = n_original - len(out)
    if quitadas:
        log(f"Quité {quitadas} palabras repetidas o mal sincronizadas de la transcripción")
    return out


def agrupar_subtitulos(palabras, cfg):
    grupos, actual = [], []
    for p in palabras:
        if actual:
            pausa = p["i"] - actual[-1]["f"]
            fin_frase = re.search(r"[.,;:!?]$", actual[-1]["p"])
            if len(actual) >= cfg["max_palabras"] or pausa > cfg["pausa_corte_seg"] or fin_frase:
                grupos.append(actual)
                actual = []
        actual.append(p)
    if actual:
        grupos.append(actual)
    out = []
    for i, g in enumerate(grupos):
        fin = g[-1]["f"]
        if i + 1 < len(grupos) and grupos[i + 1][0]["i"] - fin < 0.35:
            fin = grupos[i + 1][0]["i"]
        out.append({"i": g[0]["i"], "f": fin, "palabras": g})
    return out


def frases(palabras, pausa=0.6):
    """Agrupa la transcripción en frases con tiempos (para elegir momentos)."""
    out, act = [], []
    for p in palabras:
        fin = re.search(r"[.!?]$", act[-1]["p"]) and \
            re.sub(r"[^a-z]", "", act[-1]["p"].lower()) not in ("mr", "mrs", "sr", "sra", "dr", "dra", "lic", "ing") \
            if act else False
        if act and (p["i"] - act[-1]["f"] > pausa or fin or len(act) >= 25):
            out.append(act)
            act = []
        act.append(p)
    if act:
        out.append(act)
    return [{"i": f[0]["i"], "f": f[-1]["f"], "texto": " ".join(x["p"] for x in f)} for f in out]


# ----------------------------------------------------------------------------
# IA local (Ollama): contexto del tema, momentos e imágenes verificadas
# ----------------------------------------------------------------------------
PREFERIDOS_TEXTO = ["qwen2.5:14b", "qwen3:14b", "gemma3:12b", "qwen2.5:7b", "llama3.1:8b", "mistral", "qwen2.5:3b"]
PREFERIDOS_VISION = ["qwen2.5vl:7b", "gemma3:12b", "gemma3:4b", "llava:7b", "llava"]


class IA:
    """Acceso a Ollama. modelo='auto' elige el mejor modelo instalado."""

    def __init__(self, url="http://localhost:11434", modelo="auto", modelo_vision="auto"):
        self.url = url.rstrip("/")
        self.instalados = self._instalados()
        self.modelo = self._elegir(modelo, PREFERIDOS_TEXTO)
        self.modelo_vision = self._elegir(modelo_vision, PREFERIDOS_VISION)
        if self.instalados is None:
            log("Ollama no está corriendo: no habrá análisis del tema ni verificación de imágenes")
        else:
            log(f"Modelo de texto: {self.modelo or 'ninguno'} | modelo de visión: {self.modelo_vision or 'ninguno'}")
            if not self.modelo_vision:
                log("Para verificar imágenes instala un modelo de visión: ollama pull qwen2.5vl:7b")

    def _instalados(self):
        import requests
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return None

    def _elegir(self, pedido, preferidos):
        if self.instalados is None or pedido == "none":
            return None
        nombres = set(self.instalados) | {n.split(":")[0] for n in self.instalados if n.endswith(":latest")}
        if pedido and pedido != "auto":
            if pedido in nombres or f"{pedido}:latest" in self.instalados:
                return pedido
            log(f"El modelo '{pedido}' no está instalado en Ollama; elijo otro")
        for m in preferidos:
            if m in nombres:
                return m
        return None

    @property
    def disponible(self):
        return bool(self.modelo)

    def json(self, prompt, imagenes=None, vision=False, timeout=300):
        import requests
        modelo = self.modelo_vision if vision else self.modelo
        if not modelo:
            raise RuntimeError("sin modelo")
        datos = {"model": modelo, "prompt": prompt, "stream": False, "format": "json",
                 "options": {"temperature": 0.1, "num_ctx": 8192}}
        if imagenes:
            datos["images"] = imagenes
        r = requests.post(f"{self.url}/api/generate", json=datos, timeout=timeout)
        r.raise_for_status()
        texto = r.json().get("response", "")
        texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S).strip()
        return json.loads(texto)


def texto_con_tiempos(palabras, cada=20):
    """Transcripción en renglones con su tiempo, para que el modelo la lea completa."""
    renglones, act = [], []
    for p in palabras:
        act.append(p)
        if len(act) >= cada or re.search(r"[.!?]$", p["p"]):
            renglones.append(f"[{act[0]['i']:.0f}s] " + " ".join(x["p"] for x in act))
            act = []
    if act:
        renglones.append(f"[{act[0]['i']:.0f}s] " + " ".join(x["p"] for x in act))
    return "\n".join(renglones)


def analizar_tema(palabras, titulo, ia):
    """Paso 1: entender de qué trata el video completo y con qué nombres correctos."""
    if not ia or not ia.disponible or not palabras:
        return {}
    prompt = (
        "Eres editor de noticias políticas en México. Lee COMPLETA esta transcripción automática de un video "
        "(puede tener nombres mal escritos por el reconocimiento de voz). Quien habla es el conductor del canal "
        "El Chapucero, Nacho Rodríguez; no lo incluyas como tema.\n"
        "Devuelve SOLO un JSON con esta forma:\n"
        "{\"tema\": \"una frase\", \"resumen\": \"3 a 5 frases\", "
        "\"personas\": [{\"nombre\": \"nombre completo bien escrito\", \"quien_es\": \"cargo o papel en la historia\"}], "
        "\"lugares\": [\"...\"], \"instituciones\": [\"...\"], \"hechos\": [\"hechos concretos que se narran\"], "
        "\"correcciones\": {\"palabra mal transcrita\": \"palabra correcta\"}}\n"
        "En 'correcciones' pon solo nombres propios que la transcripción escribió mal (ej. \"Ensinada\": \"Ensenada\").\n\n"
        f"Título del video: {titulo}\n\nTranscripción:\n{texto_con_tiempos(palabras)}"
    )
    try:
        ctx = ia.json(prompt)
        log(f"Tema: {ctx.get('tema', '?')}")
        nombres = [p.get("nombre") for p in ctx.get("personas", []) if isinstance(p, dict)]
        if nombres:
            log("Personas: " + ", ".join(n for n in nombres if n))
        return ctx
    except Exception as e:
        log(f"No pude analizar el tema ({e})")
        return {}


def aplicar_correcciones(palabras, correcciones):
    """Corrige nombres propios mal transcritos (solo coincidencias exactas de palabra)."""
    if not correcciones:
        return 0
    mapa = {}
    for mal, bien in correcciones.items():
        if isinstance(mal, str) and isinstance(bien, str) and mal.strip() and bien.strip() and len(mal.split()) == 1:
            mapa[mal.strip().lower()] = bien.strip()
    n = 0
    for p in palabras:
        m = re.match(r"^([¿¡\"']*)(.*?)([.,;:!?\"']*)$", p["p"])
        pre, nucleo, suf = m.groups()
        if nucleo.lower() in mapa:
            p["p"] = pre + mapa[nucleo.lower()] + suf
            n += 1
    if n:
        log(f"{n} palabras corregidas en la transcripción")
    return n


def elegir_momentos(palabras, duracion, titulo, cfg, proy, segs, ia, contexto):
    """Paso 2: con el contexto del tema, decide dónde van imágenes y qué buscar."""
    margen = cfg.get("margen_proyeccion", 2)
    bloqueado = [(p["inicio"] - margen, p["fin"] + margen) for p in proy]
    bloqueado += [(s["inicio"] - 1.0, s["inicio"] + 1.0) for s in segs[1:]]  # no tapar cambios de toma

    def libre(a, b):
        return all(b <= x or a >= y for x, y in bloqueado)

    fr = [f for f in frases(palabras) if libre(f["i"], f["i"] + cfg["duracion_min"])]
    maximo = max(1, int(duracion // cfg["cada_seg"]))
    candidatos = []

    if fr and ia and ia.disponible:
        lineas = "\n".join(f"{k}. [{f['i']:.1f}s] {f['texto']}" for k, f in enumerate(fr))
        prompt = (
            "Eres editor de videos cortos de noticias políticas en México. Vas a elegir dónde poner FOTOS de apoyo.\n"
            f"CONTEXTO DEL VIDEO (úsalo para entender de quién y de qué se habla):\n{json.dumps(contexto, ensure_ascii=False)}\n\n"
            f"Abajo está la transcripción numerada por frases. Elige hasta {maximo} frases donde se mencione a una persona, "
            "lugar, institución, objeto o hecho CONCRETO que se pueda mostrar con una foto real.\n"
            "Reglas para la búsqueda de imágenes:\n"
            "- Usa el NOMBRE COMPLETO y bien escrito según el contexto (ej. 'Ernesto Ruffo Appel', no 'Rufo').\n"
            "- Si la frase habla de algo que le pasa a una persona, combina persona + hecho (ej. 'Ernesto Ruffo Appel hospital').\n"
            "- Si menciona un objeto o lugar genérico, descríbelo concreto (ej. 'avión comercial Aeroméxico', 'carretera México').\n"
            "- NUNCA busques al conductor (Nacho Rodríguez), ni frases abstractas, ni el título del video.\n"
            "- Da dos búsquedas: una principal y una alternativa (otra forma de buscar LO MISMO, no algo más general).\n"
            "- En 'claves' pon 1 a 3 palabras que DEBEN aparecer en el título de la página de una foto correcta "
            "(normalmente el apellido de la persona o el nombre propio del lugar o institución). Si es un objeto genérico, deja [].\n"
            "- Si en una frase no hay nada concreto y reconocible que mostrar, NO la elijas. Es mejor pocas fotos correctas.\n"
            "Responde SOLO JSON: {\"momentos\": [{\"frase\": numero, \"que_mostrar\": \"descripción corta de la foto ideal\", "
            "\"busqueda\": \"...\", \"alternativa\": \"...\", \"claves\": [\"...\"]}]}\n\n"
            f"Título: {titulo}\n\nFrases:\n{lineas}"
        )
        res = {}
        for intento in range(2):  # si el modelo responde mal, otro intento
            try:
                res = ia.json(prompt)
                if res.get("momentos"):
                    break
            except Exception as e:
                log(f"El modelo no respondió bien al elegir momentos ({e})")
        try:
            for m in res.get("momentos", []):
                try:
                    k = int(m.get("frase", -1))
                except (TypeError, ValueError):
                    continue
                b = str(m.get("busqueda", "")).strip()
                if 0 <= k < len(fr) and b and "nacho" not in b.lower():
                    alt = str(m.get("alternativa", "")).strip()
                    claves = m.get("claves") or []
                    claves = [str(c).strip() for c in (claves if isinstance(claves, list) else [claves])
                              if str(c).strip() and "nacho" not in str(c).lower()][:3]
                    candidatos.append({"inicio": fr[k]["i"], "busqueda": b,
                                       "busquedas": [b] + ([alt] if alt and alt != b else []),
                                       "que_mostrar": str(m.get("que_mostrar", b)), "texto": fr[k]["texto"],
                                       "claves": claves})
            log(f"El modelo propuso {len(candidatos)} momentos")
        except Exception as e:
            log(f"No pude elegir momentos con el modelo ({e}); los reparto de forma pareja")

    if not candidatos and ia and ia.disponible:
        log("El modelo no encontró momentos claros para imágenes; no pongo imágenes al azar")
    elif not candidatos:
        t = max(cfg["inicio_min"], 8.0)
        while t + cfg["duracion_min"] < duracion and len(candidatos) < maximo:
            if libre(t, t + cfg["duracion_max"]):
                texto = " ".join(p["p"] for p in palabras if t <= p["i"] < t + cfg["duracion_max"])
                candidatos.append({"inicio": t, "busqueda": titulo, "busquedas": [titulo], "que_mostrar": titulo, "texto": texto})
                t += cfg["cada_seg"]
            else:
                t += 1.0

    estilos = cfg.get("estilos", ["dividida", "completa"])
    momentos = []
    for c in sorted(candidatos, key=lambda c: c["inicio"]):
        ini = max(c["inicio"], cfg["inicio_min"])
        if momentos and ini < momentos[-1]["fin"] + cfg["separacion_min"]:
            continue
        fin = min(ini + cfg["duracion_max"], duracion - 1.0)
        fines = [p["f"] for p in palabras if ini + cfg["duracion_min"] <= p["f"] <= ini + cfg["duracion_max"]]
        if fines:
            fin = min(fines[-1], fin)
        for x, y in bloqueado:  # no invadir la proyección
            if ini < x < fin:
                fin = x
        if fin - ini < cfg["duracion_min"] * 0.8 or not libre(ini, fin):
            continue
        momentos.append({"inicio": round(ini, 2), "fin": round(fin, 2), "estilo": estilos[len(momentos) % len(estilos)],
                         "busqueda": c["busqueda"], "busquedas": c["busquedas"], "que_mostrar": c["que_mostrar"],
                         "texto": c["texto"], "claves": c.get("claves", []),
                         "tema": (contexto or {}).get("tema", ""), "imagenes": []})
    return momentos[:maximo]


def calificar_imagen(img, busqueda, que_mostrar, ia, contexto="", titulo_pagina=""):
    """Paso 3: un modelo de visión revisa si la foto corresponde. Devuelve 0-10 (None si no hay modelo)."""
    import base64
    if not ia or not ia.modelo_vision:
        return None
    h, w = img.shape[:2]
    esc = 768 / max(h, w)
    peq = cv2.resize(img, (int(w * esc), int(h * esc))) if esc < 1 else img
    ok, buf = cv2.imencode(".jpg", peq, [cv2.IMWRITE_JPEG_QUALITY, 85])
    prompt = (
        "Eres editor de fotos de un noticiero mexicano. Esta imagen se usaría para ilustrar:\n"
        f"- Foto que se necesita: {que_mostrar}\n- Búsqueda: {busqueda}\n"
        + (f"- Tema del video: {contexto}\n" if contexto else "")
        + (f"- Título de la página de donde viene: {titulo_pagina}\n" if titulo_pagina else "")
        + "Primero describe qué se ve. Luego decide si CORRESPONDE a lo que se necesita: misma persona, lugar, "
        "institución u objeto (si es una persona y no la reconoces, apóyate en el título de la página). "
        "No corresponde si muestra a otra persona, otro tema, un meme, una caricatura, un logotipo, una captura con mucho "
        "texto, un collage, una gráfica, una marca de agua grande o si es de baja calidad.\n"
        "Responde SOLO JSON: {\"que_se_ve\": \"...\", \"corresponde\": true/false, \"puntaje\": 0-10}"
    )
    try:
        res = ia.json(prompt, imagenes=[base64.b64encode(buf.tobytes()).decode()], vision=True, timeout=120)
        puntaje = float(res.get("puntaje", 0))
        corresponde = res.get("corresponde")
        if corresponde is False or str(corresponde).lower() in ("false", "no"):
            puntaje = min(puntaje, 3.0)
        return puntaje
    except Exception as e:
        log(f"  (no pude verificar una imagen: {e})")
        return None


# Sitios que casi siempre dan imágenes inútiles (marca de agua, productos, collages, memes).
SITIOS_BLOQUEADOS = ("pinterest.", "pinimg.", "shutterstock.", "alamy.", "dreamstime.", "istockphoto.", "depositphotos.",
                     "123rf.", "gettyimages.", "freepik.", "vecteezy.", "aliexpress.", "amazon.", "mercadolibre.",
                     "ebay.", "temu.", "etsy.", "tiktok.", "facebook.", "fbsbx.", "lookaside.", "instagram.",
                     "memedroid.", "imgflip.", "cuantocabron.", "wallpaper", "clipart", "redbubble.")
_URLS_USADAS = set()


def menciona(texto, claves):
    """True si alguna clave (sin acentos ni mayúsculas) aparece en el texto."""
    if not claves:
        return True
    t = solo_letras(texto)
    return any(solo_letras(c) and solo_letras(c) in t for c in claves)


def buscar_imagenes(busquedas, n, destino, prefijo, ia=None, que_mostrar="", max_candidatos=8, puntaje_min=7,
                    claves=None, contexto="", usar_mejor=False):
    """Busca en internet, descarga candidatas, las verifica con el modelo de visión y guarda las mejores n.
    claves: palabras que deben aparecer en el título o la dirección de la página (filtra fotos de otra cosa).
    usar_mejor: si ninguna pasa la verificación, usa la mejor que hubo (para búsquedas hechas a mano)."""
    import requests
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    if isinstance(busquedas, str):
        busquedas = [busquedas]
    destino.mkdir(parents=True, exist_ok=True)
    candidatas, vistos = [], set(_URLS_USADAS)
    verificar = bool(ia and ia.modelo_vision)
    descartadas = 0
    for consulta in busquedas:
        try:
            resultados = DDGS().images(consulta, region="mx-es", safesearch="moderate", max_results=max_candidatos * 4)
        except Exception as e:
            log(f"Búsqueda '{consulta}' falló: {e}")
            continue
        revisadas = 0
        for r in resultados:
            if revisadas >= max_candidatos or (not verificar and len(candidatas) >= n):
                break
            url = r.get("image")
            if not url or url in vistos:
                continue
            vistos.add(url)
            pagina = f"{r.get('title', '')} {r.get('url', '')}"
            if any(b in (url + " " + str(r.get("url", ""))).lower() for b in SITIOS_BLOQUEADOS):
                descartadas += 1
                continue
            if not menciona(pagina + " " + url, claves):
                descartadas += 1  # la página no habla de quien/lo que buscamos
                continue
            try:
                ancho, alto = int(r.get("width") or 0), int(r.get("height") or 0)
                if ancho and alto and (ancho < 600 or alto < 400):
                    continue
                resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                img = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
                if img is None or img.shape[1] < 600 or img.shape[0] < 400:
                    continue
                revisadas += 1
                puntaje = calificar_imagen(img, consulta, que_mostrar or consulta, ia, contexto,
                                           r.get("title", "")) if verificar else None
                candidatas.append((puntaje if puntaje is not None else 5.0, img, url))
                if verificar:
                    log(f"    candidata {len(candidatas)}: puntaje {puntaje}  ({str(r.get('title', ''))[:60]})")
                    if sum(1 for c in candidatas if c[0] >= 8) >= n:  # ya hay suficientes muy buenas
                        break
            except Exception:
                continue
        buenas = [c for c in candidatas if not verificar or c[0] >= puntaje_min]
        if len(buenas) >= n:
            break
    if descartadas:
        log(f"    {descartadas} resultados descartados por sitio o porque la página no menciona {claves}")
    candidatas.sort(key=lambda c: -c[0])
    elegidas = [c for c in candidatas if not verificar or c[0] >= puntaje_min][:n]
    if verificar and not elegidas and candidatas:
        if usar_mejor:
            log("    ninguna candidata pasó la verificación; uso la mejor que hubo")
            elegidas = candidatas[:1]
        else:
            log("    ninguna candidata pasó la verificación; mejor sin imagen que una equivocada")
    guardadas = []
    for k, (puntaje, img, url) in enumerate(elegidas, 1):
        p = destino / f"{prefijo}_{k:02d}.jpg"
        if guardar_img(p, img):
            guardadas.append(str(p))
            _URLS_USADAS.add(url)
        else:
            log(f"No pude guardar {p}")
    return guardadas


def buscar_para_momento(m, i, trabajo, cfg, ia):
    m["imagenes"] = buscar_imagenes(m.get("busquedas") or [m["busqueda"]], cfg["imagenes_por_momento"],
                                    Path(trabajo) / "imagenes", f"momento{i:02d}", ia, m.get("que_mostrar", ""),
                                    cfg.get("max_candidatos", 8), cfg.get("puntaje_min", 7),
                                    claves=m.get("claves"), contexto=m.get("tema", ""))
    if not m["imagenes"] and m.get("claves"):
        # segundo intento: búsqueda solo con las claves (ej. el nombre de la persona)
        m["imagenes"] = buscar_imagenes([" ".join(m["claves"])], cfg["imagenes_por_momento"],
                                        Path(trabajo) / "imagenes", f"momento{i:02d}", ia, m.get("que_mostrar", ""),
                                        cfg.get("max_candidatos", 8), cfg.get("puntaje_min", 7),
                                        claves=m.get("claves"), contexto=m.get("tema", ""))
    log(f"Momento {i} ({m['inicio']:.1f}-{m['fin']:.1f}s) '{m['busqueda']}': {len(m['imagenes'])} imágenes")


def asignar_imagenes(momentos, cfg, args, trabajo, ia):
    n = cfg["imagenes_por_momento"]
    if args.imagenes:
        lista = sorted(str(p) for p in Path(args.imagenes).iterdir() if p.suffix.lower() in EXT_IMG)
        if not lista:
            log("La carpeta de imágenes está vacía")
            return
        k = 0
        for m in momentos:
            m["imagenes"] = [lista[(k + j) % len(lista)] for j in range(n)]
            k += n
        return
    if args.sin_web:
        return
    for i, m in enumerate(momentos, 1):
        Progreso.paso(0.62 + 0.36 * (i - 1) / max(1, len(momentos)), f"Buscando imágenes ({i} de {len(momentos)})")
        buscar_para_momento(m, i, trabajo, cfg, ia)


EXT_VIDEO = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".flv", ".wmv"]


def duracion_video(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 0.0


def asegurar_mp4(video):
    """Si el video no es .mp4 lo convierte a MP4 (H.264 + AAC, cuadros fijos) junto al original.
    Si ya existe una conversión más nueva que el original, la reutiliza."""
    video = Path(video)
    if video.suffix.lower() == ".mp4":
        return video
    destino = video.with_suffix(".mp4")
    if os.environ.get("CONVERTIR_EN"):  # Subtitulador: la conversión va a la carpeta de trabajo, no junto al original
        destino = Path(os.environ["CONVERTIR_EN"]) / (video.stem + ".mp4")
    if destino.exists() and destino.stat().st_mtime >= video.stat().st_mtime and duracion_video(destino) > 0:
        log(f"Uso la conversión que ya existe: {destino}")
        return destino
    log(f"Convirtiendo {video.name} a MP4...")
    total = max(1.0, duracion_video(video))
    temporal = destino.with_name(destino.stem + ".convirtiendo.mp4")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0?",
           *(["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "19", "-b:v", "0"] if SOLO_GPU and nvenc_disponible() else
             ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]),
           "-pix_fmt", "yuv420p", "-fps_mode", "cfr",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
           "-progress", "pipe:1", "-nostats", str(temporal)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace")
    for linea in proc.stdout:
        if linea.startswith("out_time_us=") or linea.startswith("out_time_ms="):
            try:
                seg = int(linea.split("=", 1)[1]) / 1e6
                Progreso.paso(seg / total, "Convirtiendo el video a MP4")
            except ValueError:
                pass
    err = proc.stderr.read()
    if proc.wait() != 0 or not temporal.exists():
        if temporal.exists():
            temporal.unlink()
        raise SystemExit("No pude convertir el video a MP4: " + err[-400:])
    os.replace(temporal, destino)
    Progreso.paso(1.0, "Conversión terminada")
    log(f"Convertido: {destino}")
    return destino


def unir_cortes(cortes):
    """Cortes ordenados y sin encimarse."""
    out = []
    for c in sorted(cortes or [], key=lambda c: c["inicio"]):
        if out and c["inicio"] <= out[-1]["fin"]:
            out[-1]["fin"] = max(out[-1]["fin"], c["fin"])
        else:
            out.append({"inicio": c["inicio"], "fin": c["fin"]})
    return out


def a_editado(t, cortes):
    """Tiempo del video original -> tiempo en el video ya editado (sin las pausas quitadas)."""
    quitado = 0.0
    for c in unir_cortes(cortes):
        if t >= c["fin"]:
            quitado += c["fin"] - c["inicio"]
        elif t > c["inicio"]:
            quitado += t - c["inicio"]
            break
        else:
            break
    return max(0.0, t - quitado)


def a_original(te, cortes):
    """Tiempo en el video ya editado -> tiempo del video original."""
    t = float(te)
    for c in unir_cortes(cortes):
        if t >= c["inicio"] - 1e-9:
            t += c["fin"] - c["inicio"]
        else:
            break
    return t


def estilo_solo_subtitulos():
    """Módulo del estilo 'Solo subtítulos' (a un clip ya editado solo le pone subtítulos)."""
    import solo_subtitulos
    solo_subtitulos.ed = sys.modules[__name__]
    return solo_subtitulos


def estilo_motivacional():
    """Módulo del estilo 'Tiktoks Motivacionales' (usa las funciones de este archivo)."""
    import motivacional
    motivacional.ed = sys.modules[__name__]
    return motivacional


def cmd_convertir(args):
    return asegurar_mp4(Path(args.video.strip().strip('"')).resolve())


VERSION = "2026-10-15 voz-cv"


def cmd_analizar(args):
    log(f"editor.py versión {VERSION}")
    video = Path(args.video.strip().strip('"')).resolve()
    if video.exists() and video.suffix.lower() != ".mp4":
        video = asegurar_mp4(video)
    if not video.exists():
        raise SystemExit(f"No existe el video: {video}")
    pl_path = ruta(args.plantilla)
    pl = cargar_json(pl_path)
    trabajo = Path(args.trabajo) if args.trabajo else video.with_name(video.stem + "_trabajo")
    trabajo.mkdir(parents=True, exist_ok=True)
    if pl.get("estilo") == "motivacional":
        return estilo_motivacional().analizar(args, video, pl, pl_path, trabajo)
    if pl.get("estilo") == "solo_subtitulos":
        return estilo_solo_subtitulos().analizar(args, video, pl, pl_path, trabajo)

    Progreso.paso(0.0, "Analizando las tomas del video")
    segs, info = analizar_tomas(video, pl)
    proy = analizar_proyeccion(segs, pl)
    for s in segs:
        log(f"  toma {s['layout']:9s} {s['inicio']:7.2f} -> {s['fin']:7.2f}")

    if args.transcripcion:
        palabras = cargar_json(args.transcripcion)
    elif args.sin_subtitulos or not info["audio"]:
        palabras = []
    else:
        palabras = transcribir(video, args.whisper_modelo, args.dispositivo)

    Progreso.paso(0.46, "Leyendo el texto de la pantalla")
    proy = analizar_lectura(video, segs, proy, palabras, pl)
    proy = presentar_videos(segs, proy, pl, video)
    proy = mostrar_al_que_habla(segs, proy, pl, video)
    for s in segs:
        s.pop("muestras", None)
    for p in proy:
        log(f"  proyección {p['tipo']:8s} {p['inicio']:7.2f} -> {p['fin']:7.2f}")
    Progreso.paso(0.49, "Recortando pausas")
    cortes = [] if args.sin_cortes else calcular_cortes(video, info, proy, pl.get("cortes", {}))
    Progreso.paso(0.50, "Entendiendo el tema del video")

    ia = None if args.sin_ia else IA(args.url_ollama, args.modelo_ollama, "none" if args.sin_verificar else args.modelo_vision)
    contexto = analizar_tema(palabras, args.titulo, ia)
    aplicar_correcciones(palabras, contexto.get("correcciones"))
    guardar_json(trabajo / "transcripcion.json", palabras)
    if (trabajo / "subtitulos.json").exists():
        (trabajo / "subtitulos.json").unlink()  # la transcripción es nueva

    Progreso.paso(0.58, "Eligiendo dónde van las imágenes")
    if pl["momentos"].get("activar", True):
        momentos = elegir_momentos(palabras, info["duracion"], args.titulo, pl["momentos"], proy, segs, ia, contexto)
        asignar_imagenes(momentos, pl["momentos"], args, trabajo, ia)
    else:
        momentos = []  # este estilo no lleva imágenes de internet (se agregan a mano si hace falta)
    if not args.imagenes and not args.sin_web:
        sin = [m for m in momentos if not m["imagenes"]]
        if sin:
            log(f"{len(sin)} momentos se quedan sin imagen porque no encontré fotos correctas")
        momentos = [m for m in momentos if m["imagenes"]]
    for m in momentos:
        log(f"  momento {m['estilo']:9s} {m['inicio']:7.2f} -> {m['fin']:7.2f}  '{m['busqueda']}'")

    plan = {
        "video": str(video),
        "plantilla": str(pl_path),
        "titulo": args.titulo,
        "info": info,
        "contexto": contexto,
        "tomas": segs,
        "proyeccion": proy,
        "momentos": momentos,
        "cortes": cortes,
        "_ayuda": ("Puedes editar: 'proyeccion' (borrar o cambiar inicio/fin), 'momentos' (inicio, fin, "
                   "estilo dividida|completa, imagenes: rutas), 'cortes' (borrar los que no quieras) "
                   "y volver a correr 'render'. O usa interfaz.py para pedirlo por chat."),
    }
    guardar_json(trabajo / "plan.json", plan)
    log(f"Plan guardado en {trabajo / 'plan.json'}")
    Progreso.paso(1.0, "Análisis terminado")
    return trabajo


# ----------------------------------------------------------------------------
# Render
# ----------------------------------------------------------------------------
def fuente_que_quepa(path_fuente, texto, ancho, alto, tam_max, contorno):
    tam = tam_max
    while tam > 12:
        f = ImageFont.truetype(str(path_fuente), tam)
        x0, y0, x1, y1 = f.getbbox(texto, stroke_width=contorno)
        if x1 - x0 <= ancho and y1 - y0 <= alto:
            return f
        tam -= 2
    return ImageFont.truetype(str(path_fuente), tam)


def a_superposicion(img_rgba):
    """Convierte RGBA (PIL) en (bgr, alpha, bbox) recortado al área visible."""
    bbox = img_rgba.getbbox()
    if bbox is None:
        return None
    rec = np.array(img_rgba.crop(bbox))
    bgr = rec[:, :, 2::-1].astype(np.float32)
    alpha = rec[:, :, 3:4].astype(np.float32) / 255.0
    return bgr, alpha, bbox


def pegar(lienzo, sup, dx=0, dy=0):
    if sup is None:
        return
    bgr, alpha, (x0, y0, x1, y1) = sup
    x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
    H, W = lienzo.shape[:2]
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    b = bgr[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    a = alpha[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    zona = lienzo[cy0:cy1, cx0:cx1].astype(np.float32)
    lienzo[cy0:cy1, cx0:cx1] = (b * a + zona * (1 - a)).astype(np.uint8)


def capa_titulo(pl, titulo, W, H):
    cfg = pl["titulo"]
    capa = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    barra = Image.open(ruta(cfg["barra"])).convert("RGBA")
    if barra.size != (W, H):
        barra = barra.resize((W, H), Image.LANCZOS)
    x, y, w, h = cfg["caja_texto"]
    esc = cfg.get("escala", 1.0)
    if abs(esc - 1.0) > 1e-3:
        # agranda la barra desde su centro; el texto escala con ella
        bx0, by0, bx1, by1 = barra.getbbox()
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        rec = barra.crop((bx0, by0, bx1, by1))
        rec = rec.resize((int(rec.width * esc), int(rec.height * esc)), Image.LANCZOS)
        barra = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        barra.alpha_composite(rec, (int(cx - rec.width / 2), max(0, int(cy - rec.height / 2))))
        x, y = cx - (cx - x) * esc, max(0, cy - (cy - y) * esc)
        w, h = w * esc, h * esc
    capa.alpha_composite(barra)
    if titulo:
        texto = titulo.upper() if cfg.get("mayusculas") else titulo
        f = fuente_que_quepa(ruta(cfg["fuente"]), texto, w, h, cfg["tam_max"], cfg["contorno"])
        d = ImageDraw.Draw(capa)
        d.text((x + w / 2, y + h / 2), texto, font=f, fill=hex_rgb(cfg["color"]), anchor="mm",
               stroke_width=cfg["contorno"], stroke_fill=hex_rgb(cfg["color_contorno"]))
    return a_superposicion(capa)


class Subtitulos:
    def __init__(self, grupos, cfg):
        self.grupos = sorted(grupos, key=lambda g: g["i"])
        self.cfg = cfg
        self.cache = {}
        self.fuente_path = ruta(cfg["fuente"])
        self.fuentes = {}

    def fuente(self, tam):
        if tam not in self.fuentes:
            self.fuentes[tam] = ImageFont.truetype(str(self.fuente_path), tam)
        return self.fuentes[tam]

    def activo(self, t):
        # si dos se enciman gana el que empezó al último (igual que la vista previa del editor)
        elegido = None
        for gi, g in enumerate(self.grupos):
            if g["i"] > t:
                break
            if t < g["f"]:
                elegido = gi
        if elegido is None:
            return None
        wi = 0
        for k, p in enumerate(self.grupos[elegido]["palabras"]):
            if p["i"] <= t:
                wi = k
        return elegido, wi

    def _lineas(self, palabras, ancho, saltos=None):
        cfg, c = self.cfg, self.cfg["contorno"]
        tam = cfg["tam"]
        while True:
            f = self.fuente(tam)
            hb = f.getbbox("H")
            esp = (hb[3] - hb[1]) * cfg.get("espacio", 0.28)
            anchos = [f.getbbox(p, stroke_width=c)[2] - f.getbbox(p, stroke_width=c)[0] for p in palabras]
            if saltos:
                # renglones que puso el usuario: se respetan; si uno no cabe, se achica la letra
                lineas, act = [], []
                for k in range(len(palabras)):
                    act.append(k)
                    if k in saltos:
                        lineas.append(act)
                        act = []
                if act:
                    lineas.append(act)
                mayor = max(sum(anchos[k] for k in ln) + esp * (len(ln) - 1) for ln in lineas)
                if mayor <= ancho or tam <= cfg["tam_min"]:
                    return f, esp, anchos, lineas
                tam -= 4
                continue
            lineas, act, w_act = [], [], 0
            for k, aw in enumerate(anchos):
                nuevo = aw if not act else w_act + esp + aw
                if act and nuevo > ancho:
                    lineas.append(act)
                    act, w_act = [k], aw
                else:
                    act.append(k)
                    w_act = nuevo
            if act:
                lineas.append(act)
            cabe = len(lineas) <= cfg["max_lineas"] and max(anchos) <= ancho
            if cabe or tam <= cfg["tam_min"]:
                return f, esp, anchos, lineas
            tam -= 4

    def imagen(self, gi, wi, w, h):
        clave = (gi, wi, w, h)
        if clave in self.cache:
            return self.cache[clave]
        cfg, c = self.cfg, self.cfg["contorno"]
        palabras = [p["p"] for p in self.grupos[gi]["palabras"]]
        saltos = {k for k, p in enumerate(self.grupos[gi]["palabras"][:-1]) if p.get("salto")}
        if cfg.get("mayusculas"):
            palabras = [p.upper() for p in palabras]
        f, esp, anchos, lineas = self._lineas(palabras, w, saltos)
        hb = f.getbbox("H")
        cap = hb[3] - hb[1]
        alto_linea = cap * 1.45
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        y_centro0 = h / 2 - alto_linea * (len(lineas) - 1) / 2
        for li, linea in enumerate(lineas):
            total = sum(anchos[k] for k in linea) + esp * (len(linea) - 1)
            x = (w - total) / 2
            yc = y_centro0 + li * alto_linea
            base = yc + cap / 2
            for k in linea:
                if k == wi:
                    d.rounded_rectangle([x - 6, yc - cap * 0.95, x + anchos[k] + 6, yc + cap * 0.95],
                                        radius=10, fill=hex_rgb(cfg["color_resalte"]))
                d.text((x - f.getbbox(palabras[k], stroke_width=c)[0], base), palabras[k], font=f, anchor="ls",
                       fill=hex_rgb(cfg["color"]), stroke_width=c, stroke_fill=hex_rgb(cfg["color_contorno"]))
                x += anchos[k] + esp
        sup = a_superposicion(img)
        self.cache[clave] = sup
        return sup


def interp_cara(pista, t):
    if len(pista) == 1:
        return pista[0][1:]
    ts = [p[0] for p in pista]
    return [float(np.interp(t, ts, [p[k] for p in pista])) for k in (1, 2, 3)]


def recorte_persona(region, cara, area_wh, cfg):
    """Ventana de recorte dentro de 'region' para que la persona quede bien enmarcada."""
    rx, ry, rw, rh = region
    aw, ah = area_wh
    asp = aw / ah
    zoom = max(1.0, cfg.get("zoom", 1.0))
    w = rw / zoom
    h = w / asp
    if h > rh / zoom:
        h = rh / zoom
        w = h * asp
    cx, cy, fh = cara
    x0 = cx - w / 2
    y0 = cy - cfg.get("cara_y", 0.38) * h
    # la cabeza (pelo incluido) no debe cortarse ni quedar bajo el título
    tope_cabeza = cy - fh * 1.1
    margen = cfg.get("margen_sup", 0) * (h / ah)
    if tope_cabeza - margen < y0:
        y0 = tope_cabeza - margen
    x0 = min(max(x0, rx), rx + rw - w)
    y0 = min(max(y0, ry), ry + rh - h)
    return int(round(x0)), int(round(y0)), int(round(w)), int(round(h))


def persona_contenida(frame, region, cara, aw, ah, cfg):
    """Recuadro angosto de la persona (p. ej. el de un video de celular) sin acercarlo de más:
    al centro, con la cabeza completa bajo el título, y a los lados el mismo cuadro ampliado y desenfocado."""
    rx, ry, rw, rh = [int(v) for v in region]
    panel = frame[ry:ry + rh, rx:rx + rw]
    cx, cy, fh = cara
    s = aw * cfg.get("ancho_panel", 0.593) / rw
    pw, ph = int(round(rw * s)), int(round(rh * s))
    nitido = cv2.resize(panel, (pw, ph), interpolation=cv2.INTER_LANCZOS4)
    # fondo: el panel cubriendo todo el recuadro, centrado en la cara y muy desenfocado
    sf = max(aw / rw, ah / rh) * 1.15
    fondo = cv2.resize(panel, (int(rw * sf), int(rh * sf)), interpolation=cv2.INTER_AREA)
    fx = int(min(max((cx - rx) * sf - aw / 2, 0), fondo.shape[1] - aw))
    fy = int(min(max((cy - ry) * sf - ah * 0.4, 0), fondo.shape[0] - ah))
    fondo = cv2.GaussianBlur(fondo[fy:fy + ah, fx:fx + aw], (0, 0), cfg.get("desenfoque", 22))
    fondo = (fondo * cfg.get("oscurecer_fondo", 0.85)).astype(np.uint8)
    # la cabeza (pelo incluido) queda justo bajo el título
    tope_cabeza = (cy - ry - fh * 0.9) * s
    y_off = int(round(cfg.get("margen_sup_panel", 110) - tope_cabeza))
    y_off = min(0, max(ah - ph, y_off))  # sin dejar huecos arriba ni abajo
    x_off = int(round((aw - pw) / 2))
    ys0, ys1 = max(0, -y_off), min(ph, ah - y_off)
    fondo[y_off + ys0:y_off + ys1, max(0, x_off):max(0, x_off) + min(pw, aw)] = \
        nitido[ys0:ys1, max(0, -x_off):max(0, -x_off) + min(pw, aw)]
    return fondo


def ajustar(img, w, h, modo="cubrir", foco_y=0.5, zoom=1.0):
    ih, iw = img.shape[:2]
    if modo == "contener":
        esc = min(w / iw, h / ih)
        fondo = cv2.GaussianBlur(ajustar(img, w, h, "cubrir"), (0, 0), 25)
        fondo = (fondo * 0.55).astype(np.uint8)
        nw, nh = int(iw * esc), int(ih * esc)
        peq = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_CUBIC)
        x, y = (w - nw) // 2, (h - nh) // 2
        fondo[y:y + nh, x:x + nw] = peq
        return fondo
    esc = max(w / iw, h / ih) * zoom
    cw, ch = w / esc, h / esc
    x0 = (iw - cw) / 2
    y0 = (ih - ch) * foco_y
    rec = img[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]
    return cv2.resize(rec, (w, h), interpolation=cv2.INTER_AREA if esc < 1 else cv2.INTER_CUBIC)


def enfocar(img, cantidad):
    if cantidad <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)
    return cv2.addWeighted(img, 1 + cantidad, blur, -cantidad, 0)


EFECTOS = ["zoom_in", "zoom_out", "paneo_der", "paneo_izq", "paneo_arriba", "paneo_abajo", "ninguno"]
CICLO_AUTO = ["zoom_in", "paneo_der", "zoom_out", "paneo_izq"]
_DETECTOR = None


def foco_auto(img):
    """Centro sugerido de una foto: la cara más grande si hay, si no el centro un poco arriba."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = DetectorCara()
    h, w = img.shape[:2]
    esc = min(1.0, 1000 / max(h, w))
    peq = cv2.resize(img, (int(w * esc), int(h * esc))) if esc < 1 else img
    c = _DETECTOR.mayor(peq)
    if c and c[3] > 0.05 * peq.shape[0]:
        return round((c[0] + c[2] / 2) / peq.shape[1], 3), round((c[1] + c[3] / 2) / peq.shape[0], 3)
    return 0.5, 0.45


def encuadre_de(m, k, idx_global, cfg_m, img=None):
    """Parámetros de encuadre y movimiento de la imagen k de un momento (con valores por defecto)."""
    lista = m.get("encuadres") or []
    e = dict(lista[k]) if 0 <= k < len(lista) and isinstance(lista[k], dict) else {}
    efecto = e.get("efecto") or cfg_m.get("efecto", "auto")
    if efecto == "auto" or efecto not in EFECTOS:
        efecto = CICLO_AUTO[idx_global % len(CICLO_AUTO)]
    e["efecto"] = efecto
    e.setdefault("intensidad", cfg_m.get("intensidad", 0.25))
    e.setdefault("zoom", 1.0)
    if "fx" not in e or "fy" not in e:
        e["fx"], e["fy"] = foco_auto(img) if img is not None else (0.5, 0.45)
    return e


def ventana(iw, ih, w, h, fase, e):
    """Parte de la foto (x0, y0, ancho, alto, escala) que se ve en un cuadro w x h. fase 0..1."""
    base = max(w / iw, h / ih)
    z = max(1.0, float(e.get("zoom", 1.0)))
    I = max(0.0, float(e.get("intensidad", 0.25)))
    ef = e.get("efecto", "zoom_in")
    p = min(1.0, max(0.0, fase))
    dx = dy = 0.0
    if ef == "zoom_in":
        zz = z * (1 + I * p)
    elif ef == "zoom_out":
        zz = z * (1 + I * (1 - p))
    elif ef.startswith("paneo"):
        zz = z * (1 + I * 0.6)
    else:
        zz = z
    s = base * zz
    cw, ch = w / s, h / s
    if ef.startswith("paneo"):
        recorrido = (p - 0.5) * 2 * I
        if ef == "paneo_der":
            dx = recorrido * cw
        elif ef == "paneo_izq":
            dx = -recorrido * cw
        elif ef == "paneo_abajo":
            dy = recorrido * ch
        elif ef == "paneo_arriba":
            dy = -recorrido * ch
    cx = float(e.get("fx", 0.5)) * iw + dx
    cy = float(e.get("fy", 0.45)) * ih + dy
    x0 = min(max(cx - cw / 2, 0), max(0, iw - cw))
    y0 = min(max(cy - ch / 2, 0), max(0, ih - ch))
    return x0, y0, cw, ch, s


def ajustar_marco(r, iw, ih, w, h):
    """Marco normalizado [x, y, ancho, alto] con la proporción exacta del área (w/h) y dentro de la foto."""
    A = w / h
    x, y, rw, rh = [float(v) for v in r]
    cx, cy = (x + rw / 2) * iw, (y + rh / 2) * ih
    cw = max(8.0, rw * iw)
    ch = cw / A
    if ch > ih:
        ch = float(ih)
        cw = ch * A
    if cw > iw:
        cw = float(iw)
        ch = cw / A
    x0 = min(max(cx - cw / 2, 0), iw - cw)
    y0 = min(max(cy - ch / 2, 0), ih - ch)
    return [x0 / iw, y0 / ih, cw / iw, ch / ih]


def marcos(e, iw, ih, w, h):
    """Marcos de inicio y final (normalizados) y tipo de suavizado. Convierte el formato anterior (fx, fy, efecto)."""
    if e.get("inicio") and e.get("fin"):
        return ajustar_marco(e["inicio"], iw, ih, w, h), ajustar_marco(e["fin"], iw, ih, w, h), e.get("suavizado", "suave")
    res = []
    for fase in (0.0, 1.0):
        x0, y0, cw, ch, _ = ventana(iw, ih, w, h, fase, e)
        res.append([x0 / iw, y0 / ih, cw / iw, ch / ih])
    return res[0], res[1], "lineal"


def curva(fase, suavizado):
    p = min(1.0, max(0.0, fase))
    if suavizado == "suave":
        return 0.5 - 0.5 * math.cos(math.pi * p)
    return p


def preparar_imagen(img, w, h, e):
    """Reduce fotos muy grandes al tamaño necesario (evita escalones y acelera el render)."""
    ih, iw = img.shape[:2]
    a, b, _ = marcos(e, iw, ih, w, h)
    ancho_min = max(1e-3, min(a[2], b[2])) * iw
    r = (w / ancho_min) * 1.05
    if r < 1:
        img = cv2.resize(img, (max(1, int(iw * r)), max(1, int(ih * r))), interpolation=cv2.INTER_AREA)
    return img


def encuadrar(img, w, h, fase, e):
    """Cuadro (w x h) de la foto: interpola entre el marco de inicio y el final. fase 0..1."""
    ih, iw = img.shape[:2]
    a, b, suav = marcos(e, iw, ih, w, h)
    p = curva(fase, suav)
    x = (a[0] + (b[0] - a[0]) * p) * iw
    y = (a[1] + (b[1] - a[1]) * p) * ih
    cw = (a[2] + (b[2] - a[2]) * p) * iw
    s = w / cw
    M = np.float32([[s, 0, -x * s], [0, s, -y * s]])
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def sanear_grupos(grupos, max_palabra=1.6, max_cola=0.8):
    """Evita subtítulos 'atorados': ningún subtítulo dura más allá de su última palabra (más una cola corta)
    ni se encima con el siguiente. Pasa cuando whisper le da a una palabra un final muy tardío (ruido o música
    después de la frase) o cuando un tiempo se escribió mal a mano."""
    gs = sorted([dict(g, palabras=[dict(p) for p in g["palabras"]]) for g in grupos if g.get("palabras")],
                key=lambda g: g["i"])
    for k, g in enumerate(gs):
        for p in g["palabras"]:
            if p["f"] - p["i"] > max_palabra:
                p["f"] = round(p["i"] + max_palabra, 3)
        fin = min(g["f"], g["palabras"][-1]["f"] + max_cola) if g["palabras"][-1]["f"] > g["i"] else g["f"]
        if k + 1 < len(gs):
            fin = min(fin, gs[k + 1]["i"])
        g["f"] = round(max(fin, g["i"] + 0.3), 3)
    return gs


def grupos_subtitulos(trabajo, palabras, cfg):
    """Usa los subtítulos editados a mano (subtitulos.json) si existen."""
    p = Path(trabajo) / "subtitulos.json"
    if p.exists():
        return sanear_grupos(cargar_json(p))
    palabras = limpiar_transcripcion(palabras)
    if cfg.get("modo") == "frases":
        return sanear_grupos(estilo_motivacional().agrupar_frases(palabras, cfg))
    return sanear_grupos(agrupar_subtitulos(palabras, cfg))


def recorte_cubrir(frame, caja, aspect, cx=None, cy=None, zoom=1.0):
    """Recorta de 'caja' (x,y,w,h) una ventana con la proporción 'aspect' (ancho/alto) centrada en (cx, cy)."""
    x, y, w, h = caja
    if w / h > aspect:
        ch, cw = h, h * aspect
    else:
        cw, ch = w, w / aspect
    cw, ch = cw / max(1.0, zoom), ch / max(1.0, zoom)
    cx = x + w / 2 if cx is None else cx
    cy = y + h / 2 if cy is None else cy
    x0 = min(max(cx - cw / 2, x), x + w - cw)
    y0 = min(max(cy - ch * 0.4, y), y + h - ch)
    return frame[int(y0):int(y0 + ch), int(x0):int(x0 + cw)]


def filtro_audio(pl, info, hay_musica, cortes):
    """Voz (sin pausas, nivelada) + música en loop que baja sola cuando la persona habla (sidechain)."""
    a = pl.get("audio", {})
    if not info.get("audio"):
        return None, None
    voz = "[1:a:0]"
    if cortes:
        # se quitan las pausas armando los pedazos que se quedan (atrim + concat). Antes era una sola expresión
        # 'aselect' que con muchos cortes ffmpeg no podía leer ("Cannot allocate memory")
        dur = info.get("duracion") or 1e9
        quedan, t0 = [], 0.0
        for c in sorted(cortes, key=lambda c: c["inicio"]):
            if c["inicio"] - t0 > 0.001:
                quedan.append((t0, c["inicio"]))
            t0 = max(t0, c["fin"])
        if dur - t0 > 0.001:
            quedan.append((t0, dur))
        if not quedan:
            quedan = [(0.0, dur)]
        n = len(quedan)
        partes = "[1:a:0]asplit=" + str(n) + "".join(f"[s{k}]" for k in range(n)) + ";" if n > 1 else ""
        for k, (a0, b0) in enumerate(quedan):
            ent = f"[s{k}]" if n > 1 else "[1:a:0]"
            fin_txt = f":end={b0:.4f}" if b0 < 1e8 else ""
            partes += f"{ent}atrim=start={a0:.4f}{fin_txt},asetpts=PTS-STARTPTS[q{k}];"
        partes += "".join(f"[q{k}]" for k in range(n)) + f"concat=n={n}:v=0:a=1[vcort];"
        voz = partes + "[vcort]"
    voz += ("dynaudnorm=f=150:g=15:p=0.9:m=8," if a.get("nivelar_voz", True) else "") + "aresample=48000"
    if not hay_musica:
        return voz + "[a]", "[a]"
    f = (f"{voz},asplit=2[voz][sc];"
         f"[2:a]aresample=48000,volume={a.get('volumen_musica', 0.5)}[mus];"
         f"[mus][sc]sidechaincompress=threshold={a.get('duck_umbral', 0.02)}:ratio={a.get('duck_ratio', 6)}"
         f":attack={a.get('duck_ataque_ms', 20)}:release={a.get('duck_liberacion_ms', 400)}[mduck];"
         f"[voz][mduck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
    return f, "[a]"


def cuadro_lectura(frame, pv, t, w, h, cfg):
    """Parte del documento alrededor de la palabra que se está leyendo (unas 5 líneas, legible)."""
    cam = pv["camino"]
    ts = [c[0] for c in cam]
    cx = float(np.interp(t, ts, [c[1] for c in cam]))
    cy = float(np.interp(t, ts, [c[2] for c in cam]))
    x, y, cw_max, ch_max = pv["caja"]
    alto = cfg.get("lineas_visibles", 5.0) * pv["alto_linea"]
    ancho = alto * w / h
    if ancho > cw_max:
        ancho = float(cw_max)
        alto = ancho * h / w
    x0 = cx - ancho * cfg.get("posicion_x", 0.45)
    y0 = cy - alto * cfg.get("posicion_y", 0.42)
    x0 = min(max(x0, x), x + cw_max - ancho)
    y0 = min(max(y0, y), max(y, y + ch_max - alto))
    s = w / ancho
    M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
    return cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def pantalla_de(pv, cfg_p):
    """Cómo se muestra un tramo de la proyección: 'completa', 'dividida' o 'nacho' (no se muestra).
    Se puede forzar por tramo con el campo 'pantalla' (lo cambia la interfaz)."""
    if pv.get("pantalla") in ("completa", "dividida", "nacho"):
        return pv["pantalla"]
    if pv["tipo"] == "video":
        return "completa"
    if pv["tipo"] == "imagen" and cfg_p.get("imagen_estilo") == "completa" and not pv.get("documento") \
            and not pv.get("vertical"):
        return "completa"
    return "dividida"


class VistaPrevia:
    """Video chico SIN subtítulos (vista_previa.mp4) para que la interfaz muestre los subtítulos editados
    encima, en vivo, sin tener que renderizar. También guarda qué diseño hay en cada momento."""

    def __init__(self, trabajo, W, H, fps):
        self.dir = Path(trabajo)
        self.tmp = self.dir / "vista_previa_tmp.mp4"
        self.w, self.h = W // 2, H // 2
        self.fps = fps
        self.linea = []
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.w}x{self.h}",
               "-r", f"{fps}", "-i", "-",
               *(["-c:v", "h264_nvenc", "-preset", "p1", "-cq", "32"] if SOLO_GPU else
                 ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "30", "-threads", "2"]),
               "-pix_fmt", "yuv420p", str(self.tmp)]
        try:
            self.ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        except Exception:
            self.ff = None

    def escribir(self, lienzo, te, lay):
        if not self.linea or self.linea[-1][1] != lay:
            self.linea.append([round(te, 3), lay])
        if self.ff:
            try:
                self.ff.stdin.write(cv2.resize(lienzo, (self.w, self.h), interpolation=cv2.INTER_AREA).tobytes())
            except (BrokenPipeError, OSError):
                self.ff = None

    def cerrar(self, final):
        guardar_json(self.dir / "vista_previa.json", {"diseno": self.linea, "fps": self.fps})
        if not self.ff:
            return
        self.ff.stdin.close()
        if self.ff.wait() != 0:
            return
        # el audio se toma del video final
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.tmp), "-i", str(final),
                            "-map", "0:v", "-map", "1:a?", "-c", "copy", "-movflags", "+faststart",
                            str(self.dir / "vista_previa.mp4")])
        if r.returncode == 0:
            self.tmp.unlink(missing_ok=True)


def nvenc_disponible():
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
                        "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True, text=True)
    return r.returncode == 0


def cmd_render(args):
    if not shutil.which("ffmpeg"):
        raise SystemExit("No encontré ffmpeg en el PATH")
    trabajo = Path(args.trabajo.strip())
    plan = cargar_json(trabajo / "plan.json")
    pl = cargar_json(plan["plantilla"])
    info = plan["info"]
    if plan.get("estilo") == "solo_subtitulos" or pl.get("estilo") == "solo_subtitulos":
        return estilo_solo_subtitulos().render(args, trabajo, plan, pl)
    if plan.get("estilo") == "motivacional" or pl.get("estilo") == "motivacional":
        return estilo_motivacional().render(args, trabajo, plan, pl)
    escalar_cajas(pl, info["ancho"], info["alto"])
    W, H = pl["salida"]["ancho"], pl["salida"]["alto"]
    fps = info["fps"]
    salida = Path(args.salida) if args.salida else trabajo / (Path(plan["video"]).stem + "_vertical.mp4")

    titulo = capa_titulo(pl, plan.get("titulo", ""), W, H)
    palabras = cargar_json(trabajo / "transcripcion.json") if (trabajo / "transcripcion.json").exists() else []
    subs = Subtitulos(grupos_subtitulos(trabajo, palabras, pl["subtitulos"]), pl["subtitulos"])

    L = pl["layouts"]
    franja = None
    if L["dividida"].get("franja", {}).get("fondo"):
        fx, fy, fw, fh = L["dividida"]["franja"]["area"]
        franja = ajustar(leer_img(ruta(L["dividida"]["franja"]["fondo"])), fw, fh)

    cache_img = {}
    cfg_m = pl["momentos"]

    def imagen_momento(mom, k, w, h, fase):
        """Imagen k del momento, encuadrada y con su movimiento."""
        p = mom["imagenes"][k]
        idx_global = momentos.index(mom) * 3 + k
        clave = (p, w, h, id(mom), k)
        if clave not in cache_img:
            im = leer_img(p)
            if im is None:
                log(f"No pude leer la imagen {p}")
                im = np.zeros((h, w, 3), np.uint8)
            e = encuadre_de(mom, k, idx_global, cfg_m, im)
            cache_img[clave] = (preparar_imagen(im, w, h, e), e)
        im, e = cache_img[clave]
        return encuadrar(im, w, h, fase, e)

    cfg_p = pl.get("proyeccion", {})
    caja_p = pl["fuente_16x9"]["persona_dividida"]
    tomas = plan["tomas"]
    proy = plan.get("proyeccion", [])
    momentos = [m for m in plan.get("momentos", []) if m.get("imagenes") and not m.get("oculto")]
    cortes = sorted(plan.get("cortes", []), key=lambda c: c["inicio"])

    musica = ruta(pl.get("audio", {}).get("musica")) if not args.sin_musica else None
    hay_musica = bool(musica and Path(musica).exists())
    f_audio, map_audio = filtro_audio(pl, info, hay_musica, cortes)

    venc = codificador(args.gpu, pl["salida"]["crf"])
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps}", "-i", "-",
           "-i", plan["video"]]
    if hay_musica:
        cmd += ["-stream_loop", "-1", "-i", str(musica)]
    cmd += ["-map", "0:v"]
    if f_audio:
        # audio_kbps es el máximo: el codificador AAC se pasa ~2%, así que se pide un poco menos
        kbps = int(pl["salida"].get("audio_kbps", 125))
        objetivo = max(32, int(kbps * 0.96))
        cmd += ["-filter_complex", f_audio, "-map", map_audio, "-c:a", "aac", "-b:a", f"{objetivo}k"]
    cmd += [*venc, "-pix_fmt", "yuv420p", "-shortest", "-movflags", "+faststart", str(salida)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    vista = VistaPrevia(trabajo, W, H, fps)

    cap = cv2.VideoCapture(plan["video"])
    # transición (cortinilla) al entrar y salir de la pantalla dividida, si la plantilla la tiene
    cortinilla, ventanas_cort = [], []
    cfg_tr = pl.get("transicion_dividida") or {}
    if cfg_tr.get("carpeta"):
        carpeta = ruta(cfg_tr["carpeta"])
        archivos = sorted(p for p in Path(carpeta).glob("*") if p.suffix.lower() in EXT_IMG) if carpeta.exists() else []
        cortinilla = [cv2.resize(leer_img(p), (W, H)) for p in archivos]
        if cortinilla:
            def es_dividida(tt):
                pv_ = dentro(tt, proy)
                if pv_ and pv_["tipo"] == "lectura" and any(a <= tt < b for a, b in pv_.get("ocultar", [])):
                    pv_ = None if not cfg_p.get("scroll_como_documento") else pv_
                if pv_:
                    return pantalla_de(pv_, cfg_p) == "dividida"
                m_ = dentro(tt, momentos)
                return bool(m_ and m_.get("estilo") != "completa")
            dur_c = len(cortinilla) / cfg_tr.get("fps", 25)
            antes = es_dividida(0.0)
            paso_t = 1.0 / fps
            tt = paso_t
            while tt < info["duracion"]:
                ahora = es_dividida(tt)
                if ahora != antes:
                    te_b = a_editado(tt, cortes)
                    ini_c = te_b - dur_c * cfg_tr.get("antes", 0.5)
                    if not ventanas_cort or ini_c > ventanas_cort[-1][0] + dur_c:
                        ventanas_cort.append((ini_c, ini_c + dur_c))
                    antes = ahora
                tt += paso_t
            log(f"{len(ventanas_cort)} transiciones a pantalla dividida")

    idx, ti, ci, n_out = 0, 0, 0, 0
    total = info["cuadros"] or 1
    lienzo = np.zeros((H, W, 3), np.uint8)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        idx += 1
        if idx % max(1, int(fps)) == 0:
            Progreso.paso(idx / total, "Armando el video")
        # pausas recortadas
        while ci < len(cortes) and t >= cortes[ci]["fin"] - 1e-6:
            ci += 1
        if ci < len(cortes) and cortes[ci]["inicio"] - 1e-6 <= t < cortes[ci]["fin"] - 1e-6:
            continue
        te = n_out / fps  # tiempo en el video editado
        n_out += 1

        while ti + 1 < len(tomas) and t >= tomas[ti]["fin"]:
            ti += 1
        toma = tomas[ti]
        pv = dentro(t, proy)
        if pv and pv["tipo"] == "lectura" and any(a <= t < b for a, b in pv.get("ocultar", [])):
            # la página se está desplazando: se ve a Nacho (o la página completa, según la plantilla)
            pv = dict(pv, tipo="imagen", documento=True) if cfg_p.get("scroll_como_documento") else None
        if pv and pantalla_de(pv, cfg_p) == "nacho":
            pv = None
        modo_pv = pantalla_de(pv, cfg_p) if pv else None
        mom = dentro(t, momentos) if not pv else None
        lienzo[:] = 0

        if pv and modo_pv == "completa" and (pv["tipo"] == "video" or pv.get("vertical") or not cfg_p.get("imagen_izquierda")):
            # video de la proyección a pantalla completa, centrado en quien habla (la cara más grande)
            # o en lo que se mueve; con un poco de acercamiento si hay una persona
            cx = cy = None
            if pv.get("cara") and len(pv["cara"]) > 1:
                cx, cy, _ = interp_cara(pv["cara"], t)
            z = cfg_p.get("zoom_video", 1.0) if pv.get("con_cara", 0) >= 0.3 else 1.0
            rec = recorte_cubrir(frame, pv["caja"], W / H, cx, cy, z)
            lienzo[:] = enfocar(cv2.resize(rec, (W, H), interpolation=cv2.INTER_CUBIC), 0.3)
            lay = "persona"
        elif pv and modo_pv == "completa":
            # imagen fija de la pantalla (p. ej. un tweet) a pantalla completa, desde la izquierda, con zoom lento
            x, y, w, h = pv["caja"]
            fase = (t - pv["inicio"]) / max(0.1, pv["fin"] - pv["inicio"])
            z = 1.0 + 0.06 * fase
            ch2 = min(h, w / (W / H)) / z
            cw2 = ch2 * W / H
            x0 = x + min(w - cw2, w * 0.02)
            y0 = y + (h - ch2) * 0.4
            rec = frame[int(y0):int(y0 + ch2), int(x0):int(x0 + cw2)]
            lienzo[:] = enfocar(cv2.resize(rec, (W, H), interpolation=cv2.INTER_CUBIC), 0.3)
            lay = "persona"
        elif mom and mom.get("estilo") == "completa":
            imgs = mom["imagenes"]
            dur = (mom["fin"] - mom["inicio"]) / len(imgs)
            k = min(int((t - mom["inicio"]) // dur), len(imgs) - 1)
            fase = min(1.0, (t - mom["inicio"] - k * dur) / dur)
            lienzo[:] = imagen_momento(mom, k, W, H, fase)
            lay = "persona"
        else:
            lay = "dividida" if (mom or pv) else "persona"
            cfg = L[lay]
            cara = interp_cara(toma["cara"], t)
            region = caja_p if toma["layout"] == "dividida" else [0, 0, frame.shape[1], frame.shape[0]]
            ax, ay, aw, ah = cfg["persona"]["area"]
            if cfg["persona"].get("ajuste") == "contener" and region[2] / region[3] < (aw / ah) * 0.8:
                pers = persona_contenida(frame, region, cara, aw, ah, cfg["persona"])
            else:
                x0, y0, cw, ch = recorte_persona(region, cara, (aw, ah), cfg["persona"])
                pers = cv2.resize(frame[y0:y0 + ch, x0:x0 + cw], (aw, ah), interpolation=cv2.INTER_LANCZOS4)
            lienzo[ay:ay + ah, ax:ax + aw] = enfocar(pers, cfg["persona"].get("nitidez", 0))

            if lay == "dividida":
                if franja is not None:
                    fx, fy, fw, fh = cfg["franja"]["area"]
                    lienzo[fy:fy + fh, fx:fx + fw] = franja
                ix, iy, iw, ih = cfg["inferior"]["area"]
                if mom:
                    imgs = mom["imagenes"]
                    dur = (mom["fin"] - mom["inicio"]) / len(imgs)
                    k = min(int((t - mom["inicio"]) // dur), len(imgs) - 1)
                    fase = min(1.0, (t - mom["inicio"] - k * dur) / dur)
                    lienzo[iy:iy + ih, ix:ix + iw] = imagen_momento(mom, k, iw, ih, fase)
                elif pv["tipo"] == "lectura":
                    lienzo[iy:iy + ih, ix:ix + iw] = cuadro_lectura(frame, pv, t, iw, ih, pl.get("lectura", {}))
                else:
                    # imagen fija de la proyección (p. ej. un mapa): acercamiento con paneo horizontal
                    x, y, w, h = pv["caja"]
                    asp = iw / ih
                    fase = (t - pv["inicio"]) / max(0.1, pv["fin"] - pv["inicio"])
                    if pv["tipo"] == "video" or pv.get("vertical"):
                        # video o imagen de celular: a todo lo ancho, la parte central
                        cw2 = float(w)
                        ch2 = min(float(h), cw2 / asp)
                        cw2 = ch2 * asp
                        x0 = x + (w - cw2) / 2
                        y0 = y + (h - ch2) * 0.45
                    elif pv.get("documento"):
                        # documento: a todo lo ancho, bajando despacio desde arriba
                        cw2 = float(w)
                        ch2 = cw2 / asp
                        x0 = x
                        y0 = y + max(0.0, h - ch2) * (0.05 + 0.5 * fase)
                    else:
                        z = cfg_p.get("zoom_imagen", 1.5)
                        cw2 = min(w, h * asp) / z
                        ch2 = cw2 / asp
                        x0 = x + (w - cw2) * (0.2 + 0.6 * fase)
                        y0 = y + (h - ch2) * 0.45
                    rec = frame[int(y0):int(y0 + ch2), int(x0):int(x0 + cw2)]
                    lienzo[iy:iy + ih, ix:ix + iw] = cv2.resize(rec, (iw, ih), interpolation=cv2.INTER_CUBIC)

        for a_c, b_c in ventanas_cort:
            if a_c <= te < b_c:
                k = min(len(cortinilla) - 1, int((te - a_c) / (b_c - a_c) * len(cortinilla)))
                lienzo[:] = cortinilla[k]
                break
        pegar(lienzo, titulo)
        vista.escribir(lienzo, te, lay)

        act = subs.activo(t)
        if act:
            sx, sy, sw, sh = L[lay]["subtitulos"]["area"]
            pegar(lienzo, subs.imagen(act[0], act[1], sw, sh), sx, sy)

        try:
            ff.stdin.write(lienzo.tobytes())
        except (BrokenPipeError, OSError):  # ffmpeg se cerró (el error real sale arriba en el registro)
            break
    cap.release()
    try:
        ff.stdin.close()
    except OSError:
        pass
    if ff.wait() != 0:
        raise SystemExit("ffmpeg falló al codificar")
    vista.cerrar(salida)
    Progreso.paso(1.0, "Listo")
    log(f"Listo: {salida}")
    return salida


# ----------------------------------------------------------------------------
# Calibrar
# ----------------------------------------------------------------------------
def cmd_calibrar(args):
    pl = cargar_json(ruta(args.plantilla))
    video = args.video.strip()
    fps, n, w, h = info_video(video)
    escalar_cajas(pl, w, h)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.t * fps))
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("No pude leer ese cuadro")
    lay = clasificar_cuadro(frame, pl["deteccion"])
    colores = {"persona_dividida": (0, 255, 0), "proyeccion": (0, 160, 255)}
    for k, c in colores.items():
        x, y, bw, bh = [int(v) for v in pl["fuente_16x9"][k]]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), c, 3)
        cv2.putText(frame, k, (x + 8, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
    for k in ("franja_superior", "columna_separacion"):
        x, y, bw, bh = pl["deteccion"][k]
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (255, 0, 255), 2)
    cv2.putText(frame, f"detectado: {lay}", (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    out = Path(video).with_name(Path(video).stem + f"_calibrar_{args.t:.0f}s.jpg")
    guardar_img(out, frame)
    log(f"Detectado '{lay}'. Imagen: {out}")
    return out, lay


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Editor vertical automático")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def opciones_analisis(p):
        p.add_argument("video")
        p.add_argument("--titulo", required=True)
        p.add_argument("--plantilla", default="plantillas/chapucero_vertical.json")
        p.add_argument("--trabajo", help="Carpeta de trabajo (por defecto, junto al video)")
        p.add_argument("--imagenes", help="Carpeta con imágenes propias, en orden (reemplaza la búsqueda web)")
        p.add_argument("--sin-web", action="store_true", help="No buscar imágenes en internet")
        p.add_argument("--sin-ia", action="store_true", help="No usar Ollama (sin análisis del tema)")
        p.add_argument("--sin-cortes", action="store_true", help="No recortar pausas")
        p.add_argument("--transcripcion", help="JSON de palabras ya transcritas [{p,i,f}]")
        p.add_argument("--sin-subtitulos", action="store_true")
        p.add_argument("--whisper-modelo", default="large-v3")
        p.add_argument("--dispositivo", default="auto", choices=["auto", "cuda", "cpu"])
        p.add_argument("--modelo-ollama", default="auto", help="Modelo de texto (auto = el mejor instalado)")
        p.add_argument("--modelo-vision", default="auto", help="Modelo de visión para verificar imágenes")
        p.add_argument("--sin-verificar", action="store_true", help="No verificar las imágenes con el modelo de visión")
        p.add_argument("--url-ollama", default="http://localhost:11434")
        p.add_argument("--solo-gpu", action="store_true", help="Todo lo pesado en la tarjeta gráfica; nunca cae al procesador")

    def opciones_render(p):
        p.add_argument("--salida")
        p.add_argument("--gpu", action="store_true", help="Codificar con NVENC (NVIDIA); si falla usa CPU")
        p.add_argument("--sin-musica", action="store_true")

    pa = sub.add_parser("analizar")
    opciones_analisis(pa)

    pr = sub.add_parser("render")
    pr.add_argument("trabajo")
    pr.add_argument("--solo-gpu", action="store_true")
    opciones_render(pr)

    pt = sub.add_parser("todo")
    opciones_analisis(pt)
    opciones_render(pt)

    pv = sub.add_parser("convertir")
    pv.add_argument("video")

    pc = sub.add_parser("calibrar")
    pc.add_argument("video")
    pc.add_argument("--t", type=float, default=10)
    pc.add_argument("--plantilla", default="plantillas/chapucero_vertical.json")
    for p_ in (pv, pc):
        p_.add_argument("--solo-gpu", action="store_true")

    args = ap.parse_args()
    cuidar_procesador()
    if args.cmd == "analizar":
        tr = cmd_analizar(args)
        print(json.dumps({"ok": True, "trabajo": str(tr), "plan": str(tr / "plan.json")}, ensure_ascii=False))
    elif args.cmd == "render":
        out = cmd_render(args)
        print(json.dumps({"ok": True, "salida": str(out)}, ensure_ascii=False))
    elif args.cmd == "todo":
        Progreso.rango(0.0, 0.6)
        tr = cmd_analizar(args)
        args.trabajo = str(tr)
        Progreso.rango(0.6, 1.0)
        out = cmd_render(args)
        print(json.dumps({"ok": True, "trabajo": str(tr), "salida": str(out)}, ensure_ascii=False))
    elif args.cmd == "convertir":
        out = cmd_convertir(args)
        print(json.dumps({"ok": True, "video": str(out)}, ensure_ascii=False))
    elif args.cmd == "calibrar":
        out, lay = cmd_calibrar(args)
        print(json.dumps({"ok": True, "imagen": str(out), "layout": lay}, ensure_ascii=False))


if __name__ == "__main__":
    main()
