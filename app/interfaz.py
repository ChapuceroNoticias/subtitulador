#!/usr/bin/env python3
"""
Interfaz del editor vertical.

    python interfaz.py

Se abre en el navegador (http://127.0.0.1:7860). Desde ahí:
  - Procesas un video nuevo o cargas una carpeta _trabajo ya hecha.
  - Pestaña Subtítulos: tabla con todos los subtítulos; editas el texto o los tiempos y guardas.
  - Pestaña Imágenes: por cada momento eliges estilo, tiempos, imagen, encuadre (clic sobre la foto),
    y el movimiento con dos marcos (inicio y final) y vista previa animada; buscas o subes imágenes.
  - Pestaña Título y cortes: cambias el título o quitas un tramo del video.
  - Renderizas de nuevo con un botón.
"""
import inspect
import json
import os
import shutil
import string
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote

import cv2
try:
    import gradio as gr
except ImportError:  # el subtitulador no usa gradio
    gr = None
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import editor as ed  # noqa: E402

# compatibilidad Gradio 5 y 6
_TEMA_EN_BLOCKS = bool(gr) and "theme" in inspect.signature(gr.Blocks.__init__).parameters

URL_OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
CARPETA_VIDEOS = Path(os.environ.get("CARPETA_VIDEOS", r"C:\Users\USER\Desktop\VideosEdit"))
EXT_VIDEO = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mts", ".flv", ".wmv"]


def buscar_video(nombre):
    """Busca el video en la carpeta fija. Sin extensión, primero .mp4 y luego los demás formatos."""
    nombre = limpiar_ruta(nombre)
    if not nombre:
        return None
    p = Path(nombre)
    if p.is_absolute() and p.exists():
        return p
    base = CARPETA_VIDEOS / nombre
    if base.suffix.lower() in EXT_VIDEO and base.exists():
        return base
    for ext in EXT_VIDEO:
        candidato = base.with_name(base.name + ext)
        if candidato.exists():
            return candidato
    # sin importar mayúsculas en la extensión
    if CARPETA_VIDEOS.exists():
        stem = Path(nombre).stem.lower() if Path(nombre).suffix.lower() in EXT_VIDEO else nombre.lower()
        for f in CARPETA_VIDEOS.iterdir():
            if f.suffix.lower() in EXT_VIDEO and f.stem.lower() == stem:
                return f
    return None
_IA = None

EFECTOS = {
    "Acercar": "zoom_in",
    "Alejar": "zoom_out",
    "Paneo a la derecha": "paneo_der",
    "Paneo a la izquierda": "paneo_izq",
    "Paneo hacia arriba": "paneo_arriba",
    "Paneo hacia abajo": "paneo_abajo",
    "Sin movimiento": "ninguno",
}
EFECTO_NOMBRE = {v: k for k, v in EFECTOS.items()}


def ia():
    global _IA
    if _IA is None:
        _IA = ed.IA(URL_OLLAMA)
    return _IA


# ----------------------------------------------------------------------------
# Archivos del trabajo
# ----------------------------------------------------------------------------
def limpiar_ruta(txt):
    return (txt or "").strip().strip('"').strip()


def cargar(trabajo):
    t = Path(trabajo)
    plan = ed.cargar_json(t / "plan.json")
    palabras = ed.cargar_json(t / "transcripcion.json") if (t / "transcripcion.json").exists() else []
    return plan, palabras


ESTILOS = {
    "Clips Chapuceros": "plantillas/chapucero_vertical.json",
    "Tiktoks Motivacionales": "plantillas/tiktok_motivacional.json",
    "Clips Fuera": "plantillas/clips_fuera.json",
    "Solo subtítulos (Chapucero)": "plantillas/solo_subtitulos_chapucero.json",
    "Solo subtítulos (Motivacional)": "plantillas/solo_subtitulos_motivacional.json",
}


def plantilla(plan):
    return ed.cargar_json(plan["plantilla"])


def grupos_actuales(trabajo, plan, palabras):
    return ed.grupos_subtitulos(trabajo, palabras, plantilla(plan)["subtitulos"])


def respaldar(trabajo):
    t = Path(trabajo)
    h = t / ".historial"
    h.mkdir(exist_ok=True)
    marca = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
    shutil.copy2(t / "plan.json", h / f"{marca}_plan.json")
    sub = t / "subtitulos.json"
    if sub.exists():
        shutil.copy2(sub, h / f"{marca}_subtitulos.json")
    else:
        (h / f"{marca}_subtitulos.json").write_text("null", encoding="utf-8")


def deshacer(trabajo):
    h = Path(trabajo) / ".historial"
    planes = sorted(h.glob("*_plan.json")) if h.exists() else []
    if not planes:
        return False
    ultimo = planes[-1]
    marca = ultimo.name[: -len("_plan.json")]
    shutil.copy2(ultimo, Path(trabajo) / "plan.json")
    sub = h / f"{marca}_subtitulos.json"
    destino = Path(trabajo) / "subtitulos.json"
    if sub.exists():
        if sub.read_text(encoding="utf-8").strip() == "null":
            if destino.exists():
                destino.unlink()
        else:
            shutil.copy2(sub, destino)
        sub.unlink()
    ultimo.unlink()
    return True


PANTALLAS = {"completa": "Pantalla completa", "dividida": "Dividida (Nacho arriba)", "nacho": "Solo Nacho"}
TIPOS = {"video": "Video", "imagen": "Imagen", "lectura": "Documento (lectura)"}


def tabla_pantalla(trabajo):
    """Tramos en que se ve lo que Nacho muestra en pantalla, con cómo se ven (editable)."""
    if not trabajo:
        return []
    plan, _ = cargar(trabajo)
    pl = plantilla(plan)
    filas = []
    for p in plan.get("proyeccion", []):
        tipo = TIPOS.get(p["tipo"], p["tipo"])
        if p.get("documento"):
            tipo = "Documento"
        elif p.get("vertical") or (p["tipo"] == "video" and p.get("caja") and p["caja"][3] > p["caja"][2] * 1.2):
            tipo = "Video de celular"
        filas.append([mmss(ed_t(p["inicio"], plan)), mmss(ed_t(p["fin"], plan)), tipo,
                      PANTALLAS[ed.pantalla_de(p, pl.get("proyeccion", {}))]])
    return filas


def guardar_pantalla(trabajo, tabla):
    plan, _ = cargar(trabajo)
    proy = plan.get("proyeccion", [])
    inverso = {v: k for k, v in PANTALLAS.items()}
    for p, fila in zip(proy, filas_de(tabla)):
        if len(fila) < 4:
            continue
        txt = str(fila[3]).strip().lower()
        modo = inverso.get(str(fila[3]).strip()) or next((k for k in ("completa", "dividida", "nacho") if k in txt), None)
        if modo:
            p["pantalla"] = modo
        ini, fin = seg(fila[0]), seg(fila[1])
        if ini is not None and mmss(ini) != mmss(ed_t(p["inicio"], plan)):
            p["inicio"] = round(orig_t(ini, plan), 3)
        if fin is not None and mmss(fin) != mmss(ed_t(p["fin"], plan)):
            p["fin"] = round(orig_t(fin, plan), 3)
    proy.sort(key=lambda x: x["inicio"])
    respaldar(trabajo)
    guardar_plan(trabajo, plan)
    return len(proy)


# ----------------------------------------------------------------------------
# Línea de tiempo de pantalla (todos los estilos menos Solo subtítulos)
# ----------------------------------------------------------------------------
NOMBRES_OP = {"completa": "Pantalla completa", "dividida": "Dividida", "nacho": "Solo Nacho",
              "cerca": "Nacho cerca", "lejos": "Nacho lejos", "imagen": "Imagen", "quitar": "Quitar"}


def estilo_de(plan):
    pl = plantilla(plan)
    return plan.get("estilo") or pl.get("estilo") or "chapucero"


def tramos_pantalla(t):
    """Todo el video en tramos, cada uno con cómo se ve y qué opciones tiene. Tiempos del video editado."""
    plan, _ = cargar(t)
    pl = plantilla(plan)
    est = estilo_de(plan)
    dur = plan["info"]["duracion"]
    fin_ed = ed_t(dur, plan)
    out = []

    def add(k, a, b, tipo, pantalla, opciones):
        a_e, b_e = ed_t(max(0.0, a), plan), ed_t(min(dur, b), plan)
        if b_e - a_e < 0.2:
            return
        out.append({"k": k, "inicio": mmss(a_e), "fin": mmss(b_e), "ini_s": round(a_e, 3), "fin_s": round(b_e, 3),
                    "tipo": tipo, "pantalla": pantalla, "opciones": opciones})

    if est == "solo_subtitulos":
        return {"tramos": [], "duracion": fin_ed, "mensaje": "En «Solo subtítulos» el video no se reacomoda."}

    if est == "motivacional":
        for i, e in enumerate(plan.get("encuadres", [])):
            add(f"e{i}", e["inicio"], e["fin"], "Nacho", e.get("zoom", "cerca"), ["cerca", "lejos"])
        for i, m in enumerate(plan.get("momentos", [])):
            if m.get("imagenes"):
                add(f"m{i}", m["inicio"], m["fin"], "Imagen: " + (m.get("busqueda") or "")[:40],
                    "quitar" if m.get("oculto") else "imagen", ["imagen", "quitar"])
    else:
        cfg_p = pl.get("proyeccion", {})
        proy = plan.get("proyeccion", [])
        for i, p in enumerate(proy):
            tipo = TIPOS.get(p["tipo"], p["tipo"])
            if p.get("documento"):
                tipo = "Documento"
            elif p.get("vertical") or (p["tipo"] == "video" and p.get("caja") and p["caja"][3] > p["caja"][2] * 1.2):
                tipo = "Video de celular"
            if p.get("habla_otro"):
                tipo += " (habla la persona del video)"
            add(f"p{i}", p["inicio"], p["fin"], tipo, ed.pantalla_de(p, cfg_p), ["completa", "dividida", "nacho"])
        # ratos de pantalla partida donde no se usa lo que Nacho proyecta (sale solo Nacho)
        for i, sg in enumerate(plan.get("tomas", [])):
            if sg.get("layout") != "dividida":
                add(f"n{i}", sg["inicio"], sg["fin"], "Nacho (sin pantalla)", "nacho", [])  # solo se ve en la línea
                continue
            libres = [(sg["inicio"], sg["fin"])]
            for p in proy:
                nuevos = []
                for a, b in libres:
                    if p["fin"] <= a or p["inicio"] >= b:
                        nuevos.append((a, b))
                        continue
                    if p["inicio"] > a:
                        nuevos.append((a, p["inicio"]))
                    if p["fin"] < b:
                        nuevos.append((p["fin"], b))
                libres = nuevos
            for a, b in libres:
                if b - a >= 1.0:
                    add(f"g{a:.3f}_{b:.3f}", a, b, "Lo que proyecta (no se está usando)", "nacho",
                        ["completa", "dividida", "nacho"])
        for i, m in enumerate(plan.get("momentos", [])):
            if m.get("imagenes"):
                add(f"m{i}", m["inicio"], m["fin"], "Imagen de internet: " + (m.get("busqueda") or "")[:40],
                    "quitar" if m.get("oculto") else (m.get("estilo") or "dividida"), ["completa", "dividida", "quitar"])
    out.sort(key=lambda x: (x["ini_s"], x["k"][0] != "m"))
    return {"tramos": out, "duracion": round(fin_ed, 3), "estilo": est}


def pista_cara_en(video, caja, a, b):
    """Pista de la cara más grande dentro de la caja entre a y b (para encuadrar un video a pantalla completa)."""
    try:
        det = ed.DetectorCara()
    except Exception:
        return []
    cap = cv2.VideoCapture(str(video))
    x, y, w, h = [int(v) for v in caja]
    pista = []
    for tt in np.arange(a, b, 0.5):
        cap.set(cv2.CAP_PROP_POS_MSEC, tt * 1000)
        ok, fr = cap.read()
        if not ok:
            break
        c = det.mayor(fr[y:y + h, x:x + w])
        if c and c[3] > 40:
            pista.append([round(float(tt), 3), x + c[0] + c[2] / 2, y + c[1] + c[3] / 2, c[3]])
    cap.release()
    return pista if len(pista) >= 2 else []


def aplicar_pantalla(t, cambios):
    """Guarda lo elegido en la línea de tiempo de pantalla."""
    plan, _ = cargar(t)
    pl = plantilla(plan)
    respaldar(t)
    proy = plan.setdefault("proyeccion", [])
    nuevos_proy = []

    def tiempos(c, ini0, fin0):
        a, b = seg(c.get("inicio")), seg(c.get("fin"))
        a = orig_t(a, plan) if a is not None and mmss(a) != mmss(ed_t(ini0, plan)) else ini0
        b = orig_t(b, plan) if b is not None and mmss(b) != mmss(ed_t(fin0, plan)) else fin0
        return round(a, 3), round(max(b, a + 0.3), 3)

    for c in cambios:
        k, op = str(c.get("k", "")), c.get("pantalla")
        if k.startswith("p") and k[1:].isdigit() and int(k[1:]) < len(proy):
            p = proy[int(k[1:])]
            if op in ("completa", "dividida", "nacho"):
                p["pantalla"] = op
            p["inicio"], p["fin"] = tiempos(c, p["inicio"], p["fin"])
        elif k.startswith("g"):
            try:
                a0, b0 = [float(x) for x in k[1:].split("_")]
            except ValueError:
                continue
            if op not in ("completa", "dividida"):
                continue
            a, b = tiempos(c, a0, b0)
            # misma caja que el tramo de proyección más cercano (o la de la plantilla)
            cerca = sorted(proy, key=lambda p: min(abs(p["inicio"] - b), abs(p["fin"] - a)))
            caja = (cerca[0].get("caja") if cerca else None) or [int(v) for v in pl["fuente_16x9"]["proyeccion"]]
            item = {"tipo": "video", "inicio": a, "fin": b, "caja": caja, "pantalla": op, "manual": True}
            cara = pista_cara_en(plan["video"], caja, a, b)
            if cara:
                item["cara"] = cara
            nuevos_proy.append(item)
        elif k.startswith("e") and k[1:].isdigit():
            enc = plan.get("encuadres", [])
            i = int(k[1:])
            if i < len(enc):
                if op in ("cerca", "lejos"):
                    enc[i]["zoom"] = op
                enc[i]["inicio"], enc[i]["fin"] = tiempos(c, enc[i]["inicio"], enc[i]["fin"])
        elif k.startswith("m") and k[1:].isdigit():
            moms = plan.get("momentos", [])
            i = int(k[1:])
            if i < len(moms):
                m = moms[i]
                m["oculto"] = op == "quitar"
                if op in ("completa", "dividida"):
                    m["estilo"] = op
                m["inicio"], m["fin"] = tiempos(c, m["inicio"], m["fin"])
    proy.extend(nuevos_proy)
    proy.sort(key=lambda x: x["inicio"])
    guardar_plan(t, plan)
    return len(cambios)


def guardar_plan(trabajo, plan):
    ed.guardar_json(Path(trabajo) / "plan.json", plan)


def video_salida(trabajo):
    if not trabajo:
        return None
    plan, _ = cargar(trabajo)
    for fin in ("_vertical.mp4", "_subtitulado.mp4"):
        p = Path(trabajo) / (Path(plan["video"]).stem + fin)
        if p.exists():
            return str(p)
    return None


def mmss(t):
    """Segundos -> 'MM:SS:CC' (minutos : segundos : centésimas)."""
    cc = int(round(max(0.0, float(t)) * 100))
    return f"{cc // 6000:02d}:{cc // 100 % 60:02d}:{cc % 100:02d}"


def seg(valor):
    """'MM:SS:CC', 'MM:SS' o segundos ('75.5') -> segundos. None si está vacío o no se entiende."""
    if valor is None:
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    txt = str(valor).strip().replace(",", ".").replace(";", ":").replace(" ", "")
    if not txt or txt.lower() == "nan":
        return None
    try:
        partes = txt.split(":")
        if len(partes) == 1:
            return float(partes[0])
        if len(partes) == 2:
            return int(partes[0] or 0) * 60 + float(partes[1] or 0)
        if len(partes) == 3:
            cc = partes[2] or "0"
            return int(partes[0] or 0) * 60 + int(partes[1] or 0) + int(cc.ljust(2, "0")[:2]) / 100
    except ValueError:
        return None
    return None


# ----------------------------------------------------------------------------
# Subtítulos
# ----------------------------------------------------------------------------
def ed_t(t, plan):
    """Tiempo original -> tiempo en el video ya editado (lo que ve el usuario)."""
    return ed.a_editado(t, plan.get("cortes", []))


def orig_t(te, plan):
    """Tiempo que escribe el usuario (video editado) -> tiempo del video original."""
    return ed.a_original(te, plan.get("cortes", []))


def tabla_subtitulos(trabajo):
    """Los tiempos se muestran como aparecen en el video ya editado."""
    if not trabajo:
        return []
    plan, palabras = cargar(trabajo)
    return [[mmss(ed_t(g["i"], plan)), mmss(ed_t(g["f"], plan)), " ".join(p["p"] for p in g["palabras"])]
            for g in grupos_actuales(trabajo, plan, palabras)]


def filas_de(tabla):
    """Acepta lista de listas o DataFrame de pandas."""
    if tabla is None:
        return []
    if hasattr(tabla, "values"):
        return tabla.values.tolist()
    return list(tabla)


def texto_con_saltos(palabras):
    """Texto del subtítulo con los saltos de línea que puso el usuario (Shift+Enter en el editor)."""
    out = ""
    for k, p in enumerate(palabras):
        out += p["p"]
        if k + 1 < len(palabras):
            out += "\n" if p.get("salto") else " "
    return out


def leer_saltos(texto):
    """'a Donald Trump\na un proyecto' -> (['a','Donald','Trump','a','un','proyecto'], {2})."""
    palabras, saltos = [], set()
    for linea in str(texto).replace("\r", "").split("\n"):
        ws = linea.split()
        if not ws:
            continue
        if palabras:
            saltos.add(len(palabras) - 1)
        palabras += ws
    return palabras, saltos


def guardar_subtitulos(trabajo, tabla):
    plan, palabras = cargar(trabajo)
    viejos = grupos_actuales(trabajo, plan, palabras)
    nuevos = []
    for n, fila in enumerate(filas_de(tabla)):
        if len(fila) < 3:
            continue
        ini_e, fin_e = seg(fila[0]), seg(fila[1])
        if ini_e is None or fin_e is None:
            continue
        texto = str(fila[2] or "").strip()
        if not texto or texto.lower() == "nan":
            continue  # fila vacía = subtítulo quitado
        palabras_nuevas, saltos = leer_saltos(texto)
        if not palabras_nuevas:
            continue
        # la fila puede traer el número del subtítulo original (el editor en vivo lo manda; -1 = nuevo)
        ref = n
        if len(fila) >= 4 and str(fila[3]).lstrip("-").isdigit():
            ref = int(fila[3])
        viejo = viejos[ref] if 0 <= ref < len(viejos) else None
        try:
            dy = int(round(float(fila[4]))) if len(fila) >= 5 and str(fila[4]).strip() not in ("", "nan") else \
                int((viejo or {}).get("dy", 0) or 0)
        except ValueError:
            dy = 0
        saltos_viejos = {k for k, p in enumerate(viejo["palabras"]) if p.get("salto")} if viejo else set()
        if viejo and [p["p"] for p in viejo["palabras"]] == palabras_nuevas and saltos_viejos == saltos and \
                mmss(ed_t(viejo["i"], plan)) == mmss(ini_e) and mmss(ed_t(viejo["f"], plan)) == mmss(fin_e):
            nuevos.append(dict(viejo, dy=dy) if dy else {k: v for k, v in viejo.items() if k != "dy"})
            continue
        # los tiempos de la tabla son del video editado; se guardan en tiempo del original
        ini, fin = orig_t(ini_e, plan), orig_t(fin_e, plan)
        if viejo and mmss(ed_t(viejo["i"], plan)) == mmss(ini_e):
            ini = viejo["i"]
        if viejo and mmss(ed_t(viejo["f"], plan)) == mmss(fin_e):
            fin = viejo["f"]
        if fin <= ini:
            fin = ini + 0.5
        if viejo and len(viejo["palabras"]) == len(palabras_nuevas) and viejo["f"] > viejo["i"]:
            # mismo número de palabras: conserva el ritmo original, ajustado a los tiempos nuevos
            esc = (fin - ini) / (viejo["f"] - viejo["i"])
            pal = [{"p": t, "i": round(ini + (p["i"] - viejo["i"]) * esc, 3), "f": round(ini + (p["f"] - viejo["i"]) * esc, 3)}
                   for t, p in zip(palabras_nuevas, viejo["palabras"])]
        else:
            paso = (fin - ini) / len(palabras_nuevas)
            pal = [{"p": t, "i": round(ini + k * paso, 3), "f": round(ini + (k + 1) * paso, 3)}
                   for k, t in enumerate(palabras_nuevas)]
        for k in saltos:
            if k < len(pal) - 1:
                pal[k]["salto"] = True
        nuevos.append({"i": round(ini, 3), "f": round(fin, 3), "palabras": pal, **({"dy": dy} if dy else {})})
    nuevos.sort(key=lambda g: g["i"])
    respaldar(trabajo)
    ed.guardar_json(Path(trabajo) / "subtitulos.json", nuevos)
    return len(nuevos)


# ----------------------------------------------------------------------------
# Momentos e imágenes
# ----------------------------------------------------------------------------
def opciones_momentos(plan):
    return [f"{i} · {mmss(ed_t(m['inicio'], plan))}–{mmss(ed_t(m['fin'], plan))} · {m.get('busqueda', '')[:40]}"
            + ("" if m.get("imagenes") else " (sin imágenes)")
            for i, m in enumerate(plan.get("momentos", []), 1)]


def num_momento(etiqueta):
    try:
        return int(str(etiqueta).split("·")[0].strip())
    except (ValueError, IndexError):
        return None


def area_de(plan, estilo):
    pl = plantilla(plan)
    if estilo == "completa" or "layouts" not in pl:
        return pl["salida"]["ancho"], pl["salida"]["alto"]
    a = pl["layouts"]["dividida"]["inferior"]["area"]
    return a[2], a[3]


def imagen_vista(ruta_img, max_lado=900):
    img = ed.leer_img(ruta_img)
    if img is None:
        return None
    h, w = img.shape[:2]
    esc = min(1.0, max_lado / max(h, w))
    return cv2.resize(img, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA) if esc < 1 else img


def datos_momento(t, n):
    """Datos para el editor de movimiento: marcos de inicio/final de cada imagen del momento n."""
    plan, _ = cargar(t)
    pl = plantilla(plan)
    m = plan["momentos"][n - 1]
    w, h = area_de(plan, m.get("estilo", "dividida"))
    con_imgs = [x for x in plan.get("momentos", []) if x.get("imagenes")]
    lista = m.get("encuadres") or []
    imgs = []
    for k, ruta_img in enumerate(m.get("imagenes", [])):
        img = ed.leer_img(ruta_img)
        if img is None:
            continue
        ih, iw = img.shape[:2]
        guardado = k < len(lista) and isinstance(lista[k], dict) and bool(lista[k].get("inicio"))
        idx = (con_imgs.index(m) if m in con_imgs else 0) * 3 + k
        e = ed.encuadre_de(m, k, idx, pl["momentos"], img)
        a, b, suav = ed.marcos(e, iw, ih, w, h)
        imgs.append({"k": k, "url": f"/mov/img?t={quote(t)}&n={n}&k={k}&v={int(time.time())}",
                     "inicio": a, "fin": b, "suavizado": suav if guardado else "suave", "guardado": guardado})
    dur = (m["fin"] - m["inicio"]) / max(1, len(m.get("imagenes", [])))
    return {"estilo": m.get("estilo", "dividida"), "area": [w, h], "duracion_img": dur, "imagenes": imgs}


def iframe_editor(t, n):
    if not t or not n:
        return "<div style='padding:20px;color:#888'>Selecciona un momento.</div>"
    url = f"/mov/editor?t={quote(t)}&n={n}&v={int(time.time() * 1000)}"
    return (f'<iframe src="{url}" style="width:100%;height:760px;border:0;border-radius:12px"></iframe>'
            f'<div style="text-align:right;font-size:13px"><a href="{url}" target="_blank">Abrir el editor en otra ventana ↗</a></div>')


def iframe_subs(t):
    if not t:
        return "<div style='padding:20px;color:#888'>Carga o procesa un video para editar sus subtítulos.</div>"
    url = f"/sub/editor?t={quote(t)}&v={int(time.time() * 1000)}"
    return (f'<iframe src="{url}" style="width:100%;height:820px;border:0;border-radius:12px"></iframe>'
            f'<div style="text-align:right;font-size:13px"><a href="{url}" target="_blank">Abrir en otra ventana ↗</a></div>')


def iframe_pant(t):
    if not t:
        return "<div style='padding:20px;color:#888'>Carga o procesa un video.</div>"
    url = f"/pant/editor?t={quote(t)}&v={int(time.time() * 1000)}"
    return (f'<iframe src="{url}" style="width:100%;height:820px;border:0;border-radius:12px"></iframe>'
            f'<div style="text-align:right;font-size:13px"><a href="{url}" target="_blank">Abrir en otra ventana ↗</a></div>')


def video_vista(t, plan=None):
    """Clip SIN subtítulos para el editor en vivo cuando aún no se renderiza (Solo subtítulos):
    el clip original si el navegador lo reproduce, o su copia ligera (vista_clip.mp4)."""
    if plan is None:
        plan, _ = cargar(t)
    if not plan.get("vista_original"):
        return None
    ligera = Path(t) / "vista_clip.mp4"
    if ligera.exists():
        return ligera
    v = Path(plan["video"])
    return v if v.exists() else None


def diseno_solo_subtitulos(plan, cfg):
    """[[t, diseño]] para el editor en vivo a partir del análisis (título y franja azul), sin render."""
    base = "frases" if cfg.get("modo") == "frases" else "persona"
    marcas = []
    if plan.get("titulo_fin"):
        marcas.append([0.0, "titulo"])
        marcas.append([plan["titulo_fin"], base])
    else:
        marcas.append([0.0, base])
    for f in plan.get("franjas", []):
        marcas.append([f[0], "dividida"])
        marcas.append([f[1], base])
    marcas.sort(key=lambda m: m[0])
    return [[round(a, 3), b] for a, b in marcas]


def datos_subtitulos(t):
    """Todo lo que necesita el editor de subtítulos en vivo (tiempos ya en el video editado)."""
    plan, palabras = cargar(t)
    pl = plantilla(plan)
    cfg = pl["subtitulos"]
    cs = plan.get("cortes", [])
    grupos = []
    for g in grupos_actuales(t, plan, palabras):
        grupos.append({"i": round(ed_t(g["i"], plan), 3), "f": round(ed_t(g["f"], plan), 3),
                       "dy": int(g.get("dy", 0) or 0),
                       "texto": texto_con_saltos(g["palabras"]),
                       "palabras": [{"p": p["p"], "i": round(ed.a_editado(p["i"], cs), 3),
                                     "f": round(ed.a_editado(p["f"], cs), 3), "salto": bool(p.get("salto"))}
                                    for p in g["palabras"]]})
    vp = Path(t) / "vista_previa.json"
    diseno = ed.cargar_json(vp)["diseno"] if vp.exists() else []
    if not diseno and plan.get("estilo") == "solo_subtitulos":
        diseno = diseno_solo_subtitulos(plan, cfg)
    areas = {k: v["subtitulos"]["area"] for k, v in pl.get("layouts", {}).items() if "subtitulos" in v}
    salida = [pl["salida"].get("ancho", 1080), pl["salida"].get("alto", 1920)]
    if plan.get("subtitulos_render"):  # solo subtítulos: medidas del video real
        sr = plan["subtitulos_render"]
        cfg, areas, salida = sr["estilo"], sr["areas"], sr["salida"]
    modo = cfg.get("modo", "palabras")
    return {
        "grupos": grupos, "modo": modo, "diseno": diseno, "areas": areas,
        "salida": salida,
        "estilo": {k: cfg.get(k) for k in ("tam", "tam_min", "max_lineas", "color", "contorno", "color_contorno",
                                            "color_resalte", "espacio", "mayusculas", "centro", "ancho_max",
                                            "interlineado", "sombra", "sombra_opacidad")},
        "limpia": (Path(t) / "vista_previa.mp4").exists() or bool(video_vista(t, plan)),
        "hay_video": (Path(t) / "vista_previa.mp4").exists() or bool(video_vista(t, plan)) or bool(video_salida(t)),
    }


def crear_app():
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, Response

    app = FastAPI()

    def validar(t):
        t = limpiar_ruta(t)
        if not t or not (Path(t) / "plan.json").exists():
            raise HTTPException(404, "Carpeta de trabajo no válida")
        return t

    @app.get("/mov/editor")
    def pagina():
        return FileResponse(BASE / "web" / "movimiento.html", headers={"Cache-Control": "no-store"})

    @app.get("/mov/datos")
    def datos(t: str, n: int):
        return JSONResponse(datos_momento(validar(t), n))

    @app.get("/mov/img")
    def imagen(t: str, n: int, k: int):
        plan, _ = cargar(validar(t))
        img = imagen_vista(plan["momentos"][n - 1]["imagenes"][k], 1400)
        if img is None:
            raise HTTPException(404, "No pude leer la imagen")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return Response(buf.tobytes(), media_type="image/jpeg")

    @app.post("/mov/guardar")
    def guardar(d: dict = Body(...)):
        t = validar(d.get("t"))
        n, k = int(d["n"]), int(d["k"])
        plan, _ = cargar(t)
        respaldar(t)
        m = plan["momentos"][n - 1]
        lista = m.get("encuadres") or []
        while len(lista) < len(m["imagenes"]):
            lista.append({})
        lista[k] = {"inicio": [round(float(v), 5) for v in d["inicio"]],
                    "fin": [round(float(v), 5) for v in d["fin"]],
                    "suavizado": "lineal" if d.get("suavizado") == "lineal" else "suave"}
        m["encuadres"] = lista
        guardar_plan(t, plan)
        return {"ok": True}

    @app.post("/mov/quitar_imagen")
    def quitar_imagen(d: dict = Body(...)):
        t = validar(d.get("t"))
        n, k = int(d["n"]), int(d["k"])
        plan, _ = cargar(t)
        m = plan["momentos"][n - 1]
        if not 0 <= k < len(m.get("imagenes", [])):
            raise HTTPException(400, "Esa imagen no existe")
        respaldar(t)
        m["imagenes"].pop(k)
        enc = m.get("encuadres") or []
        if k < len(enc):
            enc.pop(k)
        m["encuadres"] = enc
        guardar_plan(t, plan)
        return {"ok": True, "quedan": len(m["imagenes"])}

    @app.post("/mov/restablecer")
    def restablecer(d: dict = Body(...)):
        t = validar(d.get("t"))
        n, k = int(d["n"]), int(d["k"])
        plan, _ = cargar(t)
        respaldar(t)
        m = plan["momentos"][n - 1]
        lista = m.get("encuadres") or []
        if k < len(lista):
            lista[k] = {}
        m["encuadres"] = lista
        guardar_plan(t, plan)
        return next(i for i in datos_momento(t, n)["imagenes"] if i["k"] == k)

    # ---------------- subtítulos en vivo ----------------
    @app.get("/sub/editor")
    def sub_pagina():
        return FileResponse(BASE / "web" / "subtitulos.html", headers={"Cache-Control": "no-store"})

    @app.get("/sub/datos")
    def sub_datos(t: str):
        return JSONResponse(datos_subtitulos(validar(t)))

    @app.get("/sub/video")
    def sub_video(t: str):
        t = validar(t)
        p = Path(t) / "vista_previa.mp4"
        if not p.exists():
            p = video_vista(t)
        if not p:
            v = video_salida(t)
            if not v:
                raise HTTPException(404, "Aún no hay video")
            p = Path(v)
        return FileResponse(p, media_type="video/mp4", headers={"Cache-Control": "no-store"})

    @app.get("/sub/fuente")
    def sub_fuente(t: str):
        plan, _ = cargar(validar(t))
        f = ed.ruta(plantilla(plan)["subtitulos"]["fuente"])
        return FileResponse(f, media_type="font/otf" if str(f).lower().endswith(".otf") else "font/ttf")

    @app.post("/sub/guardar")
    def sub_guardar(d: dict = Body(...)):
        t = validar(d.get("t"))
        filas = [[str(f.get("i", "")), str(f.get("f", "")), str(f.get("texto", "")), str(f.get("id", -1)),
                  str(f.get("dy", 0))] for f in d.get("filas", [])]
        n = guardar_subtitulos(t, filas)
        return {"ok": True, "n": n, "datos": datos_subtitulos(t)}

    # ---------------- pantalla por tramo ----------------
    @app.get("/pant/editor")
    def pant_pagina():
        return FileResponse(BASE / "web" / "pantalla.html", headers={"Cache-Control": "no-store"})

    @app.get("/pant/datos")
    def pant_datos(t: str):
        return JSONResponse(tramos_pantalla(validar(t)))

    @app.get("/pant/cuadro")
    def pant_cuadro(t: str, k: str):
        t = validar(t)
        plan, _ = cargar(t)
        img = None
        if k.startswith("m") and k[1:].isdigit():
            m = plan["momentos"][int(k[1:])]
            img = imagen_vista(m["imagenes"][0], 360) if m.get("imagenes") else None
        else:
            if k.startswith("p") and k[1:].isdigit():
                p = plan["proyeccion"][int(k[1:])]
                a, b, caja = p["inicio"], p["fin"], p.get("caja")
            elif k.startswith("g"):
                a, b = [float(x) for x in k[1:].split("_")]
                caja = [int(v) for v in plantilla(plan)["fuente_16x9"]["proyeccion"]]
            elif k.startswith("e") and k[1:].isdigit():
                e = plan["encuadres"][int(k[1:])]
                a, b, caja = e["inicio"], min(e["fin"], plan["info"]["duracion"]), None
            else:
                raise HTTPException(404, "Tramo no válido")
            cap = cv2.VideoCapture(plan["video"])
            cap.set(cv2.CAP_PROP_POS_MSEC, (a + min(1.5, (b - a) / 2)) * 1000)
            ok, fr = cap.read()
            cap.release()
            if ok:
                x, y, w, h = [int(v) for v in caja or [0, 0, fr.shape[1], fr.shape[0]]]
                img = fr[y:y + h, x:x + w]
                esc = 360 / max(img.shape[:2])
                img = cv2.resize(img, (max(1, int(img.shape[1] * esc)), max(1, int(img.shape[0] * esc))))
        if img is None:
            raise HTTPException(404, "No pude leer el cuadro")
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "max-age=600"})

    @app.post("/pant/guardar")
    def pant_guardar(d: dict = Body(...)):
        t = validar(d.get("t"))
        n = aplicar_pantalla(t, d.get("cambios", []))
        return {"ok": True, "n": n}

    return app


# ----------------------------------------------------------------------------
# Procesos largos con barra de progreso
# ----------------------------------------------------------------------------
def barra(frac=None, etapa="", estado="trabajando", detalle=""):
    """HTML de la barra de progreso que va arriba del video."""
    if frac is None:
        return ""
    pct = int(round(100 * min(1.0, max(0.0, frac))))
    color = {"trabajando": "#5b5bd6", "listo": "#16a34a", "error": "#dc2626"}[estado]
    texto = f"{etapa}… {pct}%" if estado == "trabajando" else etapa
    extra = (f'<div style="margin-top:6px;font-size:12px;color:#b91c1c;white-space:pre-wrap">{detalle}</div>'
             if detalle else "")
    return (f'<div style="padding:4px 2px 8px">'
            f'<div style="font-weight:600;font-size:14px;margin-bottom:6px;color:{color}">{texto}</div>'
            f'<div style="background:rgba(127,127,127,.18);border-radius:999px;height:12px;overflow:hidden">'
            f'<div style="width:{pct}%;height:100%;background:{color};border-radius:999px;transition:width .4s"></div>'
            f'</div>{extra}</div>')


# Registro: todo lo que va haciendo la plataforma (para ver cómo trabaja). Se guarda también en registro.log
REGISTRO = []
REGISTRO_MAX = 4000
ARCHIVO_REGISTRO = BASE / "registro.log"
_reg_lock = threading.Lock()


def registrar(linea):
    linea = f"{time.strftime('%H:%M:%S')}  {linea}"
    with _reg_lock:
        REGISTRO.append(linea)
        if len(REGISTRO) > REGISTRO_MAX:
            del REGISTRO[:len(REGISTRO) - REGISTRO_MAX]
        try:
            with open(ARCHIVO_REGISTRO, "a", encoding="utf-8") as f:
                f.write(linea + "\n")
        except OSError:
            pass


def texto_registro():
    with _reg_lock:
        return "\n".join(REGISTRO[-1500:]) or "Aquí aparece lo que va haciendo la plataforma cuando procesas un video."


def limpiar_registro():
    with _reg_lock:
        REGISTRO.clear()
    return texto_registro()


def correr(cmd):
    """Corre editor.py y va devolviendo (html_barra, resultado_final_o_None)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(BASE), env=env)
    lineas, avisos = [], []
    nombres = {"todo": "Analizar y armar", "render": "Renderizar", "convertir": "Convertir a MP4", "analizar": "Analizar"}
    registrar("─" * 60)
    registrar(f"▶ {nombres.get(cmd[2] if len(cmd) > 2 else '', 'Proceso')}: {Path(cmd[3]).name if len(cmd) > 3 else ''}")
    inicio = time.time()
    etapa_ant = None
    yield barra(0.0, "Preparando"), None
    for raw in iter(proc.stdout.readline, b""):
        linea = raw.decode("utf-8", errors="replace").rstrip()
        if linea.startswith("@@AVISO@@"):
            avisos.append("⚠ " + linea.split(" ", 1)[1])
            registrar("⚠ " + linea.split(" ", 1)[1])
            continue
        if linea.startswith("@@PROGRESO@@"):
            try:
                valor, etapa = linea.split(" ", 1)[1].split("|", 1)
                if etapa != etapa_ant:
                    registrar(f"● {etapa}")
                    etapa_ant = etapa
                yield barra(float(valor), etapa), None
            except ValueError:
                pass
            continue
        if "WARN" in linea and "dnn" in linea:
            continue
        lineas.append(linea)
        if linea.strip():
            registrar("   " + linea.replace("[editor] ", ""))
    proc.wait()
    seg = int(time.time() - inicio)
    registrar(f"{'✔ Terminó' if proc.returncode == 0 else '✖ Falló'} en {seg // 60} min {seg % 60} s")
    final = None
    for linea in reversed(lineas):
        if linea.startswith("{"):
            try:
                final = json.loads(linea)
            except json.JSONDecodeError:
                pass
            break
    if proc.returncode == 0 and final and final.get("ok"):
        yield barra(1.0, "Listo ✓", "listo", "\n".join(avisos).replace("<", "&lt;")), final
    else:
        utiles = [x for x in lineas if x.strip() and not x.startswith("[editor]")][-6:] or lineas[-6:]
        detalle = "\n".join(utiles).replace("<", "&lt;")
        yield barra(1.0, "Algo falló. Esto fue lo último que pasó:", "error", detalle), {"ok": False}


def unidades():
    if os.name == "nt":
        return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]
    return ["/"]


# ----------------------------------------------------------------------------
# Interfaz
# ----------------------------------------------------------------------------
def construir():
    extra = {"theme": gr.themes.Soft()} if _TEMA_EN_BLOCKS else {}
    with gr.Blocks(title="Editor vertical", **extra) as demo:
        trabajo = gr.State("")

        gr.Markdown("## Editor vertical · El Chapucero")
        with gr.Accordion("Procesar un video nuevo", open=True):
            gr.Markdown(f"Los videos se toman de **{CARPETA_VIDEOS}**. Escribe solo el nombre "
                        f"(si no pones extensión se busca **.mp4**; otros formatos se convierten a MP4 antes de empezar).")
            with gr.Row():
                ruta_video = gr.Textbox(label="Nombre del video", placeholder="VIDEO 4 NACHO MAÑANA", scale=3)
                titulo_nuevo = gr.Textbox(label="Título (no hace falta en «Solo subtítulos»)", scale=2)
            estilo_edicion = gr.Radio(list(ESTILOS), value="Clips Chapuceros", label="Estilo de edición")
            with gr.Row():
                usar_gpu = gr.Checkbox(label="Solo tarjeta gráfica (cuidar el procesador)", value=True,
                                       info="Whisper y la codificación en la GPU; nunca cae al procesador")
                btn_procesar = gr.Button("Analizar y armar video", variant="primary")
        with gr.Row():
            ruta_trabajo = gr.Textbox(label="…o carga una carpeta _trabajo existente", scale=4)
            btn_cargar = gr.Button("Cargar", scale=1)

        with gr.Row():
            with gr.Column(scale=2, min_width=320):
                progreso = gr.HTML()
                video = gr.Video(label="Video", height=620, interactive=False)
                btn_render = gr.Button("Renderizar de nuevo", variant="primary")
                btn_deshacer = gr.Button("Deshacer último cambio")
                aviso = gr.Markdown()
            with gr.Column(scale=4):
                with gr.Tabs():
                    # ---------------- Subtítulos ----------------
                    with gr.Tab("Subtítulos"):
                        editor_subs = gr.HTML(iframe_subs(None))

                    # ---------------- Imágenes ----------------
                    with gr.Tab("Imágenes"):
                        sel_momento = gr.Dropdown(label="Momento", choices=[])
                        with gr.Row():
                            estilo = gr.Radio(["dividida", "completa"], label="Pantalla", value="dividida")
                            m_ini = gr.Textbox(label="Inicio (min:seg:cent)", placeholder="00:00:00")
                            m_fin = gr.Textbox(label="Fin (min:seg:cent)", placeholder="00:00:00")
                            btn_guardar_img = gr.Button("Guardar pantalla y tiempos", variant="primary")
                        editor_mov = gr.HTML()
                        with gr.Accordion("Cambiar imágenes de este momento", open=False):
                            modo_img = gr.Radio(["Agregar a las que ya tiene", "Reemplazar las que tiene"],
                                                value="Agregar a las que ya tiene", show_label=False)
                            with gr.Row():
                                nueva_busqueda = gr.Textbox(label="Buscar otras imágenes en internet", scale=4)
                                btn_buscar = gr.Button("Buscar", scale=1)
                            with gr.Row():
                                archivos = gr.File(file_count="multiple", file_types=["image"], label="Subir mis imágenes")
                                btn_subir = gr.Button("Subir")
                        with gr.Accordion("Agregar o quitar momentos", open=False):
                            with gr.Row():
                                nuevo_ini = gr.Textbox(label="Inicio (min:seg:cent)", placeholder="00:00:00")
                                nuevo_fin = gr.Textbox(label="Fin (min:seg:cent)", placeholder="00:00:00")
                                nuevo_estilo = gr.Radio(["dividida", "completa"], value="dividida", label="Pantalla")
                            nuevo_busq = gr.Textbox(label="Qué buscar (ej. Ernesto Ruffo Appel)")
                            with gr.Row():
                                btn_agregar = gr.Button("Agregar momento")
                                btn_quitar = gr.Button("Quitar el momento seleccionado", variant="stop")

                    # ---------------- Título y cortes ----------------
                    with gr.Tab("Pantalla") as tab_pantalla:
                        editor_pant = gr.HTML(iframe_pant(None))

                    with gr.Tab("Título y cortes"):
                        titulo = gr.Textbox(label="Título")
                        btn_titulo = gr.Button("Guardar título")
                        gr.Markdown("**Quitar un tramo del video** (imagen y audio), con los tiempos del video ya editado (min:seg:cent):")
                        with gr.Row():
                            corte_ini = gr.Textbox(label="Desde (min:seg:cent)", placeholder="00:00:00")
                            corte_fin = gr.Textbox(label="Hasta (min:seg:cent)", placeholder="00:00:00")
                            btn_cortar = gr.Button("Quitar tramo")
                        cortes_info = gr.Markdown()

        # ---------------- Registro (abajo, aparte) ----------------
        with gr.Tabs():
            with gr.Tab("Registro de la plataforma"):
                gr.Markdown("Lo que va haciendo la plataforma paso a paso (se actualiza solo). "
                            f"También se guarda en `{ARCHIVO_REGISTRO.name}` dentro de la carpeta del editor.")
                caja_registro = gr.Textbox(value=texto_registro, show_label=False, lines=18, max_lines=18,
                                           interactive=False, autoscroll=True, elem_id="caja_registro")
                with gr.Row():
                    btn_reg_act = gr.Button("Actualizar", scale=1)
                    btn_reg_limpiar = gr.Button("Limpiar", scale=1)
        btn_reg_act.click(texto_registro, None, caja_registro)
        btn_reg_limpiar.click(limpiar_registro, None, caja_registro)
        if hasattr(gr, "Timer"):
            gr.Timer(2.0).tick(texto_registro, None, caja_registro, show_progress="hidden")

        # ======================== lógica ========================
        tab_pantalla.select(iframe_pant, trabajo, editor_pant)
        def info_cortes(t):
            if not t:
                return ""
            plan, _ = cargar(t)
            cs = plan.get("cortes", [])
            manuales = [c for c in cs if c.get("manual")]
            txt = f"Pausas recortadas automáticamente: {len(cs) - len(manuales)}."
            if manuales:
                txt += " Tramos quitados a mano (en el video original): " + \
                    ", ".join(f"{mmss(c['inicio'])}–{mmss(c['fin'])}" for c in manuales)
            return txt

        def todo_ui(t, momento_sel=None):
            """Valores para refrescar toda la interfaz con el trabajo t."""
            plan, _ = cargar(t)
            ops = opciones_momentos(plan)
            sel = momento_sel if momento_sel in ops else (ops[0] if ops else None)
            return (t, iframe_subs(t), gr.update(choices=ops, value=sel), plan.get("titulo", ""),
                    video_salida(t), info_cortes(t))

        salidas_ui = [trabajo, editor_subs, sel_momento, titulo, video, cortes_info]

        def accion_cargar(ruta):
            t = limpiar_ruta(ruta)
            if not (Path(t) / "plan.json").exists():
                gr.Warning("Esa carpeta no tiene plan.json")
                return ("", iframe_subs(None), gr.update(choices=[], value=None), "", None, "", "")
            return (*todo_ui(t), "")

        btn_cargar.click(accion_cargar, ruta_trabajo, salidas_ui + [progreso])

        def accion_procesar(ruta, tit, gpu, est):
            encontrado = buscar_video(ruta)
            if not encontrado:
                gr.Warning(f"No encuentro «{limpiar_ruta(ruta)}» en {CARPETA_VIDEOS}")
                yield ("", gr.update(), gr.update(), "", None, "",
                       barra(1.0, f"No encontré «{limpiar_ruta(ruta)}» en {CARPETA_VIDEOS}", "error"))
                return
            v = str(encontrado)
            if not tit.strip() and not est.startswith("Solo subtítulos"):
                gr.Warning("Escribe un título")
                yield ("", gr.update(), gr.update(), "", None, "", "")
                return
            if encontrado.suffix.lower() != ".mp4":
                # primero convertir (solo cuando el video no es MP4)
                convertido = None
                for html, final in correr([sys.executable, str(BASE / "editor.py"), "convertir", v] + (["--solo-gpu"] if gpu else [])):
                    if final is not None and final.get("ok"):
                        convertido = final["video"]
                    elif final is not None:
                        yield (gr.update(),) * 6 + (html,)
                        return
                    else:
                        yield (gr.update(),) * 6 + (html,)
                v = convertido
            cmd = [sys.executable, str(BASE / "editor.py"), "todo", v, "--titulo", tit.strip(),
                   "--plantilla", str(BASE / ESTILOS.get(est, ESTILOS["Clips Chapuceros"]))] + (["--gpu", "--solo-gpu", "--dispositivo", "cuda"] if gpu else [])
            for html, final in correr(cmd):
                if final is not None and final.get("ok"):
                    yield (*todo_ui(final["trabajo"]), html)
                else:
                    yield (gr.update(),) * 6 + (html,)

        btn_procesar.click(accion_procesar, [ruta_video, titulo_nuevo, usar_gpu, estilo_edicion], salidas_ui + [progreso])

        def accion_render(t, gpu):
            if not t:
                gr.Warning("Primero carga o procesa un video")
                yield gr.update(), "", gr.update()
                return
            cmd = [sys.executable, str(BASE / "editor.py"), "render", t] + (["--gpu", "--solo-gpu"] if gpu else [])
            for html, final in correr(cmd):
                if final is not None and final.get("ok"):
                    yield video_salida(t), html, iframe_subs(t)
                else:
                    yield gr.update(), html, gr.update()

        btn_render.click(accion_render, [trabajo, usar_gpu], [video, progreso, editor_subs])

        def accion_deshacer(t, momento_sel):
            if not t:
                return (gr.update(),) * 6 + ("",)
            ok = deshacer(t)
            return (*todo_ui(t, momento_sel), "Deshice el último cambio." if ok else "No hay cambios para deshacer.")

        btn_deshacer.click(accion_deshacer, [trabajo, sel_momento], salidas_ui + [aviso])


        # ---- imágenes ----
        def accion_momento(t, etiqueta):
            n = num_momento(etiqueta)
            if not t or not n:
                return gr.update(), None, None, iframe_editor(None, None)
            plan, _ = cargar(t)
            m = plan["momentos"][n - 1]
            return (m.get("estilo", "dividida"), mmss(ed_t(m["inicio"], plan)), mmss(ed_t(m["fin"], plan)),
                    iframe_editor(t, n))

        sel_momento.change(accion_momento, [trabajo, sel_momento], [estilo, m_ini, m_fin, editor_mov])

        def accion_guardar_img(t, etiqueta, est, ini, fin):
            n = num_momento(etiqueta)
            if not t or not n:
                return gr.update(), "Selecciona un momento."
            plan, _ = cargar(t)
            respaldar(t)
            m = plan["momentos"][n - 1]
            m["estilo"] = est
            ini, fin = seg(ini), seg(fin)
            if ini is not None and fin is not None and fin > ini:
                # solo cambia si el usuario movió el tiempo (evita redondeos)
                if mmss(ini) != mmss(ed_t(m["inicio"], plan)):
                    m["inicio"] = round(orig_t(ini, plan), 3)
                if mmss(fin) != mmss(ed_t(m["fin"], plan)):
                    m["fin"] = round(orig_t(fin, plan), 3)
            plan["momentos"].sort(key=lambda x: x["inicio"])
            guardar_plan(t, plan)
            nueva = plan["momentos"].index(m) + 1
            ops = opciones_momentos(plan)
            return gr.update(choices=ops, value=ops[nueva - 1]), f"Momento {nueva} guardado."

        btn_guardar_img.click(accion_guardar_img, [trabajo, sel_momento, estilo, m_ini, m_fin], [sel_momento, aviso]) \
            .then(accion_momento, [trabajo, sel_momento], [estilo, m_ini, m_fin, editor_mov])

        def refrescar_momento(t, n_nuevo):
            plan, _ = cargar(t)
            ops = opciones_momentos(plan)
            val = ops[n_nuevo - 1] if ops and 1 <= n_nuevo <= len(ops) else (ops[0] if ops else None)
            return gr.update(choices=ops, value=val)

        def poner_imagenes(m, nuevas, modo):
            if modo.startswith("Agregar"):
                previas = list(m.get("imagenes", []))
                enc = list(m.get("encuadres") or [])
                enc += [{}] * (len(previas) - len(enc))
                m["imagenes"] = previas + nuevas
                m["encuadres"] = enc + [{}] * len(nuevas)
            else:
                m["imagenes"] = nuevas
                m["encuadres"] = []

        def accion_buscar(t, etiqueta, busqueda, modo):
            n = num_momento(etiqueta)
            if not t or not n or not busqueda.strip():
                return gr.update(), "Selecciona un momento y escribe qué buscar."
            plan, _ = cargar(t)
            pl = plantilla(plan)
            m = plan["momentos"][n - 1]
            imgs = ed.buscar_imagenes([busqueda.strip()], pl["momentos"]["imagenes_por_momento"], Path(t) / "imagenes",
                                      f"momento{n:02d}_{time.strftime('%H%M%S')}", ia(), busqueda.strip(),
                                      pl["momentos"].get("max_candidatos", 8), pl["momentos"].get("puntaje_min", 7),
                                      usar_mejor=True)
            if not imgs:
                return gr.update(), f"No encontré imágenes que sirvan para «{busqueda}»."
            respaldar(t)
            m.update(busqueda=busqueda.strip(), busquedas=[busqueda.strip()])
            poner_imagenes(m, imgs, modo)
            guardar_plan(t, plan)
            return refrescar_momento(t, n), f"Momento {n}: {len(imgs)} imágenes nuevas (ahora tiene {len(m['imagenes'])})."

        btn_buscar.click(accion_buscar, [trabajo, sel_momento, nueva_busqueda, modo_img], [sel_momento, aviso]) \
            .then(accion_momento, [trabajo, sel_momento], [estilo, m_ini, m_fin, editor_mov])

        def accion_subir(t, etiqueta, files, modo):
            n = num_momento(etiqueta)
            if not t or not n or not files:
                return gr.update(), "Selecciona un momento y sube imágenes."
            plan, _ = cargar(t)
            respaldar(t)
            destino = Path(t) / "imagenes"
            destino.mkdir(exist_ok=True)
            nuevas = []
            for k, f in enumerate(files, 1):
                origen = Path(f if isinstance(f, str) else f.name)
                p = destino / f"usuario_m{n:02d}_{time.strftime('%H%M%S')}_{k}{origen.suffix.lower()}"
                shutil.copy2(origen, p)
                nuevas.append(str(p))
            m = plan["momentos"][n - 1]
            poner_imagenes(m, nuevas, modo)
            guardar_plan(t, plan)
            return refrescar_momento(t, n), f"Momento {n}: ahora tiene {len(m['imagenes'])} imagen(es)."

        btn_subir.click(accion_subir, [trabajo, sel_momento, archivos, modo_img], [sel_momento, aviso]) \
            .then(accion_momento, [trabajo, sel_momento], [estilo, m_ini, m_fin, editor_mov])

        def accion_agregar(t, ini, fin, est, busq):
            ini, fin = seg(ini), seg(fin)
            if not t or ini is None or fin is None or fin <= ini or not (busq or "").strip():
                return gr.update(), "Escribe inicio, fin (mayor que inicio) y qué buscar."
            plan, _ = cargar(t)
            pl = plantilla(plan)
            ini, fin = orig_t(ini, plan), orig_t(fin, plan)
            m = {"inicio": round(float(ini), 3), "fin": round(float(fin), 3), "estilo": est,
                 "busqueda": busq.strip(), "busquedas": [busq.strip()], "que_mostrar": busq.strip(), "texto": "", "imagenes": []}
            m["imagenes"] = ed.buscar_imagenes([busq.strip()], pl["momentos"]["imagenes_por_momento"], Path(t) / "imagenes",
                                               f"momento_nuevo_{time.strftime('%H%M%S')}", ia(), busq.strip(),
                                               pl["momentos"].get("max_candidatos", 8), pl["momentos"].get("puntaje_min", 7),
                                               usar_mejor=True)
            respaldar(t)
            plan.setdefault("momentos", []).append(m)
            plan["momentos"].sort(key=lambda x: x["inicio"])
            guardar_plan(t, plan)
            n = plan["momentos"].index(m) + 1
            extra = "" if m["imagenes"] else " No encontré imágenes: súbelas en «Cambiar imágenes de este momento»."
            return refrescar_momento(t, n), f"Momento {n} agregado.{extra}"

        btn_agregar.click(accion_agregar, [trabajo, nuevo_ini, nuevo_fin, nuevo_estilo, nuevo_busq], [sel_momento, aviso])

        def accion_quitar(t, etiqueta):
            n = num_momento(etiqueta)
            if not t or not n:
                return gr.update(), "Selecciona un momento."
            plan, _ = cargar(t)
            respaldar(t)
            plan["momentos"].pop(n - 1)
            guardar_plan(t, plan)
            return refrescar_momento(t, max(1, n - 1)), f"Quité el momento {n}."

        btn_quitar.click(accion_quitar, [trabajo, sel_momento], [sel_momento, aviso])

        # ---- título y cortes ----
        def accion_titulo(t, tit):
            if not t:
                return "Primero carga un video."
            plan, _ = cargar(t)
            respaldar(t)
            plan["titulo"] = tit.strip()
            guardar_plan(t, plan)
            return "Título guardado."

        btn_titulo.click(accion_titulo, [trabajo, titulo], aviso)

        def accion_cortar(t, a, b):
            a, b = seg(a), seg(b)
            if not t or a is None or b is None or b <= a:
                return gr.update(), "Escribe desde y hasta (hasta mayor que desde)."
            plan, _ = cargar(t)
            respaldar(t)
            fps = plan["info"]["fps"]
            a, b = orig_t(a, plan), orig_t(b, plan)  # lo escrito es tiempo del video editado
            a2 = round(int(float(a) * fps + 0.999) / fps, 4)
            b2 = round(int(float(b) * fps + 0.999) / fps, 4)
            plan.setdefault("cortes", []).append({"inicio": a2, "fin": b2, "manual": True})
            plan["cortes"].sort(key=lambda c: c["inicio"])
            guardar_plan(t, plan)
            return info_cortes(t), f"Quitaré ese tramo ({mmss(a2)}–{mmss(b2)} del video original)."

        btn_cortar.click(accion_cortar, [trabajo, corte_ini, corte_fin], [cortes_info, aviso])
    return demo


def main():
    import uvicorn
    try:
        CARPETA_VIDEOS.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    demo = construir().queue()
    opciones = {"allowed_paths": unidades()}
    params = inspect.signature(gr.mount_gradio_app).parameters
    if not _TEMA_EN_BLOCKS and "theme" in params:
        opciones["theme"] = gr.themes.Soft()
    app = gr.mount_gradio_app(crear_app(), demo, path="/", **opciones)
    puerto = int(os.environ.get("PUERTO", "7860"))
    url = f"http://127.0.0.1:{puerto}"
    print(f"Interfaz en {url}  (Ctrl+C para cerrar)")
    if os.environ.get("NO_NAVEGADOR") != "1":
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=puerto, log_level="warning")


if __name__ == "__main__":
    main()
