"""
Títulos de Clips Chapuceros (barra roja de arriba).

- La plantilla (assets/titulo_chapucero.png, 1080x1920 con transparencia) trae la barra roja.
- El texto va en AkuKamu (assets/titulo_fuente.otf), blanco, en mayúsculas y centrado.
- El usuario elige el tamaño máximo de letra, el espaciado entre letras y el interlineado; el texto NUNCA
  sale de la barra: si no cabe, se acomoda en 2 o 3 renglones y/o se achica lo necesario.
- Se guarda en titulo.json dentro de la carpeta del trabajo y se pone en el video al exportar.

La misma cuenta de acomodo está en web/subtitulos.html (vista previa), para que lo que se ve sea lo que sale.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

APP = Path(__file__).resolve().parent
PLANTILLA = APP / "assets" / "titulo_chapucero.png"
FUENTE = APP / "assets" / "titulo_fuente.otf"

REF_W, REF_H = 1080, 1920                     # medidas de la plantilla
CAJA = (0, 39, 1080, 277)                     # barra roja: x, y, ancho, alto (en la plantilla)
MARGEN_X, MARGEN_Y = 22, 20                   # aire dentro de la barra
POR_DEFECTO = {"texto": "", "tam": 150, "espaciado": 0, "interlineado": 1.0, "barra": True}
LIMITES = {"tam": (40, 260), "espaciado": (-10, 40), "interlineado": (0.7, 1.8)}
PASO_RENGLON = 1.32                           # distancia entre renglones = alto de mayúscula × esto × interlineado


def normalizar(d):
    """Valores dentro de sus límites."""
    out = dict(POR_DEFECTO)
    out.update({k: v for k, v in (d or {}).items() if k in POR_DEFECTO})
    out["texto"] = str(out.get("texto") or "").replace("\r", "")
    for k, (a, b) in LIMITES.items():
        try:
            out[k] = min(b, max(a, float(out[k])))
        except (TypeError, ValueError):
            out[k] = POR_DEFECTO[k]
    out["tam"] = int(round(out["tam"]))
    out["barra"] = bool(out.get("barra", True))
    return out


def cargar(trabajo):
    p = Path(trabajo) / "titulo.json"
    if not p.exists():
        return normalizar({})
    try:
        return normalizar(json.loads(p.read_text(encoding="utf-8-sig")))
    except Exception:
        return normalizar({})


def guardar(trabajo, d):
    d = normalizar(d)
    (Path(trabajo) / "titulo.json").write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


# ----------------------------------------------------------------------------
# Acomodo del texto (misma cuenta que la vista previa)
# ----------------------------------------------------------------------------
_fuentes = {}


def fuente(tam):
    tam = max(8, int(tam))
    if tam not in _fuentes:
        _fuentes[tam] = ImageFont.truetype(str(FUENTE), tam)
    return _fuentes[tam]


def ancho_linea(txt, tam, espaciado):
    f = fuente(tam)
    return sum(f.getlength(c) for c in txt) + espaciado * max(0, len(txt) - 1)


def alto_mayuscula(tam):
    b = fuente(tam).getbbox("H")
    return b[3] - b[1]


def tinta(lineas, tam, interlineado):
    """Cuánto sube y baja el texto (acentos y letras con cola incluidos) respecto al centro de la barra."""
    f = fuente(tam)
    cap = alto_mayuscula(tam)
    top_h = f.getbbox("H")[1]
    paso = cap * PASO_RENGLON * interlineado
    alto = cap + (len(lineas) - 1) * paso
    sup, inf = 0.0, 0.0
    for k, l in enumerate(lineas):
        b = f.getbbox(l)
        y_cap = -alto / 2 + k * paso            # arriba de la mayúscula de este renglón, desde el centro
        sup = max(sup, -(y_cap + (b[1] - top_h)))
        inf = max(inf, y_cap + (b[3] - top_h))
    return sup, inf


def opciones_renglones(texto):
    """Los renglones que puso el usuario; si no puso ninguno, también se prueba repartir en 2 y 3 renglones."""
    lineas = [l.strip() for l in texto.upper().split("\n") if l.strip()]
    if len(lineas) != 1:
        return [lineas]
    pal = lineas[0].split()
    ops = [[lineas[0]]]
    for n in (2, 3):
        if len(pal) < n:
            break
        mejor, peor = None, None
        # reparto que deja los renglones lo más parejos posible
        import itertools
        for cortes in itertools.combinations(range(1, len(pal)), n - 1):
            partes = [" ".join(pal[a:b]) for a, b in zip((0,) + cortes, cortes + (len(pal),))]
            largo = max(len(p) for p in partes)
            if peor is None or largo < peor:
                peor, mejor = largo, partes
        ops.append(mejor)
    return ops


def acomodar(d, ancho_caja=CAJA[2] - 2 * MARGEN_X, alto_caja=CAJA[3] - 2 * MARGEN_Y):
    """-> (renglones, tamaño final). El tamaño nunca pasa del que eligió el usuario ni deja salir el texto."""
    d = normalizar(d)
    if not d["texto"].strip():
        return [], d["tam"]
    mejor = None
    for lineas in opciones_renglones(d["texto"]):
        tam = d["tam"]
        while tam > 12:
            ancho = max(ancho_linea(l, tam, d["espaciado"]) for l in lineas)
            arriba, abajo = tinta(lineas, tam, d["interlineado"])
            if ancho <= ancho_caja and arriba <= alto_caja / 2 and abajo <= alto_caja / 2:
                break
            tam -= 2
        # gana la opción con la letra más grande; si empatan, la de menos renglones
        if mejor is None or tam > mejor[1] + 1:
            mejor = (lineas, tam)
    return mejor


def capa(d, W, H):
    """Imagen RGBA del tamaño del video con la barra (si va) y el título. None si no hay título."""
    d = normalizar(d)
    if not d["texto"].strip():
        return None
    base = Image.new("RGBA", (REF_W, REF_H), (0, 0, 0, 0))
    if d["barra"] and PLANTILLA.exists():
        base.alpha_composite(Image.open(PLANTILLA).convert("RGBA").resize((REF_W, REF_H)))
    lineas, tam = acomodar(d)
    f = fuente(tam)
    cap = alto_mayuscula(tam)
    top_h = f.getbbox("H")[1]
    paso = cap * PASO_RENGLON * d["interlineado"]
    alto = cap + (len(lineas) - 1) * paso
    y0 = CAJA[1] + (CAJA[3] - alto) / 2
    dib = ImageDraw.Draw(base)
    for k, l in enumerate(lineas):
        x = CAJA[0] + (CAJA[2] - ancho_linea(l, tam, d["espaciado"])) / 2
        y = y0 + k * paso - top_h
        # sombra suave como en las muestras
        for c in l:
            dib.text((x + 3, y + 4), c, font=f, fill=(60, 0, 0, 110))
            x += f.getlength(c) + d["espaciado"]
        x = CAJA[0] + (CAJA[2] - ancho_linea(l, tam, d["espaciado"])) / 2
        for c in l:
            dib.text((x, y), c, font=f, fill=(255, 255, 255, 255))
            x += f.getlength(c) + d["espaciado"]
    if (W, H) != (REF_W, REF_H):
        # se escala igual que la plantilla: a lo ancho del video, pegado arriba
        esc = W / REF_W
        base = base.resize((W, int(round(REF_H * esc))), Image.LANCZOS)
        lienzo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        lienzo.alpha_composite(base.crop((0, 0, W, min(H, base.height))))
        base = lienzo
    return base


def para_pegar(d, W, H):
    """(bgr, alpha, caja) para editor.pegar(), recortado a la zona con contenido."""
    im = capa(d, W, H)
    if im is None:
        return None
    a = np.array(im)
    ys, xs = np.where(a[..., 3] > 0)
    if not len(ys):
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rec = a[y0:y1, x0:x1]
    bgr = rec[..., [2, 1, 0]].astype(np.float32)
    alpha = (rec[..., 3:4].astype(np.float32) / 255.0)
    return bgr, alpha, (int(x0), int(y0), int(x1), int(y1))
