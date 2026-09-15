"""
Configuración central del TFM: rutas, constantes y carga de datos.

Todos los scripts importan de aquí para garantizar que trabajan sobre los
mismos datos, con las mismas semillas y los mismos criterios de negocio.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Rutas del proyecto
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_DATOS = RAIZ / "data"
DIR_MODELOS = RAIZ / "modelos"
DIR_RESULTADOS = RAIZ / "resultados"
DIR_FIGURAS = DIR_RESULTADOS / "figuras"
DIR_METRICAS = DIR_RESULTADOS / "metricas"

for _d in (DIR_MODELOS, DIR_FIGURAS, DIR_METRICAS):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Constantes de negocio y de experimentación
# --------------------------------------------------------------------------
RANDOM_STATE = 42

# Regla de negocio para convertir la valoración 1-10 en sentimiento.
# Se excluye la banda neutra (5-6) para que las dos clases representen
# opiniones inequívocas; la decisión se justifica en la memoria.
UMBRAL_POSITIVO = 7   # rating >= 7  -> positivo
UMBRAL_NEGATIVO = 4   # rating <= 4  -> negativo

COLUMNAS = ["uniqueID", "drugName", "condition", "review", "rating", "date", "usefulCount"]


# --------------------------------------------------------------------------
# Carga de datos
# --------------------------------------------------------------------------
def _buscar_fichero(patrones: list[str]) -> Path:
    """Localiza el fichero de datos aceptando .csv o .tsv indistintamente."""
    for patron in patrones:
        encontrados = sorted(DIR_DATOS.glob(patron))
        if encontrados:
            return encontrados[0]
    raise FileNotFoundError(
        f"No se encuentra ninguno de {patrones} en {DIR_DATOS}.\n"
        "Descarga el dataset y déjalo en la carpeta data/ (ver README, paso 1)."
    )


def _leer(ruta: Path) -> pd.DataFrame:
    """Lee CSV o TSV detectando el separador por la extensión."""
    sep = "\t" if ruta.suffix.lower() == ".tsv" else ","
    df = pd.read_csv(ruta, sep=sep)
    # La primera columna es un identificador sin nombre en el fichero original
    if df.columns[0].startswith("Unnamed"):
        df = df.rename(columns={df.columns[0]: "uniqueID"})
    return df


def limpiar_texto(texto: str) -> str:
    """
    Limpieza mínima e imprescindible del texto.

    El fichero original codifica las comillas como entidades HTML (&#039;),
    algo que rompe la tokenización si no se corrige. Esta función NO elimina
    stopwords ni aplica stemming: eso corresponde al vectorizador, de modo que
    el preprocesado queda dentro del Pipeline y no contamina el conjunto de test.
    """
    if not isinstance(texto, str):
        return ""
    texto = html.unescape(html.unescape(texto))  # doble: el fichero viene doblemente escapado
    texto = texto.strip().strip('"')
    texto = re.sub(r"\s+", " ", texto)
    return texto


def cargar_datos(deduplicar: bool = True) -> pd.DataFrame:
    """
    Carga train + test, los une y aplica limpieza básica.

    Se unen deliberadamente para poder rehacer la partición de forma
    estratificada y controlada. Además se eliminan duplicados exactos de
    (reseña, fármaco, condición): el dataset original contiene reseñas
    repetidas que, de no tratarse, provocarían fuga de información entre
    train y test e inflarían artificialmente las métricas.
    """
    ruta_train = _buscar_fichero(["drugsComTrain_raw.*", "*Train*.csv", "*Train*.tsv"])
    ruta_test = _buscar_fichero(["drugsComTest_raw.*", "*Test*.csv", "*Test*.tsv"])

    df = pd.concat([_leer(ruta_train), _leer(ruta_test)], ignore_index=True)

    df["review"] = df["review"].map(limpiar_texto)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")
    df["condition"] = df["condition"].astype("string")

    # Las filas cuya "condition" es basura del scraping original
    # (p. ej. "3</span> users found this comment helpful.")
    basura = df["condition"].str.contains("</span>", na=False)
    df = df.loc[~basura].copy()

    df = df.dropna(subset=["review", "rating"])
    df = df.loc[df["review"].str.len() > 0].copy()

    n_antes = len(df)
    if deduplicar:
        df = df.drop_duplicates(subset=["review", "drugName", "condition"]).copy()
    df.attrs["duplicados_eliminados"] = n_antes - len(df)

    return df.reset_index(drop=True)


def anadir_sentimiento(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la etiqueta de sentimiento y descarta la banda neutra."""
    df = df.copy()
    df["sentimiento"] = pd.NA
    df.loc[df["rating"] >= UMBRAL_POSITIVO, "sentimiento"] = "positivo"
    df.loc[df["rating"] <= UMBRAL_NEGATIVO, "sentimiento"] = "negativo"
    df["es_positivo"] = (df["sentimiento"] == "positivo").astype("int8")
    return df


def solo_etiquetados(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve únicamente las filas con sentimiento inequívoco."""
    return df.dropna(subset=["sentimiento"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Utilidades de resultados
# --------------------------------------------------------------------------
def guardar_metricas(nombre: str, datos: dict) -> Path:
    """
    Guarda métricas en JSON.

    Todas las cifras que aparecen en la memoria salen de estos ficheros,
    de forma que el documento es trazable frente a una ejecución real.
    """
    ruta = DIR_METRICAS / f"{nombre}.json"
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(datos, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> métricas guardadas en {ruta.relative_to(RAIZ)}")
    return ruta


def guardar_figura(fig, nombre: str) -> Path:
    """Guarda una figura en alta resolución, lista para insertar en LaTeX."""
    ruta = DIR_FIGURAS / f"{nombre}.png"
    fig.savefig(ruta, dpi=200, bbox_inches="tight")
    print(f"  -> figura guardada en {ruta.relative_to(RAIZ)}")
    return ruta
