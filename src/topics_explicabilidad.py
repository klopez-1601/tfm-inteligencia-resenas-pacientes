"""
FASE 3 — Descubrimiento de temas y explicabilidad.

Cumple el punto b.i.iv de la guía ("Discusión de los resultados del modelo:
explicatividad/interpretabilidad") y aporta el componente no supervisado que
distingue este trabajo de un AutoML: no solo predecimos el sentimiento, sino
que explicamos DE QUÉ hablan los pacientes y POR QUÉ el modelo decide.

Contenido:
  1. Modelado de temas con NMF sobre TF-IDF (rápido y estable en CPU).
  2. Análisis de aspectos por léxico: eficacia, efectos secundarios,
     posología y coste, cruzado con la valoración media.
  3. Explicabilidad local con LIME sobre casos concretos.

Ejecutar con:  python src/topics_explicabilidad.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    DIR_MODELOS,
    RANDOM_STATE,
    etiquetar_barras,
    anadir_sentimiento,
    cargar_datos,
    guardar_figura,
    guardar_metricas,
    solo_etiquetados,
)

sns.set_theme(style="whitegrid")

N_TEMAS = 8
N_PALABRAS = 12

# Léxico de aspectos. Se define explícitamente (y se documenta en la memoria)
# porque el objetivo es un análisis accionable para farmacovigilancia, no una
# lista de temas genéricos.
ASPECTOS = {
    "Eficacia": ["work", "works", "worked", "effective", "helped", "help", "relief",
                 "improve", "improved", "better", "cured", "results"],
    "Efectos secundarios": ["side", "effects", "nausea", "headache", "dizzy", "dizziness",
                            "weight", "gain", "anxiety", "insomnia", "tired", "fatigue",
                            "pain", "sick", "vomiting", "rash", "depression"],
    "Posología y administración": ["dose", "dosage", "mg", "pill", "pills", "tablet",
                                   "injection", "daily", "twice", "morning", "night",
                                   "taper", "withdrawal"],
    "Coste y acceso": ["price", "cost", "expensive", "insurance", "cheap", "afford",
                       "pharmacy", "generic", "coupon"],
}


# --------------------------------------------------------------------------
# 1. Modelado de temas
# --------------------------------------------------------------------------
def modelar_temas(textos: pd.Series, n_temas: int = N_TEMAS) -> tuple[dict, np.ndarray]:
    """
    NMF sobre TF-IDF.

    Se elige NMF frente a LDA porque sobre TF-IDF produce temas más nítidos y
    entrena en una fracción del tiempo, algo relevante para que el proyecto sea
    reproducible en un portátil sin GPU.
    """
    vect = TfidfVectorizer(max_df=0.85, min_df=10, stop_words="english",
                           ngram_range=(1, 2), max_features=40_000,
                           strip_accents="unicode")
    X = vect.fit_transform(textos)

    nmf = NMF(n_components=n_temas, random_state=RANDOM_STATE, init="nndsvda",
              max_iter=400, beta_loss="frobenius")
    W = nmf.fit_transform(X)
    nombres = np.array(vect.get_feature_names_out())

    temas = {}
    for i, componente in enumerate(nmf.components_):
        top = componente.argsort()[-N_PALABRAS:][::-1]
        temas[f"Tema {i + 1}"] = [str(nombres[j]) for j in top]
    return temas, W


def fig_temas(temas: dict, W: np.ndarray, y: pd.Series) -> pd.DataFrame:
    """Relaciona cada tema con el sentimiento medio de las reseñas donde domina."""
    tema_dominante = W.argmax(axis=1)
    resumen = (pd.DataFrame({"tema": tema_dominante, "positivo": y.to_numpy()})
                 .groupby("tema")
                 .agg(n=("positivo", "size"), pct_positivo=("positivo", "mean")))
    resumen.index = [f"Tema {i + 1}" for i in resumen.index]
    resumen["palabras"] = [", ".join(temas[t][:6]) for t in resumen.index]
    resumen["pct_positivo"] = (resumen["pct_positivo"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    orden = resumen.sort_values("pct_positivo")
    colores = ["#c0392b" if v < 50 else "#27ae60" for v in orden["pct_positivo"]]
    ax.barh(range(len(orden)), orden["pct_positivo"], color=colores)
    ax.set_yticks(range(len(orden)))
    ax.set_yticklabels([f"{i}\n{p[:45]}" for i, p in zip(orden.index, orden["palabras"])],
                       fontsize=8)
    ax.axvline(50, color="grey", linestyle="--")
    etiquetar_barras(ax, horizontal=True, formato="{:.1f}%", tam=7)
    ax.set_xlabel("% de reseñas positivas en el tema")
    ax.set_title("Temas descubiertos y su polaridad")
    fig.tight_layout()
    guardar_figura(fig, "10_temas_nmf")
    plt.close(fig)
    return resumen


# --------------------------------------------------------------------------
# 2. Análisis de aspectos
# --------------------------------------------------------------------------
def analizar_aspectos(df: pd.DataFrame) -> pd.DataFrame:
    """Marca qué aspectos menciona cada reseña y cruza con la valoración."""
    texto = df["review"].str.lower()
    filas = []
    for aspecto, terminos in ASPECTOS.items():
        # Grupo NO capturador: evita el aviso de pandas sobre match groups
        patron = r"\b(?:" + "|".join(terminos) + r")\b"
        menciona = texto.str.contains(patron, regex=True, na=False)

        def _media(mascara):
            sub = df.loc[mascara, "rating"]
            return round(float(sub.mean()), 2) if len(sub) else None

        filas.append({
            "aspecto": aspecto,
            "n_menciones": int(menciona.sum()),
            "pct_resenas": round(100 * float(menciona.mean()), 2),
            "rating_medio_si_menciona": _media(menciona),
            "rating_medio_si_no_menciona": _media(~menciona),
        })
        df[f"aspecto_{aspecto}"] = menciona
    return pd.DataFrame(filas)


def fig_aspectos(tabla: pd.DataFrame, media_global: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    axes[0].barh(tabla["aspecto"], tabla["pct_resenas"], color=sns.color_palette("deep")[0])
    axes[0].set_xlabel("% de reseñas que lo mencionan")
    axes[0].set_title("Frecuencia de cada aspecto")
    etiquetar_barras(axes[0], horizontal=True, formato="{:.1f}%", tam=8)

    x = np.arange(len(tabla))
    alturas = tabla["rating_medio_si_menciona"].fillna(0)
    axes[1].barh(x, alturas, color=sns.color_palette("deep")[3])
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(tabla["aspecto"])
    # La referencia es la media del subconjunto etiquetado (el que se analiza
    # aquí), no la del corpus completo: son 7,14 frente a 6,99, y confundirlas
    # llevaría a comparar contra una base distinta.
    axes[1].axvline(media_global, color="grey", linestyle="--",
                    label=f"Media del conjunto etiquetado = {media_global:.2f}")
    axes[1].set_xlabel("Valoración media cuando se menciona")
    axes[1].set_title("Impacto del aspecto en la satisfacción")
    etiquetar_barras(axes[1], horizontal=True, formato="{:.2f}", tam=8)
    axes[1].legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    guardar_figura(fig, "11_aspectos")
    plt.close(fig)


def aspectos_por_condicion(df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    """
    Cruce accionable: en qué condiciones pesan más los efectos secundarios.
    Es la vista que da valor de farmacovigilancia al proyecto.
    """
    top = df["condition"].value_counts().head(top_n).index
    sub = df[df["condition"].isin(top)]
    tabla = (sub.groupby("condition")
                .agg(n=("rating", "size"),
                     rating_medio=("rating", "mean"),
                     pct_efectos=("aspecto_Efectos secundarios", "mean"),
                     pct_eficacia=("aspecto_Eficacia", "mean"))
                .round(3)
                .sort_values("pct_efectos", ascending=False))
    tabla["pct_efectos"] = (tabla["pct_efectos"] * 100).round(1)
    tabla["pct_eficacia"] = (tabla["pct_eficacia"] * 100).round(1)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.scatter(tabla["pct_efectos"], tabla["rating_medio"],
               s=tabla["n"] / 40, alpha=0.65, color=sns.color_palette("deep")[3])
    for cond, fila in tabla.iterrows():
        ax.annotate(str(cond)[:22], (fila["pct_efectos"], fila["rating_medio"]),
                    fontsize=8, alpha=0.85,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("% de reseñas que mencionan efectos secundarios")
    ax.set_ylabel("Valoración media")
    ax.set_title("Efectos secundarios frente a satisfacción, por condición\n"
                 "(tamaño = volumen de reseñas)")
    fig.tight_layout()
    guardar_figura(fig, "12_aspectos_por_condicion")
    plt.close(fig)
    return tabla


# --------------------------------------------------------------------------
# 3. Explicabilidad local (LIME)
# --------------------------------------------------------------------------
def explicar_casos(df: pd.DataFrame, n_casos: int = 3) -> list[dict]:
    """Explica predicciones concretas: qué palabras han pesado y cuánto."""
    ruta = DIR_MODELOS / "modelo_sentimiento.joblib"
    if not ruta.exists():
        print("      (aviso) No hay modelo entrenado. Ejecuta antes src/modelado.py")
        return []

    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        print("      (aviso) LIME no instalado: pip install lime")
        return []

    modelo = joblib.load(ruta)
    # Semilla fija: LIME estima por muestreo aleatorio y sin random_state los
    # pesos varían entre ejecuciones, rompiendo la reproducibilidad.
    explicador = LimeTextExplainer(class_names=["negativo", "positivo"],
                                   random_state=RANDOM_STATE)

    muestra = df.sample(n_casos, random_state=RANDOM_STATE)
    salidas = []
    for _, fila in muestra.iterrows():
        exp = explicador.explain_instance(
            fila["review"], modelo.predict_proba, num_features=10, num_samples=800)
        salidas.append({
            "farmaco": fila["drugName"],
            "condicion": str(fila["condition"]),
            "rating_real": int(fila["rating"]),
            "sentimiento_real": fila["sentimiento"],
            "probabilidad_positivo": round(
                float(modelo.predict_proba([fila["review"]])[0, 1]), 4),
            "extracto": fila["review"][:220] + ("..." if len(fila["review"]) > 220 else ""),
            "terminos_decisivos": [
                {"termino": t, "peso": round(float(p), 4)} for t, p in exp.as_list()],
        })
    return salidas


# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("FASE 3 — TEMAS Y EXPLICABILIDAD")
    print("=" * 70)

    print("\n[1/5] Cargando datos...")
    df = solo_etiquetados(anadir_sentimiento(cargar_datos()))
    print(f"      {len(df):,} reseñas")

    # El modelado de temas se hace sobre una muestra para acotar el tiempo
    # de ejecución en un portátil; la muestra es aleatoria y reproducible.
    muestra = df.sample(min(40_000, len(df)), random_state=RANDOM_STATE)

    print(f"\n[2/5] Modelando {N_TEMAS} temas con NMF sobre {len(muestra):,} reseñas...")
    temas, W = modelar_temas(muestra["review"])
    resumen_temas = fig_temas(temas, W, muestra["es_positivo"])
    for nombre, palabras in temas.items():
        print(f"      {nombre}: {', '.join(palabras[:6])}")

    print("\n[3/5] Analizando aspectos...")
    tabla_aspectos = analizar_aspectos(df)
    fig_aspectos(tabla_aspectos, float(df["rating"].mean()))
    print(tabla_aspectos.to_string(index=False))

    print("\n[4/5] Cruce aspectos x condición...")
    tabla_cond = aspectos_por_condicion(df)

    print("\n[5/5] Explicabilidad local con LIME...")
    casos = explicar_casos(df)
    for c in casos:
        print(f"      {c['farmaco']} (rating {c['rating_real']}) "
              f"-> P(positivo) = {c['probabilidad_positivo']:.3f}")

    guardar_metricas("temas_y_explicabilidad", {
        "n_temas": N_TEMAS,
        "temas": temas,
        "resumen_temas": resumen_temas.reset_index().to_dict("records"),
        "aspectos": tabla_aspectos.to_dict("records"),
        "aspectos_por_condicion": tabla_cond.reset_index().to_dict("records"),
        "lexico_aspectos": ASPECTOS,
        "casos_explicados": casos,
    })

    print("\n" + "=" * 70)
    print("FASE 3 COMPLETADA.")
    print("=" * 70)


if __name__ == "__main__":
    main()
