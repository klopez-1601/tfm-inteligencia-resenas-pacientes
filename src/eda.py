"""
FASE 1 — Análisis exploratorio y descriptivo (EDA).

Cumple el punto b.i.i de la guía: "Crear un análisis descriptivo del conjunto
(gráfico en lo posible)".

Genera las figuras y las métricas que alimentan el capítulo 4 de la memoria.
Ejecutar con:  python src/eda.py
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # backend sin ventana: permite ejecutar en cualquier equipo

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from config import (
    RANDOM_STATE,
    anadir_sentimiento,
    cargar_datos,
    guardar_figura,
    guardar_metricas,
)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.figsize"] = (10, 5)


def resumen_general(df: pd.DataFrame) -> dict:
    """Cifras de cabecera del conjunto de datos."""
    resumen = {
        "n_reseñas": int(len(df)),
        "duplicados_eliminados": int(df.attrs.get("duplicados_eliminados", 0)),
        "n_farmacos": int(df["drugName"].nunique()),
        "n_condiciones": int(df["condition"].nunique()),
        "rango_fechas": [str(df["date"].min().date()), str(df["date"].max().date())],
        "rating_medio": round(float(df["rating"].mean()), 3),
        "rating_mediana": float(df["rating"].median()),
        "rating_desv_tipica": round(float(df["rating"].std()), 3),
        "longitud_media_caracteres": round(float(df["review"].str.len().mean()), 1),
        "longitud_media_palabras": round(float(df["review"].str.split().str.len().mean()), 1),
        "valores_ausentes": {c: int(df[c].isna().sum()) for c in df.columns},
    }

    # Reparto de clases tras aplicar la regla de negocio
    df_s = anadir_sentimiento(df)
    reparto = df_s["sentimiento"].value_counts(dropna=False)
    resumen["reparto_sentimiento"] = {str(k): int(v) for k, v in reparto.items()}
    n_etiquetadas = int(df_s["sentimiento"].notna().sum())
    resumen["n_etiquetadas"] = n_etiquetadas
    resumen["pct_positivas_sobre_etiquetadas"] = round(
        100 * float((df_s["sentimiento"] == "positivo").sum()) / max(n_etiquetadas, 1), 2
    )
    return resumen


def fig_distribucion_ratings(df: pd.DataFrame) -> None:
    """La distribución de ratings es fuertemente bimodal: hallazgo clave."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    conteo = df["rating"].value_counts().sort_index()
    axes[0].bar(conteo.index, conteo.values, color=sns.color_palette("deep")[0])
    axes[0].set_title("Distribución de valoraciones (1-10)")
    axes[0].set_xlabel("Valoración")
    axes[0].set_ylabel("Nº de reseñas")
    axes[0].set_xticks(range(1, 11))

    df_s = anadir_sentimiento(df)
    reparto = df_s["sentimiento"].value_counts()
    axes[1].bar(reparto.index.astype(str), reparto.values,
                color=[sns.color_palette("deep")[2], sns.color_palette("deep")[3]])
    axes[1].set_title("Reparto por sentimiento (excluida la banda neutra 5-6)")
    axes[1].set_ylabel("Nº de reseñas")

    fig.tight_layout()
    guardar_figura(fig, "01_distribucion_ratings")
    plt.close(fig)


def fig_top_condiciones_farmacos(df: pd.DataFrame, top: int = 15) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    df["condition"].value_counts().head(top).sort_values().plot(
        kind="barh", ax=axes[0], color=sns.color_palette("deep")[0])
    axes[0].set_title(f"Top {top} condiciones por volumen de reseñas")
    axes[0].set_xlabel("Nº de reseñas")

    df["drugName"].value_counts().head(top).sort_values().plot(
        kind="barh", ax=axes[1], color=sns.color_palette("deep")[1])
    axes[1].set_title(f"Top {top} fármacos por volumen de reseñas")
    axes[1].set_xlabel("Nº de reseñas")

    fig.tight_layout()
    guardar_figura(fig, "02_top_condiciones_farmacos")
    plt.close(fig)


def fig_satisfaccion_por_condicion(df: pd.DataFrame, min_reseñas: int = 500) -> pd.DataFrame:
    """
    Valoración media por condición: la vista más accionable para negocio.
    Se filtran las condiciones con poco volumen para evitar medias inestables.
    """
    agg = (df.groupby("condition")
             .agg(n=("rating", "size"), media=("rating", "mean"))
             .query("n >= @min_reseñas")
             .sort_values("media"))

    peores, mejores = agg.head(10), agg.tail(10)
    combinado = pd.concat([peores, mejores])

    fig, ax = plt.subplots(figsize=(10, 7))
    colores = ["#c0392b"] * len(peores) + ["#27ae60"] * len(mejores)
    ax.barh(combinado.index, combinado["media"], color=colores)
    ax.set_xlabel("Valoración media (1-10)")
    ax.set_title(f"Condiciones con menor y mayor satisfacción (≥{min_reseñas} reseñas)")
    ax.axvline(df["rating"].mean(), color="grey", linestyle="--",
               label=f"Media global = {df['rating'].mean():.2f}")
    ax.legend()

    fig.tight_layout()
    guardar_figura(fig, "03_satisfaccion_por_condicion")
    plt.close(fig)
    return agg


def fig_longitud_vs_sentimiento(df: pd.DataFrame) -> dict:
    """
    ¿Las reseñas negativas son más largas? Hipótesis contrastada
    formalmente (Mann-Whitney), no solo de forma visual.
    """
    df_s = anadir_sentimiento(df).dropna(subset=["sentimiento"]).copy()
    df_s["n_palabras"] = df_s["review"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df_s, x="sentimiento", y="n_palabras", ax=ax, showfliers=False)
    ax.set_title("Longitud de la reseña según el sentimiento")
    ax.set_xlabel("Sentimiento")
    ax.set_ylabel("Nº de palabras")
    fig.tight_layout()
    guardar_figura(fig, "04_longitud_vs_sentimiento")
    plt.close(fig)

    pos = df_s.loc[df_s["sentimiento"] == "positivo", "n_palabras"]
    neg = df_s.loc[df_s["sentimiento"] == "negativo", "n_palabras"]
    u, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")

    return {
        "prueba": "Mann-Whitney U (dos colas)",
        "hipotesis_nula": "La longitud de la reseña no difiere entre positivas y negativas",
        "mediana_palabras_positivas": float(pos.median()),
        "mediana_palabras_negativas": float(neg.median()),
        "estadistico_U": float(u),
        "p_valor": float(p),
        "significativo_al_5pct": bool(p < 0.05),
    }


def fig_evolucion_temporal(df: pd.DataFrame) -> None:
    serie = (df.set_index("date")
               .resample("QE")
               .agg(n=("rating", "size"), media=("rating", "mean"))
               .dropna())

    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(serie.index, serie["n"], color=sns.color_palette("deep")[0])
    ax1.set_ylabel("Nº de reseñas por trimestre", color=sns.color_palette("deep")[0])
    ax1.set_xlabel("Fecha")

    ax2 = ax1.twinx()
    ax2.plot(serie.index, serie["media"], color=sns.color_palette("deep")[3])
    ax2.set_ylabel("Valoración media", color=sns.color_palette("deep")[3])
    ax2.grid(False)

    ax1.set_title("Evolución del volumen y de la valoración media")
    fig.tight_layout()
    guardar_figura(fig, "05_evolucion_temporal")
    plt.close(fig)


def correlacion_utilidad(df: pd.DataFrame) -> dict:
    """¿Las reseñas extremas se perciben como más útiles?"""
    rho, p = stats.spearmanr(df["rating"], df["usefulCount"])
    df_tmp = df.assign(extremo=df["rating"].isin([1, 2, 9, 10]))
    return {
        "prueba": "Correlación de Spearman entre valoración y usefulCount",
        "rho": round(float(rho), 4),
        "p_valor": float(p),
        "utilidad_media_resenas_extremas": round(
            float(df_tmp.loc[df_tmp["extremo"], "usefulCount"].mean()), 2),
        "utilidad_media_resenas_moderadas": round(
            float(df_tmp.loc[~df_tmp["extremo"], "usefulCount"].mean()), 2),
    }


def main() -> None:
    print("=" * 70)
    print("FASE 1 — ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 70)

    print("\n[1/7] Cargando datos...")
    df = cargar_datos()
    print(f"      {len(df):,} reseñas tras limpieza "
          f"({df.attrs.get('duplicados_eliminados', 0):,} duplicados eliminados)")

    print("\n[2/7] Resumen general...")
    resumen = resumen_general(df)
    for k in ("n_reseñas", "n_farmacos", "n_condiciones", "rating_medio", "n_etiquetadas"):
        print(f"      {k}: {resumen[k]}")

    print("\n[3/7] Distribución de valoraciones...")
    fig_distribucion_ratings(df)

    print("\n[4/7] Top condiciones y fármacos...")
    fig_top_condiciones_farmacos(df)

    print("\n[5/7] Satisfacción por condición...")
    agg = fig_satisfaccion_por_condicion(df)
    resumen["condiciones_peor_valoradas"] = (
        agg.head(10).round(3).reset_index().to_dict("records"))
    resumen["condiciones_mejor_valoradas"] = (
        agg.tail(10).round(3).reset_index().to_dict("records"))

    print("\n[6/7] Longitud vs sentimiento y utilidad...")
    resumen["contraste_longitud"] = fig_longitud_vs_sentimiento(df)
    resumen["correlacion_utilidad"] = correlacion_utilidad(df)

    print("\n[7/7] Evolución temporal...")
    fig_evolucion_temporal(df)

    guardar_metricas("eda_resumen", resumen)

    print("\n" + "=" * 70)
    print("EDA COMPLETADO. Revisa resultados/figuras/ y resultados/metricas/")
    print("=" * 70)


if __name__ == "__main__":
    main()
