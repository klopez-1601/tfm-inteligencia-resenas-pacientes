"""
FASE 4 — Productivización del modelo.

La guía dice que "se valorará muy positivamente que este modelo pueda
productivizarse... que al modelo se le puedan pasar nuevos valores y el modelo
devuelva una predicción". Esta aplicación es exactamente eso: el analista pega
una reseña nueva y obtiene predicción, confianza, aspectos detectados y los
términos que han pesado en la decisión.

Ejecutar con:  streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from config import DIR_MODELOS, limpiar_texto  # noqa: E402
from topics_explicabilidad import ASPECTOS  # noqa: E402

st.set_page_config(page_title="Inteligencia de reseñas de pacientes",
                   page_icon="💊", layout="wide")

EJEMPLOS = {
    "— Escribir mi propia reseña —": "",
    "Reseña favorable": (
        "I have been taking this medication for three months and it has completely "
        "changed my life. The pain is finally under control and I can sleep again. "
        "Mild dry mouth at the beginning but it went away after two weeks."),
    "Reseña desfavorable": (
        "Worst experience ever. After only four days I had severe nausea, constant "
        "headaches and could not get out of bed. It did nothing for my symptoms and "
        "the side effects were unbearable. I stopped taking it immediately."),
    "Reseña ambigua": (
        "It works reasonably well for the pain, but the weight gain has been hard to "
        "deal with. Not sure if I will continue, I am weighing the pros and cons."),
}


@st.cache_resource
def cargar_modelo():
    ruta = DIR_MODELOS / "modelo_sentimiento.joblib"
    if not ruta.exists():
        return None
    return joblib.load(ruta)


def detectar_aspectos(texto: str) -> dict[str, list[str]]:
    """Detecta qué aspectos aparecen y con qué términos concretos."""
    minus = texto.lower()
    encontrados = {}
    for aspecto, terminos in ASPECTOS.items():
        presentes = [t for t in terminos if f" {t} " in f" {minus} "]
        if presentes:
            encontrados[aspecto] = presentes
    return encontrados


def explicar(modelo, texto: str, n: int = 10):
    """Explicación local con LIME; si no está instalado, se degrada con elegancia."""
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return None
    explicador = LimeTextExplainer(class_names=["negativo", "positivo"])
    exp = explicador.explain_instance(texto, modelo.predict_proba,
                                      num_features=n, num_samples=600)
    return exp.as_list()


# --------------------------------------------------------------------------
st.title("💊 Inteligencia de reseñas de pacientes")
st.caption("Trabajo Fin de Máster — Data Science, Big Data & Business Analytics (UCM)")

modelo = cargar_modelo()

if modelo is None:
    st.error(
        "No se encuentra el modelo entrenado.\n\n"
        "Ejecuta primero:  `python src/modelado.py`"
    )
    st.stop()

with st.sidebar:
    st.header("Sobre esta herramienta")
    st.markdown(
        "Clasifica automáticamente reseñas de pacientes y explica **por qué** "
        "toma cada decisión.\n\n"
        "**Casos de uso**\n"
        "- Farmacovigilancia: detección temprana de señales negativas\n"
        "- Priorización de reseñas que requieren revisión humana\n"
        "- Seguimiento de la experiencia del paciente por fármaco\n"
    )
    st.divider()
    umbral = st.slider(
        "Umbral de alerta", 0.0, 1.0, 0.35, 0.05,
        help="Si la probabilidad de sentimiento positivo cae por debajo de este "
             "valor, la reseña se marca para revisión.")
    st.divider()
    st.caption(
        "⚠️ Herramienta de apoyo al análisis agregado. "
        "No constituye consejo médico ni sustituye el criterio clínico.")

col_izq, col_der = st.columns([1, 1])

with col_izq:
    st.subheader("Reseña a analizar")
    eleccion = st.selectbox("Cargar un ejemplo", list(EJEMPLOS.keys()))
    texto = st.text_area("Texto de la reseña (en inglés, como el corpus original)",
                         value=EJEMPLOS[eleccion], height=230,
                         placeholder="Pega aquí la reseña del paciente...")
    analizar = st.button("Analizar reseña", type="primary", use_container_width=True)

with col_der:
    st.subheader("Resultado")

    if analizar and texto.strip():
        limpio = limpiar_texto(texto)
        prob_pos = float(modelo.predict_proba([limpio])[0, 1])
        etiqueta = "POSITIVO" if prob_pos >= 0.5 else "NEGATIVO"

        m1, m2 = st.columns(2)
        m1.metric("Sentimiento previsto", etiqueta)
        m2.metric("Probabilidad de positivo", f"{prob_pos:.1%}")
        st.progress(prob_pos)

        if prob_pos < umbral:
            st.error(f"🚩 **Alerta**: por debajo del umbral ({umbral:.0%}). "
                     "Reseña marcada para revisión.")
        elif prob_pos < 0.5:
            st.warning("Sentimiento negativo, sin superar el umbral de alerta.")
        else:
            st.success("Sentimiento positivo.")

        aspectos = detectar_aspectos(limpio)
        if aspectos:
            st.markdown("**Aspectos detectados**")
            for aspecto, terminos in aspectos.items():
                icono = "⚠️" if aspecto == "Efectos secundarios" else "•"
                st.markdown(f"{icono} **{aspecto}** — {', '.join(terminos[:6])}")
        else:
            st.caption("No se han detectado aspectos del léxico definido.")

    elif analizar:
        st.info("Introduce una reseña para analizarla.")
    else:
        st.info("Carga un ejemplo o escribe una reseña y pulsa **Analizar**.")

if analizar and texto.strip():
    st.divider()
    st.subheader("¿Por qué el modelo ha decidido esto?")
    with st.spinner("Calculando la explicación..."):
        pesos = explicar(modelo, limpiar_texto(texto))

    if pesos is None:
        st.warning("Instala LIME para ver la explicación:  `pip install lime`")
    else:
        df_pesos = (pd.DataFrame(pesos, columns=["Término", "Peso"])
                      .sort_values("Peso"))
        c1, c2 = st.columns([2, 1])
        with c1:
            st.bar_chart(df_pesos.set_index("Término")["Peso"], horizontal=True)
        with c2:
            st.markdown("**Empujan a negativo**")
            for _, f in df_pesos.head(5).iterrows():
                st.markdown(f"- `{f['Término']}` ({f['Peso']:.3f})")
            st.markdown("**Empujan a positivo**")
            for _, f in df_pesos.tail(5).iloc[::-1].iterrows():
                st.markdown(f"- `{f['Término']}` (+{f['Peso']:.3f})")
        st.caption(
            "Valores negativos empujan la predicción hacia *negativo* y los "
            "positivos hacia *positivo*. Explicación local calculada con LIME.")
