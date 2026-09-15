"""
FASE 4 — Productivización del modelo.

La guía pide que "al modelo se le puedan pasar nuevos valores y el modelo
devuelva una predicción". Esta aplicación lo hace de dos formas:

  · PERFIL DEL MEDICAMENTO (pantalla principal). El analista elige un fármaco y
    obtiene su radiografía completa: satisfacción, aspectos que preocupan a los
    pacientes, evolución temporal y —usando el modelo— las reseñas que deberían
    revisarse primero. Es la vista que da sentido al caso de farmacovigilancia.

  · ANALIZAR UNA RESEÑA. Recibe un texto nuevo y devuelve predicción, aspectos
    detectados y explicación. Demuestra el modelo funcionando sobre datos que
    nunca ha visto.

Ejecutar con:  streamlit run src/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
from config import DIR_MODELOS, anadir_sentimiento, cargar_datos, limpiar_texto  # noqa: E402
from topics_explicabilidad import ASPECTOS  # noqa: E402

st.set_page_config(page_title="Inteligencia de reseñas de pacientes",
                   page_icon="💊", layout="wide")

MIN_RESENAS = 30      # umbral por debajo del cual un perfil no es fiable
RATING_GLOBAL = 6.99  # media del corpus completo, para comparar


# --------------------------------------------------------------------------
# Carga (en caché: solo se paga la primera vez)
# --------------------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    ruta = DIR_MODELOS / "modelo_sentimiento.joblib"
    return joblib.load(ruta) if ruta.exists() else None


@st.cache_data(show_spinner="Cargando el corpus de reseñas...")
def cargar_corpus() -> pd.DataFrame | None:
    try:
        df = anadir_sentimiento(cargar_datos())
    except FileNotFoundError:
        return None
    texto = df["review"].str.lower()
    for aspecto, terminos in ASPECTOS.items():
        patron = r"\b(?:" + "|".join(terminos) + r")\b"
        df[f"asp_{aspecto}"] = texto.str.contains(patron, regex=True, na=False)
    return df


@st.cache_data(show_spinner=False)
def farmacos_disponibles(_df: pd.DataFrame) -> list[str]:
    conteo = _df["drugName"].value_counts()
    return sorted(conteo[conteo >= MIN_RESENAS].index.tolist())


@st.cache_data(show_spinner="Aplicando el modelo a las reseñas...")
def priorizar_con_modelo(textos: tuple[str, ...]) -> np.ndarray:
    """Devuelve P(positivo) para cada reseña. Es el modelo en producción."""
    modelo = cargar_modelo()
    return modelo.predict_proba(list(textos))[:, 1]


def detectar_aspectos(texto: str) -> dict[str, list[str]]:
    minus = texto.lower()
    return {a: [t for t in ts if f" {t} " in f" {minus} "]
            for a, ts in ASPECTOS.items()
            if any(f" {t} " in f" {minus} " for t in ts)}


def explicar(modelo, texto: str, n: int = 10):
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        return None
    # random_state fijo: LIME estima por muestreo aleatorio; sin semilla los
    # pesos cambian entre ejecuciones y las capturas no serían reproducibles.
    explicador = LimeTextExplainer(class_names=["negativo", "positivo"],
                                   random_state=42)
    return explicador.explain_instance(texto, modelo.predict_proba,
                                       num_features=n, num_samples=600).as_list()


# --------------------------------------------------------------------------
st.title("💊 Inteligencia de reseñas de pacientes")
st.caption("Trabajo Fin de Máster — Data Science, Big Data & Business Analytics (UCM)")

modelo = cargar_modelo()
if modelo is None:
    st.error("No se encuentra el modelo entrenado.\n\n"
             "Ejecuta primero:  `python src/modelado.py`")
    st.stop()

with st.sidebar:
    st.header("Sobre esta herramienta")
    st.markdown(
        "Convierte las opiniones de pacientes en información accionable.\n\n"
        "**Perfil del medicamento** — radiografía de un fármaco a partir de "
        "todas sus reseñas.\n\n"
        "**Analizar una reseña** — clasifica un texto nuevo y explica por qué.\n"
    )
    st.divider()
    umbral = st.slider(
        "Umbral de alerta", 0.0, 1.0, 0.35, 0.05,
        help="Una reseña cuya probabilidad de ser positiva caiga por debajo de "
             "este valor se marca para revisión.")
    st.divider()
    st.caption("⚠️ Herramienta de apoyo al análisis agregado. No constituye "
               "consejo médico ni sustituye el criterio clínico.")

tab_perfil, tab_resena = st.tabs(["🔬 Perfil del medicamento",
                                  "📝 Analizar una reseña"])

# ==========================================================================
# PESTAÑA 1 — PERFIL DEL MEDICAMENTO
# ==========================================================================
with tab_perfil:
    corpus = cargar_corpus()

    if corpus is None:
        st.warning(
            "No se encuentran los datos en `data/`.\n\n"
            "Esta vista necesita el corpus completo. Descárgalo siguiendo el "
            "paso 1 del README. Mientras tanto, la pestaña **Analizar una "
            "reseña** funciona sin él.")
    else:
        lista = farmacos_disponibles(corpus)
        farmaco = st.selectbox(
            f"Medicamento ({len(lista):,} con al menos {MIN_RESENAS} reseñas)",
            lista,
            index=lista.index("Levonorgestrel") if "Levonorgestrel" in lista else 0,
            help="El perfil se actualiza automáticamente al cambiar de medicamento.")

        sub = corpus[corpus["drugName"] == farmaco].copy()
        lab = sub.dropna(subset=["sentimiento"])
        pct_neg = float((lab["sentimiento"] == "negativo").mean()) if len(lab) else 0.0
        media = float(sub["rating"].mean())

        # ---- Cifras de cabecera ----
        st.subheader(f"Perfil de {farmaco}")
        m = st.columns(4)
        m[0].metric("Reseñas", f"{len(sub):,}")
        m[1].metric("Valoración media", f"{media:.2f} / 10",
                    delta=f"{media - RATING_GLOBAL:+.2f} vs. media global")
        m[2].metric("Opiniones negativas", f"{pct_neg:.1%}",
                    delta=f"{pct_neg - 0.273:+.1%} vs. global", delta_color="inverse")
        m[3].metric("Periodo",
                    f"{sub['date'].min():%Y} – {sub['date'].max():%Y}")

        if media < 5:
            st.error("🚩 Satisfacción muy por debajo de la media del corpus. "
                     "Candidato prioritario a revisión.")
        elif media < RATING_GLOBAL:
            st.warning("Satisfacción por debajo de la media del corpus.")
        else:
            st.success("Satisfacción por encima de la media del corpus.")

        izq, der = st.columns(2)

        # ---- Distribución de valoraciones ----
        with izq:
            st.markdown("**Distribución de valoraciones**")
            dist = (sub["rating"].value_counts()
                    .reindex(range(1, 11), fill_value=0).sort_index())
            st.bar_chart(dist, color="#4c78a8")

        # ---- Aspectos ----
        with der:
            st.markdown("**¿De qué hablan los pacientes?**")
            filas = []
            for aspecto in ASPECTOS:
                col = f"asp_{aspecto}"
                menciona = sub[col]
                if menciona.sum() == 0:
                    continue
                filas.append({
                    "Aspecto": aspecto,
                    "% reseñas": round(100 * float(menciona.mean()), 1),
                    "Valoración": round(float(sub.loc[menciona, "rating"].mean()), 2),
                })
            tabla = pd.DataFrame(filas).sort_values("% reseñas", ascending=False)
            st.dataframe(tabla, hide_index=True, use_container_width=True)
            st.caption("«Valoración» es la media entre quienes mencionan ese "
                       "aspecto. Si es baja, ese aspecto arrastra la "
                       "satisfacción hacia abajo.")

        # ---- Condiciones y evolución ----
        izq2, der2 = st.columns(2)
        with izq2:
            st.markdown("**Condiciones tratadas**")
            cond = sub["condition"].value_counts().head(5)
            st.dataframe(
                pd.DataFrame({"Condición": cond.index.astype(str),
                              "Reseñas": cond.values}),
                hide_index=True, use_container_width=True)
        with der2:
            st.markdown("**Evolución de la valoración media**")
            serie = (sub.set_index("date").sort_index()
                        .resample("YE")["rating"].mean().dropna())
            if len(serie) > 1:
                serie.index = serie.index.year
                st.line_chart(serie, color="#e45756")
            else:
                st.caption("Sin histórico suficiente.")

        # ---- El modelo en acción ----
        st.divider()
        st.markdown("#### Reseñas priorizadas por el modelo")
        st.caption(
            "El modelo puntúa cada reseña de este medicamento y las ordena "
            "por probabilidad de ser negativas. Así se revisa primero lo "
            "más crítico, sin depender de la valoración que el paciente "
            "declaró.")

        muestra = sub if len(sub) <= 1500 else sub.sample(1500, random_state=42)
        probs = priorizar_con_modelo(tuple(muestra["review"].tolist()))
        muestra = muestra.assign(p_pos=probs).sort_values("p_pos")

        n_alerta = int((muestra["p_pos"] < umbral).sum())
        st.markdown(
            f"**{n_alerta:,}** de **{len(muestra):,}** reseñas analizadas "
            f"caen bajo el umbral de alerta ({umbral:.0%}).")

        for _, fila in muestra.head(3).iterrows():
            with st.expander(
                    f"P(positivo) = {fila['p_pos']:.1%}  ·  "
                    f"valoración declarada: {int(fila['rating'])}/10  ·  "
                    f"{str(fila['condition'])[:40]}"):
                st.write(fila["review"][:600] +
                         ("..." if len(fila["review"]) > 600 else ""))
                asp = detectar_aspectos(fila["review"])
                if asp:
                    st.caption("Aspectos: " + " · ".join(
                        f"**{a}** ({', '.join(t[:3])})" for a, t in asp.items()))

# ==========================================================================
# PESTAÑA 2 — ANALIZAR UNA RESEÑA
# ==========================================================================
EJEMPLOS = {
    "— Escribir mi propia reseña —": "",
    "Reseña favorable": (
        "I have been taking this medication twice a day for three months and it "
        "works better than anything I tried before. It gave me real relief from "
        "the pain and I can finally sleep again. Mild dry mouth at the beginning "
        "but it went away after two weeks."),
    "Reseña desfavorable": (
        "Worst experience ever. After only four days I had severe nausea, constant "
        "headaches and could not get out of bed. It did nothing for my symptoms and "
        "the side effects were unbearable. I stopped taking it immediately."),
    # Caso mixto: el modelo lo clasifica como positivo pese a que la paciente se
    # plantea abandonar. Ilustra la limitación descrita en la memoria.
    "Reseña mixta (caso difícil)": (
        "It works reasonably well for the pain, but the weight gain has been hard to "
        "deal with. Not sure if I will continue, I am weighing the pros and cons."),
}

with tab_resena:
    st.caption(
        "Clasifica un texto que el modelo nunca ha visto. Es la demostración "
        "directa de la productivización: entra una reseña nueva, sale una "
        "predicción explicada.")

    col_izq, col_der = st.columns([1, 1])
    with col_izq:
        eleccion = st.selectbox("Cargar un ejemplo", list(EJEMPLOS.keys()))
        texto = st.text_area("Texto de la reseña (en inglés, como el corpus original)",
                             value=EJEMPLOS[eleccion], height=230,
                             placeholder="Pega aquí la reseña del paciente...")
        analizar = st.button("Analizar reseña", type="primary",
                             use_container_width=True)

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
            # Se separan por SIGNO, no por posición: en una reseña muy
            # polarizada todos los pesos comparten signo y, ordenando por
            # posición, aparecerían términos bajo el encabezado equivocado.
            hacia_neg = df_pesos[df_pesos["Peso"] < 0]
            hacia_pos = df_pesos[df_pesos["Peso"] > 0].sort_values("Peso", ascending=False)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.bar_chart(df_pesos.set_index("Término")["Peso"], horizontal=True)
            with c2:
                if len(hacia_neg):
                    st.markdown("**Empujan a negativo**")
                    for _, f in hacia_neg.head(5).iterrows():
                        st.markdown(f"- `{f['Término']}` ({f['Peso']:+.3f})")
                if len(hacia_pos):
                    st.markdown("**Empujan a positivo**")
                    for _, f in hacia_pos.head(5).iterrows():
                        st.markdown(f"- `{f['Término']}` ({f['Peso']:+.3f})")
            st.caption("Valores negativos empujan la predicción hacia *negativo* "
                       "y los positivos hacia *positivo*. Explicación local "
                       "calculada con LIME.")
