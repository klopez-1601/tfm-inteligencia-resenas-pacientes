"""
FASE 2 — Preprocesado, modelización y comparativa.

Cumple los puntos b.i.ii y b.i.iii de la guía: transformaciones relevantes y
"crear modelos de predicción utilizando diferentes técnicas de modelización
justificando su uso, determinando el nivel de precisión y detallando las
bondades y debilidades de cada técnica".

Decisiones de diseño que van más allá de un AutoML y se justifican en la memoria:
  1. Partición agrupada por fármaco (GroupShuffleSplit), para que el modelo se
     evalúe sobre fármacos que no ha visto en entrenamiento (evaluación honesta).
     No es estratificada: agrupar por fármaco y estratificar a la vez no es
     posible con este splitter, y se prioriza evitar la fuga de información.
  2. El vectorizador se ajusta UNA sola vez y solo sobre el conjunto de
     entrenamiento; nunca ve el test, de modo que no hay fuga de información.
     El modelo ganador se reempaqueta después en un Pipeline completo para que
     la aplicación pueda recibir texto crudo.
  3. Se emplean n-gramas de palabra (unigramas y bigramas). Los bigramas
     capturan negaciones y expresiones compuestas ("stopped taking",
     "weight gain") que un unigrama pierde.
  4. Se reporta un baseline trivial para dimensionar la mejora real.

Ejecutar con:  python src/modelado.py
"""

from __future__ import annotations

import argparse
import time
import warnings

import matplotlib
matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

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

warnings.filterwarnings("ignore", category=UserWarning)
sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
# Partición
# --------------------------------------------------------------------------
def particion_por_grupo(df: pd.DataFrame, test_size: float = 0.2):
    """
    Partición agrupada por fármaco.

    Si un mismo fármaco aparece en train y test, el modelo puede memorizar su
    vocabulario específico y las métricas salen infladas. Agrupando por
    'drugName' evaluamos la capacidad real de generalizar a fármacos nuevos,
    que es el escenario de uso en producción.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=RANDOM_STATE)
    idx_train, idx_test = next(splitter.split(df, groups=df["drugName"]))
    return df.iloc[idx_train].copy(), df.iloc[idx_test].copy()


# --------------------------------------------------------------------------
# Definición de modelos
# --------------------------------------------------------------------------
# La lista de stopwords de scikit-learn descarta las nueve negaciones del
# inglés ("not", "no", "never"...). En análisis de sentimiento eso destruye
# información esencial: sin "not", las frases "it did work" y "it did not work"
# se convierten en exactamente los mismos rasgos. Se conservan explícitamente.
NEGACIONES = {"not", "no", "never", "nor", "none", "cannot",
              "nothing", "nowhere", "neither", "without"}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - NEGACIONES)


def construir_vectorizador() -> TfidfVectorizer:
    return TfidfVectorizer(
        sublinear_tf=True,      # amortigua el efecto de palabras muy repetidas
        min_df=5,               # descarta ruido y erratas irrepetibles
        max_df=0.9,             # descarta términos omnipresentes
        ngram_range=(1, 2),     # unigramas + bigramas ("not work", "weight gain")
        strip_accents="unicode",
        stop_words=STOP_WORDS,  # inglés estándar MENOS las negaciones
        max_features=200_000,
    )


def catalogo_modelos() -> dict[str, tuple]:
    """
    Cada entrada: (pipeline completo, justificación para la memoria).

    Se devuelven pipelines enteros —y no solo el clasificador— porque
    Random Forest necesita una representación distinta: sobre una matriz
    TF-IDF de cientos de miles de columnas dispersas su entrenamiento es
    inviable y su rendimiento pobre. Se le antepone por ello una reducción
    de dimensionalidad (TruncatedSVD, es decir, Análisis Semántico Latente),
    que es la práctica habitual para combinar modelos de árboles con texto.
    """
    vec = construir_vectorizador

    return {
        "Baseline (clase mayoritaria)": (
            Pipeline([("tfidf", vec()),
                      ("clf", DummyClassifier(strategy="most_frequent",
                                              random_state=RANDOM_STATE))]),
            "Referencia mínima. Con clases desbalanceadas su F1 ya es alto, "
            "lo que obliga a mirar ROC-AUC y F1-macro, no solo accuracy.",
        ),
        "Naive Bayes (Complement)": (
            Pipeline([("tfidf", vec()), ("clf", ComplementNB(alpha=0.3))]),
            "Muy rápido y sólido en texto; la variante Complement corrige el "
            "sesgo del NB multinomial cuando las clases están desbalanceadas. "
            "Debilidad: asume independencia entre términos.",
        ),
        "Regresión Logística": (
            Pipeline([("tfidf", vec()),
                      ("clf", LogisticRegression(solver="liblinear", C=4.0,
                                                 class_weight="balanced",
                                                 max_iter=1000,
                                                 random_state=RANDOM_STATE))]),
            "Lineal, probabilístico e interpretable: sus coeficientes permiten "
            "explicar la predicción a un perfil de negocio. Se usa el solver "
            "liblinear, muy eficiente en problemas binarios y dispersos.",
        ),
        "SVM lineal": (
            Pipeline([("tfidf", vec()),
                      ("clf", CalibratedClassifierCV(
                          LinearSVC(C=0.5, class_weight="balanced",
                                    random_state=RANDOM_STATE),
                          cv=3, method="sigmoid"))]),
            "Referencia clásica en clasificación de texto de alta dimensión. "
            "Se calibra para obtener probabilidades, al coste de triplicar el "
            "tiempo de entrenamiento.",
        ),
        "Random Forest (sobre LSA)": (
            Pipeline([("tfidf", vec()),
                      ("svd", TruncatedSVD(n_components=150,
                                           random_state=RANDOM_STATE)),
                      ("clf", RandomForestClassifier(
                          n_estimators=40, max_depth=14, min_samples_leaf=5,
                          n_jobs=-1, class_weight="balanced_subsample",
                          random_state=RANDOM_STATE))]),
            "No lineal y basado en árboles: comprueba si existen interacciones "
            "que los modelos lineales no capturan. Requiere reducir antes la "
            "dimensionalidad con SVD, lo que sacrifica interpretabilidad. Se "
            "limita la profundidad a 14 niveles: sin ese límite los árboles "
            "crecen hasta hacer el entrenamiento inviable (más de 5 minutos) "
            "sin mejora apreciable del resultado.",
        ),
    }


# --------------------------------------------------------------------------
# Evaluación
# --------------------------------------------------------------------------
def evaluar(nombre, pipe, X_test, y_test, tiempo_entreno) -> dict:
    y_pred = pipe.predict(X_test)

    if hasattr(pipe, "predict_proba"):
        y_score = pipe.predict_proba(X_test)[:, 1]
    elif hasattr(pipe, "decision_function"):
        y_score = pipe.decision_function(X_test)
    else:
        y_score = y_pred

    try:
        auc = float(roc_auc_score(y_test, y_score))
    except ValueError:
        auc = float("nan")

    return {
        "modelo": nombre,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "roc_auc": round(auc, 4),
        "segundos_entrenamiento": round(tiempo_entreno, 1),
        "informe": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "_y_score": y_score,
        "_y_pred": y_pred,
    }


def fig_comparativa(resultados: list[dict]) -> None:
    df = pd.DataFrame([{k: r[k] for k in ("modelo", "accuracy", "f1", "roc_auc")}
                       for r in resultados])
    df = df.set_index("modelo").sort_values("f1")

    fig, ax = plt.subplots(figsize=(11, 5))
    df.plot(kind="barh", ax=ax)
    ax.set_xlabel("Puntuación")
    ax.set_title("Comparativa de modelos sobre el conjunto de test")
    ax.set_xlim(0, 1)
    etiquetar_barras(ax, horizontal=True, formato="{:.3f}", tam=6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    guardar_figura(fig, "06_comparativa_modelos")
    plt.close(fig)


def fig_roc(resultados: list[dict], y_test) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for r in resultados:
        if r["modelo"].startswith("Baseline") or np.isnan(r["roc_auc"]):
            continue
        fpr, tpr, _ = roc_curve(y_test, r["_y_score"])
        ax.plot(fpr, tpr, label=f"{r['modelo']} (AUC = {r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curvas ROC")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    guardar_figura(fig, "07_curvas_roc")
    plt.close(fig)


def fig_matriz_confusion(mejor: dict, y_test) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, mejor["_y_pred"], display_labels=["Negativo", "Positivo"],
        cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"Matriz de confusión — {mejor['modelo']}")
    fig.tight_layout()
    guardar_figura(fig, "08_matriz_confusion")
    plt.close(fig)


def terminos_influyentes_de(vect, clf, n: int = 25) -> dict:
    """Versión que recibe el vectorizador y el clasificador por separado."""
    try:
        if not hasattr(clf, "coef_"):
            return {}
        nombres = np.array(vect.get_feature_names_out())
        coefs = np.asarray(clf.coef_)[0]
        if len(coefs) != len(nombres):
            return {}
        orden = np.argsort(coefs)
        return {
            "hacia_negativo": [
                {"termino": str(nombres[i]), "peso": round(float(coefs[i]), 4)}
                for i in orden[:n]],
            "hacia_positivo": [
                {"termino": str(nombres[i]), "peso": round(float(coefs[i]), 4)}
                for i in orden[-n:][::-1]],
        }
    except Exception:
        return {}


def terminos_influyentes(pipe, n: int = 25) -> dict:
    """
    Explicabilidad global a partir de los coeficientes del modelo.

    Se calcula siempre sobre un modelo lineal (Regresión Logística). Si el
    modelo ganador fuese Naive Bayes o Random Forest —que no exponen
    coeficientes interpretables de forma directa— se emplea la Regresión
    Logística como modelo de referencia interpretable, decisión que queda
    documentada en la memoria.
    """
    try:
        vect = pipe.named_steps["tfidf"]
        clf = pipe.named_steps["clf"]
        if not hasattr(clf, "coef_"):
            return {}
        nombres = np.array(vect.get_feature_names_out())
        coefs = np.asarray(clf.coef_)[0]
        orden = np.argsort(coefs)
        return {
            "hacia_negativo": [
                {"termino": str(nombres[i]), "peso": round(float(coefs[i]), 4)}
                for i in orden[:n]],
            "hacia_positivo": [
                {"termino": str(nombres[i]), "peso": round(float(coefs[i]), 4)}
                for i in orden[-n:][::-1]],
        }
    except Exception:
        return {}


def fig_terminos(terminos: dict) -> None:
    if not terminos:
        return
    neg = pd.DataFrame(terminos["hacia_negativo"]).head(15).iloc[::-1]
    pos = pd.DataFrame(terminos["hacia_positivo"]).head(15).iloc[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh(neg["termino"], neg["peso"], color="#c0392b")
    axes[0].set_title("Términos que empujan hacia NEGATIVO")
    axes[1].barh(pos["termino"], pos["peso"], color="#27ae60")
    axes[1].set_title("Términos que empujan hacia POSITIVO")
    for a in axes:
        a.set_xlabel("Peso del coeficiente")
        etiquetar_barras(a, horizontal=True, formato="{:.2f}", tam=7)
    fig.tight_layout()
    guardar_figura(fig, "09_terminos_influyentes")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Fase 2 — modelización")
    parser.add_argument("--muestra", type=int, default=None,
                        help="Entrena sobre una submuestra aleatoria de N reseñas. "
                             "Útil para pruebas rápidas; omítelo para el resultado final.")
    parser.add_argument("--sin-rf", action="store_true",
                        help="Omite Random Forest, que es el modelo más lento.")
    args = parser.parse_args()

    print("=" * 70)
    print("FASE 2 — MODELIZACIÓN Y COMPARATIVA")
    print("=" * 70)

    print("\n[1/6] Cargando y etiquetando...")
    df = solo_etiquetados(anadir_sentimiento(cargar_datos()))
    if args.muestra and args.muestra < len(df):
        df = df.sample(args.muestra, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"      *** SUBMUESTRA DE {args.muestra:,} RESEÑAS (prueba, no resultado final) ***")
    print(f"      {len(df):,} reseñas con sentimiento inequívoco")
    print(f"      positivas: {df['es_positivo'].mean():.1%}")

    print("\n[2/6] Partición agrupada por fármaco...")
    train, test = particion_por_grupo(df)
    X_train, y_train = train["review"], train["es_positivo"]
    X_test, y_test = test["review"], test["es_positivo"]
    print(f"      train: {len(train):,}  |  test: {len(test):,}")
    print(f"      fármacos solo en test: "
          f"{len(set(test['drugName']) - set(train['drugName'])):,}")

    # El vectorizador se ajusta UNA sola vez sobre el conjunto de
    # entrenamiento y se reutiliza para todos los modelos. Sigue sin haber
    # fuga de información (nunca ve el test) y evita repetir el mismo cálculo
    # costoso una vez por modelo.
    print("\n[3/6] Vectorizando el texto (TF-IDF)...")
    t0 = time.time()
    vect = construir_vectorizador()
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)
    print(f"      matriz {Xtr.shape[0]:,} x {Xtr.shape[1]:,} en {time.time() - t0:.0f}s")

    print("\n      Entrenando modelos...")
    resultados, entrenados = [], {}
    catalogo = catalogo_modelos()
    if args.sin_rf:
        catalogo = {k: v for k, v in catalogo.items() if not k.startswith("Random Forest")}

    for nombre, (pipe, _justificacion) in catalogo.items():
        print(f"      · {nombre} ...", end="", flush=True)
        sub = Pipeline(pipe.steps[1:])          # el pipeline sin el paso TF-IDF
        t0 = time.time()
        sub.fit(Xtr, y_train)
        tardanza = time.time() - t0
        res = evaluar(nombre, sub, Xte, y_test, tardanza)
        resultados.append(res)
        entrenados[nombre] = sub
        print(f" F1={res['f1']:.4f}  AUC={res['roc_auc']:.4f}  ({tardanza:.0f}s)")

    print("\n[4/6] Generando figuras...")
    fig_comparativa(resultados)
    fig_roc(resultados, y_test)

    candidatos = [r for r in resultados if not r["modelo"].startswith("Baseline")]
    mejor = max(candidatos, key=lambda r: r["f1"])
    fig_matriz_confusion(mejor, y_test)
    print(f"      mejor modelo: {mejor['modelo']} (F1 = {mejor['f1']:.4f})")

    print("\n[5/6] Explicabilidad global y validación cruzada...")
    # Se intenta con el modelo ganador; si no expone coeficientes
    # interpretables, se recurre a la Regresión Logística como referencia.
    modelo_explicativo = mejor["modelo"]
    terminos = terminos_influyentes_de(vect, entrenados[modelo_explicativo].steps[-1][1])
    if not terminos and "Regresión Logística" in entrenados:
        modelo_explicativo = "Regresión Logística"
        terminos = terminos_influyentes_de(
            vect, entrenados[modelo_explicativo].steps[-1][1])
        print(f"      ({mejor['modelo']} no expone coeficientes; "
              f"se explica con {modelo_explicativo})")
    fig_terminos(terminos)

    # Validación cruzada sobre la matriz ya vectorizada
    cv = cross_val_score(
        LogisticRegression(solver="liblinear", C=4.0, class_weight="balanced",
                           max_iter=1000, random_state=RANDOM_STATE),
        Xtr, y_train, cv=5, scoring="f1", n_jobs=-1)
    print(f"      CV 5-fold (Regresión Logística) F1 = {cv.mean():.4f} ± {cv.std():.4f}")

    print("\n[6/6] Guardando modelo y métricas...")
    # Se reconstruye el pipeline completo (vectorizador + modelo ganador) para
    # que la aplicación pueda recibir texto crudo directamente.
    pipeline_final = Pipeline([("tfidf", vect)] + entrenados[mejor["modelo"]].steps)
    ruta_modelo = DIR_MODELOS / "modelo_sentimiento.joblib"
    joblib.dump(pipeline_final, ruta_modelo, compress=3)
    print(f"      modelo guardado en {ruta_modelo.name}")

    guardar_metricas("modelado_resultados", {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "pct_positivas_train": round(float(y_train.mean()), 4),
        "estrategia_particion": "GroupShuffleSplit agrupado por drugName (80/20)",
        "mejor_modelo": mejor["modelo"],
        "validacion_cruzada_f1_media": round(float(cv.mean()), 4),
        "validacion_cruzada_f1_desv": round(float(cv.std()), 4),
        "resultados": [{k: v for k, v in r.items() if not k.startswith("_")}
                       for r in resultados],
        "modelo_usado_para_explicabilidad": modelo_explicativo,
        "terminos_influyentes": terminos,
        "justificaciones": {n: j for n, (_, j) in catalogo_modelos().items()},
    })

    print("\n" + "=" * 70)
    print("MODELIZACIÓN COMPLETADA.")
    print("=" * 70)


if __name__ == "__main__":
    main()
