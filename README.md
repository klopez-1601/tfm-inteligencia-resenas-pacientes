# TFM — Inteligencia de reseñas de pacientes (Drugs.com)

Análisis de sentimiento y descubrimiento de temas sobre reseñas reales de
pacientes (215.063 en el fichero original; **213.807** tras la limpieza),
con el modelo empaquetado en una aplicación.

**Resultado principal:** F1 de 0,9494 y ROC-AUC de 0,9649 sobre 721
medicamentos que el modelo no había visto durante el entrenamiento.

**Máster en Data Science, Big Data & Business Analytics — UCM**
Opción 1 de la guía: *Análisis de un dataset (orientación Data Scientist)*

---

## 1. Qué tienes que hacer (una sola vez)

### Paso 1 — Descargar los datos

Descarga el dataset desde **una** de estas dos fuentes:

- **Kaggle (más cómodo, CSV):** https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018
- **UCI (fuente original, TSV):** https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com

Coloca los dos ficheros dentro de la carpeta `data/`. El código acepta
indistintamente los nombres `.csv` o `.tsv`:

```
data/
  drugsComTrain_raw.csv    (o .tsv)
  drugsComTest_raw.csv     (o .tsv)
```

### Paso 2 — Instalar dependencias

```bash
cd TFM_DrugReviews
python -m venv .venv
source .venv/bin/activate        # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Paso 3 — Ejecutar el proyecto

```bash
python src/eda.py                    # Análisis exploratorio + figuras
python src/modelado.py               # Preprocesado + comparativa de modelos
python src/topics_explicabilidad.py  # Temas (NMF) + aspectos + explicabilidad
streamlit run src/app.py             # Aplicación (productivización)
```

Cada script deja sus salidas en `resultados/` (figuras en `resultados/figuras/`
y métricas en `resultados/metricas/`).

---

## 2. Estructura del proyecto

```
TFM_DrugReviews/
├── data/                     Los CSV/TSV descargados (no se versionan)
├── src/
│   ├── config.py             Rutas, constantes y carga de datos
│   ├── eda.py                Fase 1: análisis descriptivo
│   ├── modelado.py           Fase 2: preprocesado y modelos
│   ├── topics_explicabilidad.py  Fase 3: temas y explicabilidad
│   └── app.py                Fase 4: aplicación (productivización)
├── modelos/                  Modelos entrenados (.joblib)
├── resultados/
│   ├── figuras/              Gráficos para la memoria
│   └── metricas/             Métricas en JSON/CSV
├── memoria/                  La redacción del TFM
│   ├── memoria.tex           Fuente LaTeX
│   └── memoria.pdf           Documento final (≤20 caras)
└── requirements.txt
```

---

## 3. Reproducibilidad

- Semilla fija (`RANDOM_STATE = 42`) en todos los procesos aleatorios.
- Todas las métricas se guardan en `resultados/metricas/` en JSON, de modo que
  las cifras de la memoria proceden siempre de una ejecución real y trazable.
- `requirements.txt` fija las librerías necesarias. Los resultados de la memoria
  se generaron con Python 3.10, pandas 2.3.3, numpy 2.2.6 y scikit-learn 1.7.2;
  con versiones posteriores el código funciona igual, aunque al cargar el modelo
  ya entrenado las probabilidades individuales pueden variar unos puntos (la
  reseña mixta de la app pasa del 90,9 % al 96,2 % entre scikit-learn 1.7.2 y
  1.9.1). Ninguna clasificación ni conclusión cambia; las métricas globales se
  reproducen reentrenando con `python src/modelado.py`.
- `resultados/metricas/ablacion_negaciones.json` recoge el experimento de la
  sección 5.4 de la memoria (efecto de conservar las negaciones en la lista de
  palabras vacías), ejecutado sobre la misma partición que el resto de modelos.

---

## 4. Datos: fuente, licencia y ética

- **Fuente:** Drug Review Dataset (Drugs.com), UCI Machine Learning Repository (DOI 10.24432/C5SK5S).
- **Licencia:** CC BY 4.0. Los autores solicitan uso **exclusivamente con fines de
  investigación**, sin uso comercial, sin redistribución y **citando** el trabajo original.
  El uso en este TFM cumple esas condiciones.
- **Cita:** Gräßer, F., Kallumadi, S., Malberg, H., Zaunseder, S. (2018).
  *Aspect-Based Sentiment Analysis of Drug Reviews Applying Cross-Domain and
  Cross-Data Learning*. Proceedings of the 2018 International Conference on
  Digital Health.
- **Ética:** las reseñas son públicas y anónimas; no contienen identificadores de
  pacientes. El sistema es una herramienta de apoyo al análisis agregado y **no
  constituye consejo médico** ni sustituye el criterio clínico.

---

## 5. Resultados

Evaluación sobre 36.327 reseñas de test, correspondientes a 721 medicamentos
que no aparecen en el conjunto de entrenamiento.

| Modelo | Accuracy | F1 | F1 macro | ROC-AUC |
|---|---|---|---|---|
| Baseline (clase mayoritaria) | 0,7610 | 0,8643 | 0,4321 | 0,5000 |
| Random Forest (sobre LSA) | 0,8578 | 0,9048 | 0,8122 | 0,9092 |
| Naive Bayes (Complement) | 0,8594 | 0,9045 | 0,8189 | 0,9235 |
| Regresión Logística | 0,9171 | 0,9446 | 0,8900 | 0,9647 |
| **SVM lineal** | **0,9221** | **0,9494** | 0,8898 | **0,9649** |

Todas las cifras proceden de `resultados/metricas/modelado_resultados.json`
y se regeneran ejecutando `python src/modelado.py`.

---

## 6. Autoría

Trabajo Fin de Máster de **Keyla Lisbeth López Chamorro**.
Tutores: Carlos Ortega y Santiago Mota.
