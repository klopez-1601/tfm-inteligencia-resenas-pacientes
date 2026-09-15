# TFM — Inteligencia de reseñas de pacientes (Drugs.com)

Análisis de sentimiento y descubrimiento de temas sobre 215.063 reseñas reales de
pacientes, con modelo productivizado en una aplicación.

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
python src/topics_explicabilidad.py  # Temas (LDA/NMF) + explicabilidad
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
├── memoria/
│   └── memoria.tex           Memoria en LaTeX (≤20 caras)
└── requirements.txt
```

---

## 3. Reproducibilidad

- Semilla fija (`RANDOM_STATE = 42`) en todos los procesos aleatorios.
- Todas las métricas se guardan en `resultados/metricas/` en JSON, de modo que
  las cifras de la memoria proceden siempre de una ejecución real y trazable.
- `requirements.txt` fija las librerías necesarias.

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

## 5. Entrega (checklist de la guía)

- [ ] Memoria ≤ 20 caras (sin anexos, índice ni portada), Arial/Verdana 10-11
- [ ] Apartado de conclusiones
- [ ] Bibliografía breve (≈ media cara)
- [ ] Código en los anexos / repositorio accesible
- [ ] Permisos de acceso para **Carlos Ortega** y **Santiago Mota**
- [ ] Proyecto reproducible
- [ ] Derechos de uso de los datos revisados
- [ ] Vídeo ≤ 5 min en .mp4 (idealmente < 50 MB)
- [ ] Fichero nombrado `Nombre_Apellido1_Apellido2_Titulo.zip`
