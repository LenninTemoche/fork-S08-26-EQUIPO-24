# PredictiveMaintenance

**Código:** S08-26-EQUIPO-24

**Estado:** ⚙️ En Proceso ~85% — Modelo + Integración Core Completa, UI/UX en Mejora (Stitch)

---

## ¿Qué es?

PredictiveMaintenance es una solución de mantenimiento predictivo que permite a un responsable de mantenimiento pasar de la reactividad a la proactividad.

## Objetivo

Construir una solución que permita responder:

- ¿Qué máquinas están mostrando señales de deterioro?
- Cuáles tienen mayor riesgo de fallar?
- Qué podemos hacer ahora para evitar una parada?

## Criterio de éxito

El sistema debe permitir que un responsable de mantenimiento, sin revisar manualmente grandes cantidades de sensores o múltiples registros históricos, pueda:

1. IDENTIFICAR qué máquinas presentan mayor riesgo de falla.
2. COMPRENDER qué señales o variables justifican ese riesgo.
3. PRIORIZAR qué máquina debería atenderse primero.

## Alcance MVP

### MUST HAVE
- Dataset seleccionado y justificado ✅ **Azure PdM**
- Limpieza y tratamiento de datos
- EDA
- Modelo ML con baseline, predicción de riesgo y explicabilidad
- Dashboard con resumen, ranking de riesgo, prioridad y detalle de máquina
- Integración, pruebas y deploy

### SHOULD HAVE
- SHAP o explicación avanzada
- Filtros avanzados
- Comparación de máquinas
- Exportación
- Recomendación preventiva específica

### COULD HAVE
- RUL
- Horas hasta falla
- Alertas
- API independiente
- Base de datos
- Autenticación
- Docker

## Flujo de solución

Dataset → Limpieza → Feature Engineering → Modelo ML → Artefacto → Streamlit → Dashboard → Decisión

## Arquitectura inicial

- Dataset en data/ (Azure PdM seleccionado)
- Limpieza y feature engineering en src/
- Modelo serializado en models/
- Dashboard en dashboard/app.py (Streamlit)
- FastAPI: opcional, no es dependencia del MVP

## Stack tecnológico

- Python
- Pandas, NumPy
- Scikit-learn
- Joblib
- Plotly / Matplotlib / Seaborn
- Streamlit
- Jupyter / Google Colab
- Git, GitHub
- Deploy: Streamlit Community Cloud
- Stitch 

## Roles del equipo
Activos ✅
### Data Scientists (inicialmente 3)
- Luis Fernando Tapia — Modelado, baseline, modelos, métricas
- Oscar Arauz — Feature Engineering, transformaciones, variables temporales
- Lennin Billey Temoche Gómez — Pipeline ML, validación, serialización, integración ✅

### Data Analysts (inicialmente 4)
- Lorena Urrutia — Product & Business, historias de usuario, KPIs, criticidad ✅
- Alexander Tovar Morcillo — Data Quality, profiling, nulos, outliers
- Héctor García — EDA, visualización, tendencias ✅
- Carlos Vega — Dashboard, UX, Streamlit

### Software Engineer
- Albeiro Burbano — Arquitectura, GitHub, integración, deploy, CI ✅

## Estructura del repositorio

- **`data/`** — Datasets (raw y processed)
- **`notebooks/`** — Exploración y análisis
- **`src/`** — Código de producción (data, features, models, utils)
- **`models/`** — Artefactos del modelo
- **`dashboard/`** — Aplicación Streamlit
- **`tests/`** — Pruebas
- **`docs/`** — Documentación del proyecto
- **`.github/`** — Templates y workflows de CI (GitHub Actions)
- **`.streamlit/`** — Configuración de Streamlit (theme, server)

## Estado actual

**Core MVP (~85%):** Modelo entrenado, serializado e integrado en dashboard Streamlit. Pipeline ML-Dashboard funcional end-to-end. UI/UX en mejora continua con Stitch.

**Dataset seleccionado:** Microsoft Azure Predictive Maintenance (Azure PdM) ✅
- Evaluación completada con matriz de 20 criterios ponderados.
- Análisis automatizado en `notebooks/01_data_exploration.ipynb` (sección 10).
- Documentación en `docs/dataset_selection.md` y `docs/decisions.md` (DEC-009).

**Modelo Candidato (Random Forest):**
- PR-AUC: 0.9932 | ROC-AUC: 0.9999 | Recall: 1.0 | Precision: 0.7634 (threshold=0.5385, values recorded in the current artifact)
- Threshold stored in the current model artifact: 0.5385
- 46 features (sensores + historial errores + mantenimiento + rolling windows 3h/6h/24h + deltas)
- Serializado en `models/baseline_model.joblib` (version de scikit-learn registrada en el artefacto)

**Split temporal (sin data leakage):**
| Split | Filas | % | Periodo | Tasa positivos |
|-------|-------|---|---------|----------------|
| Train | 654,600 | 74.72% | 2015-01-01 → 2015-09-30 | 2.01% |
| Test  | 131,400 | 15.00% | 2015-10-02 → 2015-11-25 | 1.68% |
| Live  | 87,700  | 10.01% | 2015-11-25 → 2016-01-01 | 1.97% |

- Gap de 24h entre train y test
- Validación anti-leakage: 4 estrategias confirman PR-AUC > 0.99 (temporal, mensual, por máquina, aleatorio)

**Dashboard (`dashboard/app.py`):** Pipeline de inferencia batch funcional:
- **Carga de datos:** `live_demo.parquet` (87,700 filas, 100 máquinas) desde GitHub (`feat/modeling_integration/data/processed/`) con fallback a `data/processed/live_demo.parquet` local. Cache 1h (`@st.cache_data`).
- **Carga de modelo:** `baseline_model.joblib` desde GitHub (`feat/modeling_integration/models/`) con fallback local. Cache 1h (`@st.cache_resource`).
- **Selector de origen:** Sidebar con radio `GitHub` / `Local` + botón 🔄 Recargar (limpia caches y rerunea).
- **Inferencia:** `model.predict_proba()` sobre las 46 features del artefacto; el umbral binario se lee del propio artefacto. El ranking usa la última lectura disponible de cada máquina.
- **Outputs:** `df_risk` (ranking riesgo/criticidad/prioridad), `df_telemetry` (series temporales), `df_errors` (histórico simulado).
- **UI actual (en mejora con Stitch):** 3 tabs — Identificar (ranking + bar chart + alertas), Comprender (telemetría + errores + métricas dinámicas), Priorizar (cola intervención + recomendación). Sidebar muestra metadatos modelo (PR-AUC, threshold, features, fechas train/test).

**Tests:** 7/7 passing (`tests/`)

**En progreso (pendiente ~15%):**
- UI/UX refinada con Stitch (tema, componentes, responsive)
- Deploy a Streamlit Cloud / CI/CD rebuild automático
- Streaming tiempo real / Monitoreo drift en producción
- SHAP explainability (opcional)

## Roadmap de 4 semanas

- Semana 1: Discovery + Dataset + Data Foundation + EDA inicial + dashboard skeleton
- Semana 2: ML + integración inicial
- Semana 3: Producto (ranking, criticidad, explicabilidad)
- Semana 4: Testing + Deploy + Demo

Ver docs/roadmap.md para el calendario detallado.

## Backlog

Ver docs/backlog.md para el backlog completo organizado por épicas.

## Reglas de desarrollo

- Crear branch feature/<id>-descripcion, fix/<id>- descripcion, docs/< descripcion>
- Commits con prefijo: feat:, fix:, docs:, refactor:, test:, chore:
- PR por tarea, con al menos un revisor
- Definition of Done: desarrollada, funciona, probada, revisada, integrada, documentada

## Cómo preparar el entorno local

### Requisitos
- Python 3.12 o superior
- Git
- IDE con terminal integrada (VS Code, Antigravity, etc.)

### Paso 1: Clonar
```bash
git clone https://github.com/No-Country-simulation/S08-26-EQUIPO-24
cd S08-26-EQUIPO-24
```

### Paso 2: Crear entorno virtual
```bash
python -m venv .venv
```

### Paso 3: Activar
```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Windows CMD
.venv\Scripts\activate.bat
# Linux/macOS
source .venv/bin/activate
```

### Paso 4: Instalar dependencias
```bash
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install pandas numpy scikit-learn joblib
.venv\Scripts\pip.exe install matplotlib seaborn plotly
.venv\Scripts\pip.exe install streamlit
.venv\Scripts\pip.exe install jupyter ipykernel pytest
```

### Paso 5: Verificar
```bash
.venv\Scripts\python.exe -c "import pandas, streamlit, sklearn; print('OK')"
```

### Paso 6: Ejecutar
```bash
# Notebook
jupyter notebook notebooks/

# Dashboard
streamlit run dashboard/app.py
```

> Ver `CONTRIBUTING.md` para pasos detallados y solución de problemas.

## Cómo ejecutar el dashboard

```bash
streamlit run dashboard/app.py
```

### Estado actual del dashboard

- **Datos:** `live_demo.parquet` (87,700 filas, 100 máquinas, 46 features + target) desde GitHub `feat/modeling_integration/data/processed/` con fallback local.
- **Modelo:** `baseline_model.joblib` (Random Forest, 2.50 MB) desde GitHub `feat/modeling_integration/models/` con fallback local.
- **Carga dual:** Sidebar con selector `GitHub` / `Local` + botón 🔄 Recargar (limpia `st.cache_data` y `st.cache_resource`, fuerza rerun).
- **Riesgo por máquina:** Se muestra la probabilidad de falla de la última lectura disponible. Las predicciones binarias respetan el umbral guardado en el artefacto; no se fuerza una cantidad fija de positivos.
- **UI/UX:** Funcional con 3 tabs (Identificar/Comprender/Priorizar) + sidebar (metadatos modelo, filtros, metadatos máquina). **En mejora continua con Stitch** (tema dark premium, componentes, responsive, accesibilidad).
- **Decisiones:** El responsable de mantenimiento ve ranking de riesgo, señales (telemetría + errores), prioridad y recomendación de intervención (Intervenir/Inspeccionar/Monitorear/Ninguna).

## Cómo contribuir

Ver CONTRIBUTING.md.

## Limitaciones conocidas

- El artefacto del modelo puede mostrar `InconsistentVersionWarning` si la versión de scikit-learn no coincide con la de entrenamiento (1.7.2 vs 1.9.1 actual); se recomienda regenerarlo en el entorno objetivo.
- **UI/UX en mejora activa con Stitch:** Tema dark premium, componentes, responsive y accesibilidad en iteración. La funcionalidad core incluye carga dual, inferencia y ranking por lectura reciente.
- FastAPI inicialmente no está incluido en el MVP.
- Dataset AI4I 2020 solo para validación secundaria.

## Nota sobre datasets

Dataset principal: **Azure PdM** en `data/raw/` (5 archivos). Validación secundaria: AI4I 2020.
Los datos crudos van en data/raw/ y no se modifican.

## Principio de transparencia

Crear variables derivadas es válido cuando existe justificación técnica o de negocio y el proceso es reproducible. No se deben inventar datos históricos.

## Demo (En desarrollo)

El pipeline core (datos → modelo → inferencia → UI) está funcional. Ejecutar:

```bash
streamlit run dashboard/app.py
```

El dashboard permite al responsable de mantenimiento:
1. **Identificar** máquinas con mayor riesgo (ranking interactivo + bar chart + alertas críticas).
2. **Comprender** señales (telemetría temporal volt/rotate/pressure/vibration + histórico de errores + feature importance en `docs/model.md`).
3. **Priorizar** intervención (cola ordenada por `priority_score` = riesgo × criticidad + recomendación principal).

> **Nota:** La UI/UX está en proceso de refinamiento con Stitch. La funcionalidad core (carga GitHub/local, inferencia por máquina y ranking de riesgo) está completa pero en revisión.
