# Arquitectura

---

## Flujo de solución

```
DATASET (Azure PdM, data/raw/)
    ↓
DATA CLEANING + JOIN (5 tablas → master_dataset.parquet)
    ↓
FEATURE ENGINEERING (46 features temporales: rolling, lag, mantenimiento)
    ↓
MODEL (Random Forest, entrenado 75% / test 15% / live 10%)
    ↓
MODEL ARTIFACT (models/baseline_model.joblib — 2.50 MB)
    ↓
DASHBOARD LAYER
  ├── model_loader.py → descarga joblib desde GitHub (fallback local)
  ├── data_loader.py  → descarga live_demo.parquet desde GitHub (fallback local)
  ├── compute_risk_from_model() → inferencia batch sobre live_df
  │     ├── model.predict_proba() → failure_probability
  │     ├── threshold (0.5591) → predicción binaria
  │     ├── df_risk → ranking riesgo / criticidad / prioridad
  │     ├── df_telemetry → series temporales por máquina
  │     └── df_errors → histórico de errores simulado
  ↓
STREAMLIT (dashboard/app.py) → UI premium dark theme
    ↓
DECISION (ranking, prioridad, recomendación de intervención)

---

## Capas

### 1. Dataset
- Datos crudos en `data/raw/`.
- Datos procesados en `data/processed/`.

### 2. Data Cleaning
- Limpieza, tratamientos de nulos, inconsistencias, duplicados.
- Validación de calidad.

### 3. Feature Engineering
- Variables derivadas (rolling_mean, trend, rate_of_change, etc.).
- Transformaciones para el modelo.

### 4. Model
- Modelo de ML entrenado en `notebooks/05_modeling.ipynb`.
- **Random Forest** con `class_weight='balanced'`, 46 features, threshold=0.5591.
- PR-AUC=0.9919, ROC-AUC=0.9999, recall=1.0, precision=0.7447.
- Ver `docs/model.md` para detalle completo.

### 5. Model Artifact
- `models/baseline_model.joblib` — 2.50 MB (entrenado con scikit-learn 1.7.2).
- Contiene: `model`, `feature_cols` (46), `decision_threshold`, métricas y metadatos temporales.
- Cargado por el dashboard para inferencia batch vía `dashboard/utils/model_loader.py`.
- Carga con estrategia GitHub → fallback local, cacheada con `@st.cache_resource(ttl=3600)`.

### 6. Streamlit
- Frontend del dashboard.
- `dashboard/app.py` — Entry point, orquestación y UI.
- `dashboard/utils/model_loader.py` — Carga del artefacto ML desde GitHub/local.
- `dashboard/utils/data_loader.py` — Carga de `live_demo.parquet` desde GitHub/local + inferencia (`compute_risk_from_model()`).
- Estrategia de origen: `st.radio('Origen de datos', ['GitHub', 'Local'])` en la sidebar.

### 7. Dashboard
- Interfaz para el responsable de mantenimiento.
- **Tab 1 — Identificar:** Ranking de riesgo (tabla + bar chart), alertas críticas.
- **Tab 2 — Comprender:** Telemetría (selector de sensor + line chart), histórico de errores, métricas dinámicas.
- **Tab 3 — Priorizar:** Cola de intervención ordenada por `priority_score` (riesgo × criticidad), recomendación principal.
- Sidebar muestra metadatos del modelo: PR-AUC, threshold, features, fechas train/test.

### 8. Decision
- Acción de mantenimiento.

---

## FastAPI

**Opcional. No es dependencia del MVP inicial.**

**Razón:**
- 4 semanas de proyecto.
- Equipo junior.
- 1 solo Software Engineer.
- Prioridad: producto funcional.

Si en el futuro se requiere API, se documentará como decisión de arquitectura.