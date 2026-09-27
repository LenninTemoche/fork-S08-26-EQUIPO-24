# Guion de demo

Este documento es el guion para la demostración del MVP.

**IMPORTANTE:** ✅ **Producto implementado y funcional.** El modelo ML está entrenado e integrado en el dashboard de Streamlit con datos reales.

---

## Objetivo de la demo

Demostrar que el sistema permite a un responsable de mantenimiento:

1. Identificar máquinas con mayor riesgo.
2. Comprender las señales del riesgo.
3. Priorizar qué atender primero.

---

## Flujo de demo

### 1. Contexto
- Problema: paradas imprevistas, análisis manual de sensores.
- Solución: mantenimiento predictivo con modelo ML (Random Forest, PR-AUC 0.9919).

### 2. Dashboard general
- Resumen de KPIs (máquinas monitoreadas, riesgo crítico, próximo mantenimiento, riesgo promedio).
- Ranking de riesgo (bar chart interactivo con tooltips).
- Alertas de máquinas en nivel crítico.

### 3. Identificar riesgo
- Seleccionar la máquina con mayor riesgo o filtrar por estado/criticidad.
- Mostrar score de riesgo (0–100) y probabilidad de falla.

### 4. Comprender riesgo
- Señales relevantes (feature importance en `docs/model.md`).
- Tendencias de sensores (volt, rotate, pressure, vibration) en gráfico temporal.
- Histórico de errores por máquina.

### 5. Priorizar
- Combinar riesgo y criticidad.
- Mostrar prioridad y orden de intervención.
- Recomendación principal destacada en card.

### 6. Acción
- Recomendación de intervención (Intervenir / Inspeccionar / Monitorear / Ninguna).

---

## Estado actual

**✅ COMPLETADO - Dashboard e Integración ML Funcionando**

---

## Resumen del Estado Actual

| Componente | Estado | Ubicación |
|------------|--------|-----------|
| Notebook modelado | ✅ Completado | `notebooks/05_modeling.ipynb` |
| Modelo serializado | ✅ En GitHub | `feat/modeling_integration/models/baseline_model.joblib` |
| Live demo data | ✅ En GitHub | `feat/modeling_integration/data/processed/live_demo.parquet` |
| Dashboard | ✅ Funcional | `dashboard/` |
| Model loader | ✅ Funcional | `dashboard/utils/model_loader.py` (GitHub/local + cache) |
| Data loader | ✅ Funcional | `dashboard/utils/data_loader.py` (carga real, inferencia) |
| Tests | ✅ Pasando | `tests/test_prevalence.py`, `tests/test_project_structure.py` (7/7) |

---

## Verificación del Notebook (05_modeling.ipynb) ✅

### ✅ Conforme
- **Split temporal correcto**: 75% train / 15% test / 10% live con gap de 24h
- **live_df exportado**: `data/processed/live_demo.parquet` (87,700 filas, 100 máquinas)
- **Modelo guardado**: `models/baseline_model.joblib` con Random Forest, 46 features, threshold=0.5591
- **Artefacto completo**: Incluye model, feature_cols (46), decision_threshold (0.5591), métricas y metadatos temporales
- **Validación anti-leakage**: 4 splits confirmados (temporal, mensual, por máquina, aleatorio) → PR-AUC > 0.99 en todos

### ⚠️ Observaciones Menores
- El scaler **no** se guarda en el joblib (RF no lo necesita) → `model_loader.py` lo maneja correctamente
- `feature_cols` = 46 features (excluye datetime, machineID, failure_next_24h)
- Target `failure_next_24h` balanceado en train/test/live (~2% positivos)
- El artefacto fue entrenado con scikit-learn 1.7.2; el entorno actual usa 1.9.1 (`InconsistentVersionWarning`)

---

## Plan de Implementación (COMPLETADO)

### FASE 1: Carga de Datos Reales (data_loader.py) ✅

1. **`load_live_demo_data()`** - Carga `live_demo.parquet` desde GitHub con fallback local + cache 1h (`@st.cache_data`)
2. **`_extract_machine_metadata()`** - Extrae metadatos únicos de máquinas desde live_df
3. **`compute_risk_from_model()`** - Ejecuta inferencia del modelo sobre live_df y retorna df_risk + df_telemetry + df_errors

### FASE 2: Integración en app.py ✅

- Reemplazado `load_mock_data()` por carga real (`load_live_demo_data`, `compute_risk_from_model`, `get_model`)
- Integrada metadata del modelo en UI (sidebar: fuente, PR-AUC, threshold, fechas train/test)

### FASE 3: Componentes - Adaptación a Datos Reales ✅

| Componente | Estado |
|------------|--------|
| `risk_table.py` | ✅ Usa `df_risk` real (machine_id, risk_score, risk_level, criticality, priority) |
| `sensor_chart.py` | ✅ Filtra `df_telemetry` por machine_id, usa timestamps reales |
| `machine_detail.py` | ✅ Muestra error_code/description reales de `df_errors` |
| `priority_list.py` | ✅ Ordena por `priority_score` real (riesgo × criticidad) |

### FASE 4: Validación y Pruebas ✅

```bash
cd dashboard && streamlit run app.py
```

- [x] Modelo carga desde GitHub (fallback local OK)
- [x] live_demo.parquet carga (87,700 filas, cache 1h)
- [x] 100 máquinas con risk_score real (inferencia del modelo)
- [x] Telemetría temporal real (datetime de live_df)
- [x] Errores históricos reales (df_errors generado desde recent_errors)
- [x] Sidebar expone PR-AUC, threshold, fechas train/test

---

## Riesgos y Mitigación

| Riesgo | Mitigación |
|--------|------------|
| GitHub rate limit / caída | Fallback local implementado en `model_loader.py` y `data_loader.py` |
| Columnas feature mismatch | Validación estricta en `predict_probabilities()` + logging claro |
| live_demo.parquet grande (~87k filas) | Cache `@st.cache_data(ttl=3600)` en loader |
| Zona horaria datetime | `pd.to_datetime()` en `data_loader.py` |
| InconsistentVersionWarning (sklearn) | Documentado; recomendar alinear versión o regenerar artefacto |

---

## Próximos Pasos (Out of Scope)

- [ ] Hyperparameter tuning (actualmente Random Forest optimizado)
- [ ] SHAP explainability (feature importance ya documentada)
- [ ] Deploy a Streamlit Cloud / Azure Container Apps
- [ ] CI/CD para rebuild automático del modelo
- [ ] Streaming en tiempo real (actualizar live_demo.parquet periódicamente)
- [ ] Monitoreo de drift en producción

---

## Criterios de Aceptación

1. ✅ **Dashboard carga sin mock data**: `app.py` usa `load_live_demo_data()` + `compute_risk_from_model()`, no `load_mock_data()`
2. ✅ **Modelo desde GitHub**: `model_loader.py` descarga desde `feat/modeling_integration/models/baseline_model.joblib`
3. ✅ **Live demo visible**: 100 máquinas con risk_score basado en inferencia real
4. ✅ **Telemetría temporal**: Gráficos usan `datetime` real de live_df
5. ✅ **Errores reales**: `machine_detail.py` muestra códigos/descripciones de df_errors
6. ✅ **Métricas expuestas**: Sidebar muestra PR-AUC (0.9919), threshold (0.5591), fechas train/test
