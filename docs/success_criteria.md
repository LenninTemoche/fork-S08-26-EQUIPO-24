# Criterios de Éxito

Este documento define cómo medimos el éxito del proyecto PredictiveMaintenance.

---

## Criterio de éxito oficial

El sistema debe permitir que un responsable de mantenimiento, **sin revisar manualmente grandes cantidades de sensores o múltiples registros históricos**, pueda:

1. **IDENTIFICAR** ¿Qué máquinas presentan mayor riesgo de falla?
2. **COMPRENDER** ¿Qué señales o variables justifican ese riesgo?
3. **PRIORIZAR** ¿Qué máquina debería atenderse primero?

---

## Matriz de criterios

| Criterio | Qué debe permitir | Evidencia final |
|---|---|---|
| Identificar riesgo | Detectar máquinas de mayor riesgo | Ranking de máquinas con score de riesgo |
| Comprender riesgo | Conocer señales relevantes | Explicación del score por variable |
| Priorizar | Decidir qué atender primero | Riesgo + criticidad → prioridad |
| Evitar análisis manual | Sintetizar información | Dashboard con resumen y alertas |
| Apoyar decisión preventiva | Indicar prioridad/acción | Recomendación de intervención |

---

## Prueba de aceptación final

El producto solo podrá considerarse exitoso si:

1. El sistema responde **qué máquina tiene mayor riesgo**.
2. El sistema explica **por qué**.
3. El sistema permite identificar **cuál debe atenderse primero**.
4. El usuario **no necesita revisar manualmente** grandes volúmenes de datos.
5. La información permite tomar una **decisión preventiva**.

---

## Criterios de aceptación por capa

### Capa de datos
- [x] Dataset seleccionado y justificado (Azure PdM)
- [x] Diccionario de datos disponible (`docs/data_dictionary.md`)
- [x] Limpieza aplicada (nulos, inconsistencias, duplicados)
- [x] Variables relevantes identificadas (46 features)
- [x] Separación train/validation/test definida (75% / 15% / 10%, gap 24h)

### Capa de modelo
- [x] Baseline definido (Logistic Regression, PR-AUC 0.8843)
- [x] Modelo(s) candidato(s) evaluado(s)
- [x] Score de riesgo interpretable (probabilidad de falla 0–100)
- [x] Validación realizada (4 estrategias de split, PR-AUC > 0.99)
- [x] Explicabilidad incluida (feature importance, top 15)

### Capa de dashboard
- [x] Resumen general (KPIs + bar chart de riesgo)
- [x] Estado de máquinas (tabla de riesgo con filtros)
- [x] Ranking de riesgo
- [x] Prioridad (cola de intervención ordenada)
- [x] Detalle de máquina (histórico de errores + telemetría)
- [x] Explicación del riesgo (feature importance en docs/model.md)

---

## Estado actual

**FASE:** Modelado + Integración Streamlit (Semana 3-4)

**Estado:** ✅ Modelo entrenado y serializado. ✅ Dashboard con integración ML funcional. ✅ Tests pasando (7/7).
Ver `docs/model.md` para detalle del modelo y `docs/demo_script.md` para el estado de la integración.