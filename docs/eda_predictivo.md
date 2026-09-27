# EDA Predictivo y Feature Engineering

## Análisis Exploratorio de Datos Orientado a Predicción

### Objetivo

Analizar y responder cuantitativamente qué señales de telemetría, patrones de errores e historial de mantenimiento cambian antes de que ocurra una falla, evaluando la viabilidad de las variables para el modelado predictivo con el `master_dataset.parquet`.

**Notebook asociado:** `notebooks/03_eda_predictivo.ipynb`

---

## 1. Carga y Verificación del Master Dataset

- **Dimensiones:** 876,100 filas × 19 columnas
- **Rango temporal:** 1 de enero de 2015 - 1 de enero de 2016 (registro horario continuo)
- **Máquinas únicas:** 100
- **Calidad de datos:** Sin valores nulos en columnas de telemetría
- **Estructura:** 19 columnas con tipos correctos (datetime, numéricos, categóricos)

---

## 2. Análisis de Fallas y Desbalance de Clases

- **Tasa de pre-falla:** 1.96% (~1 de cada 49 instancias)
- **Desbalance extremo:** El modelo no puede optimizar Accuracy; deben usarse Recall y PR-AUC
- **Por modelo de máquina:**
  - `model1`: 2.98% de eventos positivos (mayor riesgo)
  - `model4`: 1.47% (menor riesgo)

---

## 3. Distribución de Telemetría: Normal vs. Pre-Falla

| Sensor | Delta en Pre-Falla | Interpretación |
|--------|-------------------|----------------|
| `volt` | +5.26V | Incremento de voltaje |
| `pressure` | +4.53 psi | Aumento de presión |
| `rotate` | -28 RPM | Caída de rotación (sobrecarga) |
| `vibration` | Leve incremento | Mayor volatilidad |

**Limitación:** Las distribuciones se superponen, limitando el poder predictivo individual de cada sensor.

---

## 4. Análisis Temporal (48h antes de una falla)

- Tendencia creciente en voltaje y vibración
- Fluctuaciones anómalas en presión
- Caída progresiva de rotación
- Patrones confirman degradación física anticipada

---

## 5. Historia de Errores y Mantenimiento

- Máquinas con errores recientes tienen **3-5× más riesgo de falla**
- `errors_last_7d` es la variable más correlacionada con el target (0.42)
- Tiempo desde último mantenimiento: relación directa con probabilidad de fallo
- Equipos con >7 días sin mantenimiento concentran la mayoría de fallas

---

## 6. Matriz de Correlación

Variables más informativas para el target `failure_next_24h`:
1. `errors_last_7d` (0.42)
2. `time_since_last_error_h` (0.38)
3. `rotate` (-0.31)

**Conclusión:** El historial de errores es la señal más valiosa del dataset.

---

## 7. Conclusiones Ejecutivas

1. **Desbalance severo:** Requiere técnicas de manejo de desbalance (SMOTE, class_weight, umbral de decisión)
2. **Señales de telemetría como indicadores moderados:** Las tendencias son claras pero con superposición
3. **Historial de errores como señal fuerte:** Variable más predictiva del dataset
4. **Mantenimiento como factor protector:** Equipos con mantenimiento reciente tienen hasta 4× menos fallas
5. **Diferencias por modelo:** `model1` concentra la mayor tasa de eventos positivos

---

## 8. Catálogo de Variables Candidatas para Feature Engineering

Las siguientes variables se construirán en el notebook `04_feature_engineering.ipynb`. Todas respetan la regla de **data leakage**: solo usan información disponible hasta el instante $t$.

### Grupo A: Variables Estáticas (ya existen en master_dataset)

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `model` | Categórica | Modelo del equipo (model1, model2, model3, model4). Se codificará con One-Hot Encoding. |
| `age` | Numérica | Antigüedad del equipo en días. Se conserva tal cual. |

### Grupo B: Estadísticas Móviles (Rolling Features) — NUEVAS

| Variable | Ventana | Descripción |
|----------|---------|-------------|
| `volt_rolling_mean_3h` | 3h | Promedio móvil de voltaje (últimas 3h) |
| `volt_rolling_std_3h` | 3h | Desviación estándar móvil de voltaje (últimas 3h) |
| `volt_rolling_mean_6h` | 6h | Promedio móvil de voltaje (últimas 6h) |
| `volt_rolling_std_6h` | 6h | Desviación estándar móvil de voltaje (últimas 6h) |
| `volt_rolling_mean_24h` | 24h | Promedio móvil de voltaje (últimas 24h) |
| `volt_rolling_std_24h` | 24h | Desviación estándar móvil de voltaje (últimas 24h) |
| `rotate_rolling_mean_3h` | 3h | Promedio móvil de rotación (últimas 3h) |
| `rotate_rolling_std_3h` | 3h | Desviación estándar móvil de rotación (últimas 3h) |
| `rotate_rolling_mean_6h` | 6h | Promedio móvil de rotación (últimas 6h) |
| `rotate_rolling_std_6h` | 6h | Desviación estándar móvil de rotación (últimas 6h) |
| `rotate_rolling_mean_24h` | 24h | Promedio móvil de rotación (últimas 24h) |
| `rotate_rolling_std_24h` | 24h | Desviación estándar móvil de rotación (últimas 24h) |
| `pressure_rolling_mean_3h` | 3h | Promedio móvil de presión (últimas 3h) |
| `pressure_rolling_std_3h` | 3h | Desviación estándar móvil de presión (últimas 3h) |
| `pressure_rolling_mean_6h` | 6h | Promedio móvil de presión (últimas 6h) |
| `pressure_rolling_std_6h` | 6h | Desviación estándar móvil de presión (últimas 6h) |
| `pressure_rolling_mean_24h` | 24h | Promedio móvil de presión (últimas 24h) |
| `pressure_rolling_std_24h` | 24h | Desviación estándar móvil de presión (últimas 24h) |
| `vibration_rolling_mean_3h` | 3h | Promedio móvil de vibración (últimas 3h) |
| `vibration_rolling_std_3h` | 3h | Desviación estándar móvil de vibración (últimas 3h) |
| `vibration_rolling_mean_6h` | 6h | Promedio móvil de vibración (últimas 6h) |
| `vibration_rolling_std_6h` | 6h | Desviación estándar móvil de vibración (últimas 6h) |
| `vibration_rolling_mean_24h` | 24h | Promedio móvil de vibración (últimas 24h) |
| `vibration_rolling_std_24h` | 24h | Desviación estándar móvil de vibración (últimas 24h) |

### Grupo C: Deltas y Cambios Instantáneos — NUEVAS

| Variable | Descripción |
|----------|-------------|
| `volt_delta_1h` | Diferencia de voltaje entre  $t$ y $t-1$ hora |
| `rotate_delta_1h` | Diferencia de rotación entre  $t$ y $t-1$ hora |
| `pressure_delta_1h` | Diferencia de presión entre  $t$ y $t-1$ hora |
| `vibration_delta_1h` | Diferencia de vibración entre  $t$ y $t-1$ hora |

### Grupo D: Variables de Errores y Mantenimiento (ya existen)

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `errors_last_24h` | Numérica | Conteo de errores en últimas 24h |
| `errors_last_7d` | Numérica | Conteo de errores en últimos 7 días |
| `distinct_errors_last_24h` | Numérica | Errores distintos en 24h |
| `time_since_last_error_h` | Numérica | Horas desde el último error |
| `hours_since_maintenance` | Numérica | Horas desde el último mantenimiento |
| `maintenance_count_30d` | Numérica | Conteo de mantenimientos en últimos 30 días |
| `has_recent_maintenance` | Binaria | 1 si mantenimiento ≤ 24h, 0 si no |

**Notas**

- `days_since_maintenance` y `time_since_last_component_replacement_h` son derivadas/alias de `hours_since_maintenance` y no se recrean.
- Las variables de rolling stats y deltas son completamente nuevas y no duplican ninguna existente en `master_dataset`.