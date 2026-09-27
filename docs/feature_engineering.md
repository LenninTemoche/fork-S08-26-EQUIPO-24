# Feature Engineering Avanzado

## Visión General

Construcción de la matriz de características para Machine Learning a partir del dataset analítico unificado `master_dataset.parquet` (876,100 registros, 100 máquinas, año 2015).

**Notebook asociado:** `notebooks/04_feature_engineering.ipynb`
**Dataset resultante:** `data/processed/features_dataset_part1.parquet` + `data/processed/features_dataset_part2.parquet` (segmentado por máquinas, ~55 MB cada uno, por debajo del límite de 100 MB de GitHub)

---

## Objetivos

- Consolidar variables predictivas en grupos técnicos
- Preservar causalidad temporal (sin *data leakage*)
- Evitar duplicidad de variables
- Producir una matriz lista para modelado (0 nulos, 0 infinitos)

---

## Estructura de Features (49 columnas totales)

### Grupo G — Target (1 variable)

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `failure_next_24h` | Binaria | 1 si la falla ocurre en las próximas 24h, 0 si no. **1.96% de positivos (desbalance 1:49)** |

### Grupo A — Features de Máquina (4 variables)

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `age` | Numérica | Antigüedad del equipo en días |
| `model_model2` | Binaria | One-Hot Encoding de `model` (referencia: model1) |
| `model_model3` | Binaria | One-Hot Encoding de `model` |
| `model_model4` | Binaria | One-Hot Encoding de `model` |

### Grupo B — Rolling Media (12 variables)

| Variable | Ventana | Descripción |
|----------|---------|-------------|
| `{sensor}_roll_mean_3h` | 3h | Promedio móvil de cada sensor (últimas 3h) |
| `{sensor}_roll_mean_6h` | 6h | Promedio móvil de cada sensor (últimas 6h) |
| `{sensor}_roll_mean_24h` | 24h | Promedio móvil de cada sensor (últimas 24h) |

Sensores: `volt`, `rotate`, `pressure`, `vibration`

### Grupo C — Rolling Std (12 variables)

| Variable | Ventana | Descripción |
|----------|---------|-------------|
| `{sensor}_roll_std_3h` | 3h | Desviación estándar móvil (volatilidad) |
| `{sensor}_roll_std_6h` | 6h | Desviación estándar móvil (volatilidad) |
| `{sensor}_roll_std_24h` | 24h | Desviación estándar móvil (volatilidad) |

### Grupo D — Deltas (4 variables)

| Variable | Descripción |
|----------|-------------|
| `volt_delta` | Diferencia horaria $t - (t-1)$ de voltaje |
| `rotate_delta` | Diferencia horaria de rotación |
| `pressure_delta` | Diferencia horaria de presión |
| `vibration_delta` | Diferencia horaria de vibración |

### Grupo E — Errores (5 variables)

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `errors_last_24h` | Numérica | Conteo de errores en últimas 24h |
| `errors_last_7d` | Numérica | Conteo de errores en últimos 7 días |
| `distinct_errors_last_24h` | Numérica | Errores distintos en 24h |
| `time_since_last_error_h` | Numérica | Horas desde el último error (imputado: 8760 si no hay errores) |
| `has_error_recent` | Binaria | 1 si hubo error en 24h, 0 si no |

### Grupo F — Mantenimiento (5 variables)

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `hours_since_maintenance` | Numérica | Horas desde el último mantenimiento |
| `days_since_maintenance` | Numérica | Derivada: `hours_since_maintenance` / 24 |
| `maintenance_count_30d` | Numérica | Conteo de mantenimientos en últimos 30 días |
| `time_since_last_component_replacement_h` | Numérica | Alias de `hours_since_maintenance` |
| `has_recent_maintenance` | Binaria | 1 si mantenimiento ≤ 24h, 0 si no |

---

## Estrategia de Exportación (Segmentación por Máquinas)

Para cumplir con el límite de 100 MB de GitHub, el dataset se exporta en **dos partes** usando compresión Brotli:

- `features_dataset_part1.parquet`: Máquinas 1-50 (~55 MB)
- `features_dataset_part2.parquet`: Máquinas 51-100 (~55 MB)

**Reconstrucción en `05_modeling.ipynb`:**
```python
import pandas as pd
base = "https://raw.githubusercontent.com/No-Country-simulation/S08-26-EQUIPO-24/main/data/processed/"
features_df = pd.concat([
    pd.read_parquet(base + "features_dataset_part1.parquet"),
    pd.read_parquet(base + "features_dataset_part2.parquet")
], ignore_index=True)
```

La reconstrucción es exacta — no se pierde ni altera ninguna celda.

---

## Validaciones de Calidad

| Validación | Resultado |
|------------|-----------|
| Dimensiones | 876,100 filas × 49 columnas |
| Nulos | 0 |
| Infinitos | 0 |
| Data leakage | Preservado (rolling features con info pasada) |
| Duplicados semánticos | `days_since_maintenance` y `time_since_last_component_replacement_h` son derivadas/alias de `hours_since_maintenance` |
| Desbalance de clases | Confirmado 1:49 (1.96% positivos) |

---

## Variables por Poder Predictivo (correlación con target)

| Rango | Variable | Correlación | Grupo |
|-------|----------|-------------|-------|
| 1 | `distinct_errors_last_24h` | 0.5514 | E — Errores |
| 2 | `errors_last_24h` | 0.5508 | E — Errores |
| 3 | `errors_last_7d` | 0.1819 | E — Errores |
| 4 | `hours_since_maintenance` | 0.0924 | F — Mantenimiento |
| 5 | `vibration_roll_mean_24h` | ~0.06 | B/C — Rolling |
| 6 | `pressure_roll_mean_24h` | ~0.05 | B/C — Rolling |
| 7 | `volt_roll_mean_24h` | ~0.04 | B/C — Rolling |
| 8 | `age` | 0.0297 | A — Máquina |
| 9 | `rotate_delta` | ~-0.08 | D — Delta |
| 10 | Variables de modelo | ~0.02 | A — Máquina |

---

## Mejoras Sugeridas para el Modelado

1. **Feature Selection:** `SelectKBest` o Random Forest Importance para reducir a ~20 variables
2. **Interacciones:** Términos entre `errors_last_24h` y sensores de telemetría
3. **Scaling:** `StandardScaler` para algoritmos sensibles a escala
4. **Class Weight:** `class_weight='balanced'` o SMOTE para el desbalance 1:49
5. **Cross-Validation:** `StratifiedKFold` preservando la distribución de clases

---

**Nota**

El archivo exportado segmentado ocupa aproximadamente **110 MB en total** (~55 MB por parte) en formato Parquet con compresión Brotli, frente a los ~35.6 MB del `master_dataset.parquet` original. La segmentación por máquinas mitiga el riesgo de exceder el límite de 100 MB de GitHub por archivo.