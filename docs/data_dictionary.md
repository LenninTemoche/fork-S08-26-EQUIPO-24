# Diccionario de Datos

Documentación de variables del **dataset seleccionado: Microsoft Azure Predictive Maintenance (Azure PdM)**.

---

## Dataset A: Azure PdM — Tablas en `data/raw/`

### 1. df_telemetry (Telemetría horaria)

| Variable | Tipo | Unidad | Descripción | Fuente | Observada/Derivada | Uso |
|---|---|---|---|---|---|---|
| machineID | int | - | Identificador único de máquina (1–100) | Sensor | Observada | Agrupación, ranking, join |
| datetime | datetime | - | Timestamp de la lectura (horario) | Sensor | Observada | Series de tiempo, join |
| volt | float | V | Tensión eléctrica | Sensor | Observada | Feature, proxy estabilidad |
| rotate | float | RPM | Velocidad de rotación | Sensor | Observada | Feature, proxy movimiento |
| pressure | float | psi | Presión de trabajo | Sensor | Observada | Feature, proxy esfuerzo |
| vibration | float | mm/s² | Nivel de vibración | Sensor | Observada | Feature, **mejor indicador deterioro** |

### 2. df_errors (Errores/Alarmas)

| Variable | Tipo | Unidad | Descripción | Fuente | Observada/Derivada | Uso |
|---|---|---|---|---|---|---|
| datetime | datetime | - | Timestamp del error | Sistema | Observada | Feature, eventos pre-falla |
| machineID | int | - | Identificador de máquina | Sistema | Observada | Join, agrupación |
| errorID | string | - | Código de error (error1–error5) | Sistema | Observada | Feature, tipo de alarma |

### 3. df_failures (Fallas registradas)

| Variable | Tipo | Unidad | Descripción | Fuente | Observada/Derivada | Uso |
|---|---|---|---|---|---|---|
| datetime | datetime | - | Timestamp de la falla | Sistema | Observada | **Target**, eventos críticos |
| machineID | int | - | Identificador de máquina | Sistema | Observada | Join, agrupación |
| failure | string | - | Componente fallido (comp1–comp4) | Sistema | Observada | **Target multi-clase**, modo de falla |

### 4. df_machines (Información de máquinas)

| Variable | Tipo | Unidad | Descripción | Fuente | Observada/Derivada | Uso |
|---|---|---|---|---|---|---|
| machineID | int | - | Identificador único | Sistema | Observada | Join, agrupación |
| model | string | - | Modelo/tipo de máquina | Especificación | Observada | Feature, segmentación |
| age | int | años | Antigüedad de la máquina | Especificación | Observada | Feature, criticidad |

### 5. df_maint (Historial de mantenimiento)

| Variable | Tipo | Unidad | Descripción | Fuente | Observada/Derivada | Uso |
|---|---|---|---|---|---|---|
| datetime | datetime | - | Fecha de mantenimiento | Sistema | Observada | Feature, cálculo RUL |
| machineID | int | - | Identificador de máquina | Sistema | Observada | Join, agrupación |
| comp | string | - | Componente reemplazado (comp1–comp4) | Sistema | Observada | Feature, vida útil componente |

---

## Dataset B: AI4I 2020 — Validación secundaria

| Variable | Tipo | Unidad | Descripción | Fuente | Observada/Derivada | Uso |
|---|---|---|---|---|---|---|
| UDI | int | - | Identificador único de registro | Índice | Observada | Índice |
| Product ID | string | - | Identificador de producto/máquina | Sistema | Observada | Agrupación (limitada) |
| Type | string | - | Tipo de máquina (L, M, H) | Especificación | Observada | Feature, segmentación |
| Air temperature [K] | float | K | Temperatura ambiente | Sensor | Observada | Feature |
| Process temperature [K] | float | K | Temperatura de proceso | Sensor | Observada | Feature |
| Rotational speed [rpm] | float | rpm | Velocidad de rotación | Sensor | Observada | Feature |
| Torque [Nm] | float | Nm | Par de torsión | Sensor | Observada | Feature, proxy esfuerzo |
| Tool wear [min] | float | min | Desgaste acumulado herramienta | Sensor | Observada | Feature, proxy RUL |
| Machine failure | int | - | Fallo binario (0/1) | Sistema | Observada | **Target binario** |
| TWF, HDF, PWF, OSF, RNF | int | - | Tipos de falla específicos (multi-label) | Sistema | Observada | Target multi-label |

---

## Variables derivadas planificadas (Azure PdM → data/processed/)

| Variable | Tipo | Fórmula / Regla | Justificación |
|---|---|---|---|
| hours_since_maint | float | `(datetime - last_maint_datetime).total_seconds() / 3600` | Vida útil residual, degradación |
| rolling_mean_24h | float | `sensor.rolling(24).mean()` | Tendencia suavizada, reduce ruido |
| rolling_std_24h | float | `sensor.rolling(24).std()` | Volatilidad, inestabilidad |
| rolling_mean_168h | float | `sensor.rolling(168).mean()` | Tendencia semanal |
| vibration_trend_24h | float | `slope(vibration.rolling(24))` | Aceleración del deterioro |
| lag_1h, lag_24h | float | `sensor.shift(1), sensor.shift(24)` | Autocorrelación, memoria |
| failure_next_24h | bool | `failure in next 24h` | Target anticipado |
| failure_next_168h | bool | `failure in next 168h` | Target horizonte semanal |
| RUL_hours | float | `hours_to_next_failure` | Remaining Useful Life |

---

## Variables derivadas implementadas (usadas por el modelo final — 46 features)

Las siguientes 46 features se generaron en `notebooks/05_modeling.ipynb` y se usan para entrenar e inferir con el Random Forest final (`models/baseline_model.joblib`).

### Sensores base (observados)
| Variable | Tipo | Origen | Descripción |
|---|---|---|---|
| volt | float | Telemetría | Tensión eléctrica (V) |
| rotate | float | Telemetría | Velocidad de rotación (RPM) |
| pressure | float | Telemetría | Presión de trabajo (psi) |
| vibration | float | Telemetría | Nivel de vibración (mm/s²) |

### Características de máquina (observadas)
| Variable | Tipo | Origen | Descripción |
|---|---|---|---|
| age | int | df_machines | Antigüedad de la máquina (años) |

### Historial de errores (derivadas)
| Variable | Tipo | Fórmula / Regla | Descripción |
|---|---|---|---|
| errors_last_24h | int | Conteo errores en (t-24h, t] | Volumen de errores recientes |
| errors_last_7d | int | Conteo errores en (t-7d, t] | Volumen de errores semanales |
| distinct_errors_last_24h | int | Errores únicos en (t-24h, t] | Diversidad de fallos recientes |
| time_since_last_error_h | float | Horas desde último error | Recencia de fallos (crítico) |
| has_error_recent | bool | 1 si errors_last_24h > 0 | Flag de error reciente |

### Historial de mantenimiento (derivadas)
| Variable | Tipo | Fórmula / Regla | Descripción |
|---|---|---|---|
| hours_since_maintenance | float | Horas desde último mantenimiento | Degradación por falta de mant. |
| days_since_maintenance | float | Días desde último mantenimiento | Versión en días |
| maintenance_count_30d | int | Mantenimientos en (t-30d, t] | Frecuencia de mantenimiento |
| time_since_last_component_replacement_h | float | Horas desde último reemplazo | Vida útil componente actual |
| has_recent_maintenance | bool | 1 si maintenance_count_30d > 0 | Flag de mantenimiento reciente |

### Modelo de máquina (one-hot encoding)
| Variable | Tipo | Descripción |
|---|---|---|
| model_model2 | int (0/1) | Modelo tipo 2 |
| model_model3 | int (0/1) | Modelo tipo 3 |
| model_model4 | int (0/1) | Modelo tipo 4 |
*(model_model1 es la referencia)*

### Rolling windows (rolling mean/std) — 3h, 6h, 24h
| Sensor | Ventana | Features (media, std) |
|---|---|---|
| volt | 3h | `volt_roll_mean_3h`, `volt_roll_std_3h` |
| rotate | 3h | `rotate_roll_mean_3h`, `rotate_roll_std_3h` |
| pressure | 3h | `pressure_roll_mean_3h`, `pressure_roll_std_3h` |
| vibration | 3h | `vibration_roll_mean_3h`, `vibration_roll_std_3h` |
| volt | 6h | `volt_roll_mean_6h`, `volt_roll_std_6h` |
| rotate | 6h | `rotate_roll_mean_6h`, `rotate_roll_std_6h` |
| pressure | 6h | `pressure_roll_mean_6h`, `pressure_roll_std_6h` |
| vibration | 6h | `vibration_roll_mean_6h`, `vibration_roll_std_6h` |
| volt | 24h | `volt_roll_mean_24h`, `volt_roll_std_24h` |
| rotate | 24h | `rotate_roll_mean_24h`, `rotate_roll_std_24h` |
| pressure | 24h | `pressure_roll_mean_24h`, `pressure_roll_std_24h` |
| vibration | 24h | `vibration_roll_mean_24h`, `vibration_roll_std_24h` |

### Deltas (cambio instantáneo)
| Variable | Tipo | Fórmula | Descripción |
|---|---|---|---|
| volt_delta | float | `volt.diff()` | Cambio de voltaje vs hora anterior |
| rotate_delta | float | `rotate.diff()` | Cambio de rotación |
| pressure_delta | float | `pressure.diff()` | Cambio de presión |
| vibration_delta | float | `vibration.diff()` | Cambio de vibración |

### Target
| Variable | Tipo | Fórmula | Descripción |
|---|---|---|---|
| failure_next_24h | int (0/1) | 1 si falla en (t, t+24h] | Variable objetivo (target) |

---

## Convenciones

- **Observada:** dato registrado directamente por sensores o sistemas.
- **Derivada:** variable calculada a partir de datos observados (con justificación técnica).
- **Target:** variable objetivo del modelo (ej: failure, failure_next_Nh, RUL).

---

## Estado actual

**Dataset principal: Azure PdM** — Diccionario completado para 5 tablas raw.
**Validación secundaria: AI4I 2020** — Diccionario documentado para benchmark.

✅ **Dataset procesado generado:** `data/processed/live_demo.parquet` (87,700 filas, 100 máquinas, 46 features + target).
✅ **Feature set final:** 46 features documentadas arriba, alineadas con `models/baseline_model.joblib`.