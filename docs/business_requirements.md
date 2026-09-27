# Requisitos de Negocio

---

## 1. Problema

Un responsable de mantenimiento actualmente debe revisar manualmente grandes cantidades de sensores y registros históricos para responder:

> "¿Qué máquinas se rompieron y cuándo debemos repararlas?"

Esto genera:
- Tiempo de análisis manual elevado.
- Detección tardía de deterioro.
- Paradas no planificadas.
- Dificultad para priorizar intervenciones.

---

## 2. Objetivo

Construir una solución de mantenimiento predictivo que permita pasar de la reactividad a la proactividad:

> "¿Qué máquinas están mostrando señales de deterioro, cuáles tienen mayor riesgo de fallar y qué podemos hacer ahora para evitar una parada?"

---

## 3. Usuario principal

**Responsable de mantenimiento.**

- Necesita identificar riesgo, comprender señales y priorizar intervenciones.
- No requiere conocimiento técnico en ML.
- Quiere una respuesta rápida y accionable.

---

## 4. Stakeholders

| Stakeholder | Interés |
|---|---|
| Responsable de mantenimiento | Usario principal |
| Producción | Reducción de paradas |
| Operaciones | Eficiencia de recursos |
| Ingeniería | Mantenimiento de activos |
| Dirección | Retorno de inversión |

---

## 5. Dolor actual

- Análisis manual de sensores.
- Falta de ranking de riesgo.
- Sin explicación del porqué.
- Priorización subjetiva.
- Paradas imprevistas.

---

## 6. Objetivos específicos

1. Identificar máquinas con mayor riesgo de falla.
2. Explicar qué señales justifican el riesgo.
3. Priorizar intervenciones combinando riesgo y criticidad.
4. Reducir el análisis manual de datos.
5. Apoyar la decisión preventiva.

---

## 7. Historias de usuario

### HU-01
Como responsable de mantenimiento, quiero identificar las máquinas con mayor riesgo de falla.

### HU-02
Como responsable de mantenimiento, quiero conocer las señales que explican el riesgo.

### HU-03
Como responsable de mantenimiento, quiero priorizar las máquinas combinando riesgo y criticidad.

### HU-04
Como responsable de mantenimiento, quiero revisar tendencia e historial disponible.

### HU-05
Como responsable de mantenimiento, quiero obtener una orientación para decidir una intervención preventiva.

---

## 8. KPIs potenciales

- Tiempo de detección de deterioro.
- Número de fallas evitadas.
- Reducción de downtime.
- Tiempo de análisis de riesgo por máquina.
- Precisión de priorización.

---

## 9. Definiciones

### Falla
Evento de pérdida de función de una máquina o componente que requiere reparación o reemplazo.

### Máquina crítica
Aquella cuya falla interrumpe una línea de producción, genera seguridad o tiene alto costo operativo.

### Alto riesgo
Condición en la que el modelo estima una probabilidad de falla superior al umbral definido dentro del horizonte de predicción.

### Prioridad
Combinación de riesgo y criticidad que determina el orden de atención.

---

## 10. Criterios de aceptación

El sistema debe permitir:

1. Identificar qué máquinas tienen mayor riesgo.
2. Explicar por qué (señales relevantes).
3. Priorizar cuál atender primero.
4. Sin revisar manualmente grandes volúmenes de datos.
5. Apoyar una decisión preventiva.

---

## Estado actual

**FASE:** Modelado + Integración Streamlit (MVP Funcional)

✅ Modelo Random Forest entrenado (PR-AUC 0.9919, threshold 0.5591)
✅ Dashboard Streamlit con inferencia real sobre live_demo.parquet
✅ 46 features alineadas entre modelo y datos
✅ Tests pasando (7/7)
✅ Criterios de aceptación cumplidos (identificar riesgo, comprender señales, priorizar intervención)

Ver `docs/model.md` y `docs/demo_script.md` para detalle técnico.