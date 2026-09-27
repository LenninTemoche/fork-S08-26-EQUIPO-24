# Dashboard PredictiveMaintenance

Instrucciones rápidas (español)

Propósito
- Aplicación Streamlit para visualización de riesgos y telemetría basada en el artefacto `models/baseline_model.joblib` y el dataset `data/processed/live_demo.parquet`.

Requisitos
- Python 3.8+
- Virtualenv / venv recomendado
- Dependencias listadas en `requirements.txt` (instalar con `pip install -r requirements.txt`).

Ejecutar localmente
1. Activar entorno virtual (Windows PowerShell):

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Modo de origen (GitHub / Local)
- En la barra lateral superior hay una opción `Origen de datos`:
  - `GitHub`: intenta descargar `live_demo.parquet` y `baseline_model.joblib` desde las URLs configuradas en el código (requiere internet).
  - `Local`: fuerza la carga desde los archivos locales del repositorio (`data/processed/live_demo.parquet` y `models/baseline_model.joblib`). Útil para desarrollo sin conexión.

Recargar artefactos
- Use el botón `🔄 Recargar modelo y datos` en la barra lateral para limpiar caches y forzar recarga.

Mejoras implementadas en esta rama
- Normalización robusta de columnas: el loader ahora soporta `machineID` y `machine_id` y normaliza al formato `machine_id`.
- Opción de `Local` vs `GitHub` en la UI para elegir el origen de datos y modelo.
- Cacheo del artefacto del modelo con `st.cache_resource` (mejor para objetos pesados).
- Manejo de errores con mensajes claros y `st.spinner` durante la carga.
- Botón de recarga que limpia caches (`st.cache_data` y `st.cache_resource`) y fuerza rerun.
- Documentación (este archivo) con pasos de prueba.

Notas de desarrollo
- Si el dashboard falla por falta de columnas requeridas, verifique que `data/processed/live_demo.parquet` contenga las features que el modelo espera (la lista de features está dentro del artefacto `baseline_model.joblib`).
- Para depuración rápida, utilice `prefer_local=Local` y verifique rutas `data/processed/live_demo.parquet` y `models/baseline_model.joblib`.

Nota sobre reinicio programático
- En algunas versiones recientes de Streamlit la función `st.experimental_rerun()` puede no estar disponible. El dashboard incluye un helper `safe_rerun()` que intenta reiniciar de forma compatible, pero si tu versión no lo permite verás una advertencia y deberás refrescar la página manualmente.
- Recomendación: Mantén Streamlit actualizado en tu entorno y, si necesitas comportamiento determinista de reinicio programático durante el desarrollo, prueba con una versión estable que soporte `experimental_rerun` o usa el botón de recarga seguido de un refresh manual del navegador.

Comportamiento de la inferencia
- Cada predicción representa la probabilidad de `failure_next_24h` para una lectura. El ranking por máquina usa su lectura más reciente, en vez de tomar el máximo de todo el historial.
- La etiqueta binaria respeta `decision_threshold` del artefacto del modelo. No se altera la cantidad de positivos para forzar que coincida con la prevalencia del dataset de entrenamiento.

Advertencia de versión de scikit-learn
- Al cargar el artefacto serializado (`baseline_model.joblib`) puede aparecer la advertencia `InconsistentVersionWarning` si la versión local de scikit-learn no coincide con la usada en entrenamiento. Recomendamos regenerar el artefacto en el entorno objetivo o alinear la versión de scikit-learn si buscas reproducibilidad exacta.

Commit/PR sugeridos
- Mensaje de commit (breve): `fix(dashboard): rankea máquinas por la última predicción del modelo`
- Título PR: `Corrección de la integración del modelo con el dashboard`
- Descripción PR corta: `Usa la última lectura de cada máquina para el ranking, aplica el umbral guardado en el artefacto y valida que la clase positiva se seleccione por etiqueta. Documentación incluida.`

Contacto
- Equipo S08-26-EQUIPO-24
