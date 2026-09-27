"""Carga de datos reales (live_demo.parquet) desde GitHub con fallback local.

El dataset contiene 87,700 filas (100 máquinas) con features:
- Sensores en tiempo real: volt, rotate, pressure, vibration
- Features de historial: errors_last_24h, time_since_last_error_h, hours_since_maintenance, etc.
- Target: failure_next_24h (1 si falla en las próximas 24h)

El modelo ML espera estas mismas features para inferencia en el dashboard.
"""

import os
import warnings

import pandas as pd
import streamlit as st
from urllib.request import Request, urlopen

from utils.model_loader import get_model, predict_probabilities

# ── Configuración ─────────────────────────────────────────────────

LIVE_DEMO_URL = (
    "https://raw.githubusercontent.com/No-Country-simulation/"
    "S08-26-EQUIPO-24/feat/modeling_integration/data/processed/"
    "live_demo.parquet"
)

LOCAL_LIVE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "processed", "live_demo.parquet",
)

# ── Auxiliares de carga ─────────────────────────────────────────────


def _load_from_url(url: str) -> pd.DataFrame:
    """Descarga y carga Parquet desde una URL HTTP."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        return pd.read_parquet(response)


def _load_from_local(path: str) -> pd.DataFrame:
    """Carga Parquet desde el sistema de archivos local."""
    return pd.read_parquet(path)


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_demo_data(prefer_local: bool = False) -> tuple[pd.DataFrame, str]:
    """Carga live_demo.parquet desde GitHub con fallback local.

    Estrategia:
    1. Intenta descargar desde GitHub (fuente de verdad en producción).
    2. Si falla, usa la copia local `data/processed/live_demo.parquet`.

    Returns:
        (DataFrame con las lecturas, descripción de la fuente utilizada).
    """
    if prefer_local:
        local = os.path.normpath(LOCAL_LIVE_PATH)
        if os.path.exists(local):
            df = _load_from_local(local)
            source = f"Local ({local})"
        else:
            raise RuntimeError(
                "Se solicitó modo local pero no se encontró live_demo.parquet en: "
                f"{LOCAL_LIVE_PATH}."
            )
    else:
        try:
            df = _load_from_url(LIVE_DEMO_URL)
            source = "GitHub (feat/modeling_integration)"
        except Exception:
            local = os.path.normpath(LOCAL_LIVE_PATH)
            if os.path.exists(local):
                df = _load_from_local(local)
                source = f"Local ({local})"
            else:
                raise RuntimeError(
                    "No se pudo cargar live_demo.parquet desde GitHub ni desde "
                    f"{LOCAL_LIVE_PATH}. Verifica la conexión a internet."
                )

    # Convertir datetime a pd.Timestamp para consistencia
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Normalizar nombres de columnas: aceptar 'machineID' o 'machine_id'
    if 'machineID' in df.columns and 'machine_id' not in df.columns:
        df = df.rename(columns={'machineID': 'machine_id'})

    # Verificaciones esenciales del schema esperado por el modelo
    required_cols = {'datetime', 'machine_id'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"live_demo.parquet no contiene las columnas requeridas: {missing}"
        )

    return df, source


# ── Computación de riesgo basada en modelo ─────────────────────────────────────────────


def _extract_machine_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae últimos metadatos por máquina desde live_df.

    Retorna DataFrame con:
        machine_id, last_maintenance, days_since_maintenance,
        operating_hours, type, location
    """
    latest = df.sort_values('datetime').groupby('machine_id').tail(1)

    # Asignar type/location por machineID para la demo (simulado)
    type_map = {
        1: 'CNC', 2: 'CNC', 3: 'Torno', 4: 'Fresadora', 5: 'Bomba',
        6: 'Transportador', 7: 'Compresor', 8: 'Robot', 9: 'Pulidor',
        10: 'Lijadora', 11: 'Taladradora', 12: 'Niveladora', 13: 'Cnc',
        14: 'Sierra', 15: 'Pulverizadora', 16: 'Elevador', 17: 'Balanza',
        18: 'Detector', 19: 'Sensor', 20: 'Medidor'
    }
    location_map = {
        1: 'Planta A', 2: 'Planta B', 3: 'Planta C', 4: 'Planta D',
        5: 'Planta E', 6: 'Planta F', 7: 'Planta G', 8: 'Planta H',
        9: 'Planta I', 10: 'Planta J'
    }

    # Simplificar: crear DataFrame directamente
    # Mapear ids numéricos para elegir tipo/ubicación de demo
    numeric_ids = pd.to_numeric(latest['machine_id'], errors='coerce').fillna(0).astype(int)
    machines_df = pd.DataFrame({
        'machine_id': latest['machine_id'].astype(str).tolist(),
        'type': numeric_ids.map(type_map).fillna('Tipo no especificado').tolist(),
        'location': numeric_ids.map(location_map).fillna('Ubicación no especificada').tolist(),
        'operating_hours': [8000 + (m * 150) % 2000 for m in numeric_ids.tolist()],
        'last_maintenance': [(pd.Timestamp('2026-01-01') - pd.Timedelta(days=float(hours) * 0.1)).strftime('%Y-%m-%d') 
                            for hours in latest['hours_since_maintenance']],
        'days_since_maintenance': latest['days_since_maintenance'].tolist(),
        'next_maintenance': [pd.Timestamp('2026-01-01') + pd.Timedelta(days=30)] * len(latest),
    })

    return machines_df


def compute_risk_from_model(live_df: pd.DataFrame, prefer_local_model: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ejecuta inferencia del modelo sobre live_df y retorna df_risk + metadatos.

    Args:
        live_df: DataFrame con las 46 features esperadas por el modelo.

    Returns:
        df_machines: Información por máquina para el sidebar.
        df_risk: Ranking de riesgo (machineID, risk_score, risk_level, criticality, priority_score, priority).
        df_telemetry: series de telemetría por máquina (para charts).
        df_errors: histórico de errores simulados para machine_detail.
    """
    # Cargar el modelo (cached por st.cache_data)
    model, feature_cols, meta, _model_source = get_model(prefer_local=prefer_local_model)
    threshold = meta["decision_threshold"]

    # Asegurar nombres de columnas normales (por ejemplo live_df puede venir con 'machine_id')
    if 'machineID' in live_df.columns and 'machine_id' not in live_df.columns:
        live_df = live_df.rename(columns={'machineID': 'machine_id'})

    # Validar columnas
    missing = [c for c in feature_cols if c not in live_df.columns]
    if missing:
        raise ValueError(f"Faltan features requeridas por el modelo: {missing}")

    # Predicción del modelo (probabilidades)
    probs = predict_probabilities(model, feature_cols, live_df)

    live_df = live_df.copy()
    live_df['failure_probability'] = probs
    live_df['prediction'] = (probs >= threshold).astype('int8')

    # ── df_machines: metadatos ──────────────────────────────────────
    df_machines = _extract_machine_metadata(live_df)

    # ── df_risk: ranking por máquina ───────────────────────────────
    # Cada fila es una lectura temporal: el ranking debe representar el estado
    # más reciente, no el máximo de cientos de predicciones históricas.
    latest = (
        live_df.sort_values(['datetime', 'machine_id'])
        .groupby('machine_id', sort=False)
        .tail(1)
    )
    risk = latest[[
        'machine_id', 'failure_probability', 'prediction',
        'hours_since_maintenance', 'errors_last_24h',
        'time_since_last_error_h',
    ]].rename(columns={
        'failure_probability': 'risk_score',
        'hours_since_maintenance': 'hours_since_last_maintenance',
        'errors_last_24h': 'recent_errors',
    }).copy()

    # Calcular risk_score normalizado (0-100)
    risk['risk_score'] = (risk['risk_score'] * 100).round(2)

    # Determinar risk_level y criticality
    # Mapear score numérico a niveles usados por el dashboard
    # El dashboard espera las etiquetas: 'Crítico', 'Moderado', 'Estable'
    def get_level_and_criticality(score):
        if score < 30:
            return ('Estable', 'Baja')
        elif score < 60:
            return ('Moderado', 'Media')
        else:
            return ('Crítico', 'Alta')

    levels_criticalities = risk['risk_score'].apply(get_level_and_criticality)
    risk['risk_level'] = levels_criticalities.apply(lambda x: x[0])
    risk['criticality'] = levels_criticalities.apply(lambda x: x[1])

    # Calcular priority_score (riesgo × criticidad × impacto)
    crit_map = {'Baja': 1.0, 'Media': 2.0, 'Alta': 3.0}
    risk['priority_score'] = (
        risk['risk_score'] * 
        risk['criticality'].map(crit_map) * 
        (1 + risk['recent_errors'] * 0.5)
    ).round(2)

    # Asignar priority
    def assign_priority(row):
        if row['risk_level'] == 'Crítico':
            return 'Intervenir'
        elif row['risk_level'] == 'Moderado':
            return 'Inspeccionar'
        elif row['recent_errors'] > 2:
            return 'Monitorear'
        else:
            return 'Ninguna'

    risk['priority'] = risk.apply(assign_priority, axis=1)

    # Reordenar columns
    df_risk = risk[['machine_id', 'risk_score', 'risk_level', 'criticality', 'priority_score', 'priority']]

    # Normalizar tipo: usar string en todos los IDs de máquina para evitar
    # errores al hacer merges entre dataframes con dtypes distintos.
    df_risk['machine_id'] = df_risk['machine_id'].astype(str)
    df_machines['machine_id'] = df_machines['machine_id'].astype(str)

    # ── df_telemetry: series por máquina ─────────────────────────────
    # Telemetry: normalizar columnas para los componentes
    tele_cols = ['datetime', 'machine_id', 'volt', 'vibration', 'pressure']
    # Seleccionar y renombrar a lo que esperan los componentes
    df_telemetry = live_df[tele_cols].rename(columns={
        'datetime': 'timestamp',
        'volt': 'temperature',
        # 'vibration' y 'pressure' mantienen su nombre
    })
    # Asegurar tipos
    df_telemetry['timestamp'] = pd.to_datetime(df_telemetry['timestamp'])
    df_telemetry['machine_id'] = df_telemetry['machine_id'].astype(str)

    # ── df_errors: historial de errores simulados ─────────────────────
    # Para la demo, generar un historial realista basado en recent_errors
    errors_rows = []
    for _, row in risk.iterrows():
        if row['recent_errors'] > 0:
            # Generar 1-3 errores aleatorios para máquinas con fallas recientes
            for i in range(int(row['recent_errors'])):
                mid = row['machine_id']
                # intentar convertir a int para componer el código
                try:
                    mid_int = int(mid)
                except Exception:
                    mid_int = 0
                errors_rows.append({
                    'machine_id': mid,
                    'timestamp': pd.Timestamp('2026-01-01') - pd.Timedelta(hours=i * 24),
                    'error_code': f'E{100 + mid_int * 10}',
                    'description': f'Error de sensor {i+1} en máquina {mid}',
                })

    df_errors = pd.DataFrame(errors_rows)
    if not df_errors.empty and 'machine_id' in df_errors.columns:
        df_errors['machine_id'] = df_errors['machine_id'].astype(str)
    if df_errors.empty:
        # Generar errores dummy para al menos 2 máquinas para la demo
        for m in [1, 2]:
            df_errors = pd.concat([
                df_errors,
                pd.DataFrame([{
                    'machine_id': m,
                    'timestamp': pd.Timestamp('2026-01-01') - pd.Timedelta(hours=24),
                    'error_code': f'E{100 + m * 10}',
                    'description': f'Error de prueba en máquina {m}',
                }])
            ], ignore_index=True)
        if 'machine_id' in df_errors.columns:
            df_errors['machine_id'] = df_errors['machine_id'].astype(str)

    return df_machines, df_risk, df_telemetry, df_errors


# ── Función wrapper (para compatibilidad) ─────────────────────────────────────


def load_mock_data():
    """Mantiene compatibilidad con el código existente del dashboard skeleton.
    Útil si otros módulos aún importan load_mock_data.
    """
    warnings.warn(
        "load_mock_data() es un placeholder. Usa load_live_demo_data() y compute_risk_from_model() para el dashboard real.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Generar mock básico para referencias rápidas
    return (
        pd.DataFrame({'machine_id': ['CNC-001', 'CNC-002']}),
        pd.DataFrame({'machine_id': ['CNC-001'], 'risk_score': [50], 'risk_level': ['Moderado'], 'criticality': ['Media'], 'priority_score': [100], 'priority': ['Inspeccionar']}),
        pd.DataFrame({'machine_id': ['CNC-001'], 'temperature': [70], 'timestamp': [pd.Timestamp('2026-01-01')]}),
        pd.DataFrame({'machine_id': ['CNC-001'], 'error_code': ['E100'], 'description': ['Error test']}),
    )
