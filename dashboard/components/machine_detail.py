"""Componente: Detalle de máquina y histórico de errores."""

import pandas as pd
import streamlit as st


def render_machine_detail(df_errors, machine_id):
    """Muestra el histórico de errores de una máquina en un expander.

    Args:
        df_errors: DataFrame con columnas machine_id, timestamp, error_code, description.
        machine_id: ID de la máquina seleccionada.
    """
    # Asegurar consistencia en el nombre de la columna
    if 'machineID' in df_errors.columns:
        df_errors = df_errors.rename(columns={'machineID': 'machine_id'})

    df_machine_errors = df_errors[df_errors['machine_id'] == machine_id].copy()

    if df_machine_errors.empty:
        st.success(f'Sin errores registrados para {machine_id}.')
        return

    df_machine_errors = df_machine_errors.sort_values('timestamp', ascending=False)

    with st.expander(f' Ver histórico ({len(df_machine_errors)} registros)', expanded=True):
        for _, row in df_machine_errors.iterrows():
            st.error(
                f'**{row["timestamp"]}** — {row["error_code"]}\n\n'
                f'{row["description"]}'
            )