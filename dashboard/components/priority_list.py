"""Componente: Lista de prioridad de mantenimiento."""

import pandas as pd
import streamlit as st


def render_priority_list(df_risk, df_machines):
    """Muestra la lista de tareas prioritarias ordenadas por impacto.

    Args:
        df_risk: DataFrame con columnas machine_id, risk_score, risk_level, criticality, priority_score, priority.
        df_machines: DataFrame con información de las máquinas.
    """
    # Asegurar consistencia en el nombre de la columna para df_risk
    if 'machineID' in df_risk.columns:
        df_risk = df_risk.rename(columns={'machineID': 'machine_id'})

    # Asegurar consistencia en el nombre de la columna para df_machines
    if 'machineID' in df_machines.columns:
        df_machines = df_machines.rename(columns={'machineID': 'machine_id'})

    # Combinar riesgo con información de la máquina
    df_merged = df_risk.merge(df_machines[['machine_id', 'type', 'location']], on='machine_id')
    df_merged = df_merged.sort_values('priority_score', ascending=False)

    for _, row in df_merged.iterrows():
        color = {
            'Crítico': '🔴',
            'Moderado': '🟡',
            'Estable': '🟢'
        }.get(row['risk_level'], '⚪')

        with st.container():
            col1, col2, col3 = st.columns([3, 2, 2])
            with col1:
                st.write(f'{color} **{row["machine_id"]}** — {row["type"]}')
                st.caption(f'Ubicación: {row["location"]}')
            with col2:
                st.write(f'Riesgo: **{row["risk_score"]:.0f}%**')
                st.caption(f'Criticidad: {row["criticality"]}')
            with col3:
                st.write(f'Acción: **{row["priority"]}**')
                st.caption(f'Prioridad: {row["priority_score"]:.0f}')
            st.divider()