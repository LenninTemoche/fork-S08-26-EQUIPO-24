"""Componente: Tabla de ranking de riesgo."""

import pandas as pd
import streamlit as st


def render_risk_table(df_risk, selected_status, selected_criticality):
    """Muestra la tabla interactiva de ranking de riesgo con colores por nivel.

    Args:
        df_risk: DataFrame con columnas machine_id, risk_score, risk_level, criticality, priority.
        selected_status: Lista de niveles de riesgo filtrados.
        selected_criticality: Lista de criticidad filtrada.
    """
    # Asegurar consistencia en el nombre de la columna
    if 'machineID' in df_risk.columns:
        df_risk = df_risk.rename(columns={'machineID': 'machine_id'})

    # Filtrar por estado y criticidad
    df_filtered = df_risk[
        df_risk['risk_level'].isin(selected_status) &
        df_risk['criticality'].isin(selected_criticality)
    ].copy()

    if df_filtered.empty:
        st.warning('No hay máquinas con los filtros aplicados.')
        return

    # Formatear score como porcentaje
    df_filtered['Riesgo %'] = df_filtered['risk_score'].apply(lambda x: f'{x:.0f}%')

    # Renombrar columnas para display
    df_display = df_filtered[['machine_id', 'Riesgo %', 'risk_level', 'criticality', 'priority']].copy()
    df_display.columns = ['Máquina', 'Riesgo', 'Nivel', 'Criticidad', 'Acción']

    # Aplicar color por nivel de riesgo
    def color_risk(val):
        if val == 'Crítico':
            return 'background-color: #ffcccc; color: #b30000; font-weight: bold'
        elif val == 'Moderado':
            return 'background-color: #fff2cc; color: #7f6000'
        elif val == 'Estable':
            return 'background-color: #ccffcc; color: #006600'
        return ''

    df_styled = df_display.style.map(color_risk, subset=['Nivel'])

    st.dataframe(
        df_styled,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Máquina': st.column_config.TextColumn('Máquina', width='medium'),
            'Riesgo': st.column_config.TextColumn('Riesgo', width='small'),
            'Nivel': st.column_config.TextColumn('Nivel', width='medium'),
            'Criticidad': st.column_config.TextColumn('Criticidad', width='medium'),
            'Acción': st.column_config.TextColumn('Acción', width='medium'),
        }
    )