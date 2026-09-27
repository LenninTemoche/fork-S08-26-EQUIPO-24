"""Componente: Gráfico de telemetría de sensores."""

import pandas as pd
import streamlit as st


def render_sensor_chart(df_telemetry, machine_id):
    """Muestra un gráfico de línea con la telemetría de una máquina.

    Args:
        df_telemetry: DataFrame con columnas machine_id, timestamp, temperature, vibration, pressure.
        machine_id: ID de la máquina seleccionada.
    """
    # Asegurar consistencia en el nombre de la columna
    if 'machineID' in df_telemetry.columns:
        df_telemetry = df_telemetry.rename(columns={'machineID': 'machine_id'})

    df_machine = df_telemetry[df_telemetry['machine_id'] == machine_id].copy()

    if df_machine.empty:
        st.warning(f'No hay telemetría para {machine_id}.')
        return

    df_machine = df_machine.sort_values('timestamp')

    # Selector de variable
    sensor = st.selectbox(
        ' Variable a graficar',
        options=['temperature', 'vibration', 'pressure'],
        index=0,
        key=f'sensor_{machine_id}'
    )

    # Gráfico de línea nativo de Streamlit
    st.line_chart(
        df_machine.set_index('timestamp')[sensor],
        use_container_width=True,
        color='#ff4b4b'
    )

    # Mostrar datos tabulares del último registro
    with st.expander(' Ver último registro'):
        latest = df_machine.iloc[-1]
        st.write(f'**Fecha:** {latest["timestamp"].strftime("%Y-%m-%d %H:%M")}')
        st.write(f'**Temperatura:** {latest["temperature"]:.1f} °C')
        st.write(f'**Vibración:** {latest["vibration"]:.2f} mm/s')
        st.write(f'**Presión:** {latest["pressure"]:.1f} bar')