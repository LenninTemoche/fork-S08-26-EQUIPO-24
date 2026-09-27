"""Detalle de una maquina y enlace al diagnostico de flota."""

import pandas as pd
import streamlit as st


def navigate_to_section(machine_id, section_id):
    """Open a dashboard section with the requested asset selected."""
    st.session_state.active_section = section_id
    st.session_state.pending_selected_machine = str(machine_id)
    st.rerun(scope="app")


def navigate_to_diagnostic_matrix(machine_id):
    """Open Vista General from an error card."""
    navigate_to_section(machine_id, "overview")


def render_machine_detail(df_errors, machine_id):
    """Show the selected machine events and offer a shortcut to its matrix row."""
    if "machineID" in df_errors.columns and "machine_id" not in df_errors.columns:
        df_errors = df_errors.rename(columns={"machineID": "machine_id"})

    if "machine_id" in df_errors.columns:
        df_errors = df_errors[df_errors["machine_id"].astype(str) == str(machine_id)].copy()
    else:
        df_errors = pd.DataFrame(columns=["machine_id", "timestamp", "error_code", "description"])

    if not df_errors.empty:
        df_errors = df_errors.sort_values("timestamp", ascending=False)

    with st.expander(f"Eventos y errores ({len(df_errors)} registros)", expanded=True):
        if df_errors.empty:
            st.success(f"Sin errores registrados para {machine_id}.")
        else:
            for _, row in df_errors.iterrows():
                st.error(
                    f"**{row['timestamp']} | {row['error_code']}**\n\n"
                    f"{row['description']}"
                )

        action_cols = st.columns(2)
        with action_cols[0]:
            if st.button(
                "Ver activo en matriz diagnóstica",
                key=f"open_matrix_{machine_id}",
                width="stretch",
            ):
                navigate_to_diagnostic_matrix(machine_id)
        with action_cols[1]:
            if st.button(
                "Abrir diagnóstico",
                key=f"open_anomalies_{machine_id}",
                width="stretch",
            ):
                st.session_state.pending_selected_machine = str(machine_id)
                st.session_state.active_section = "anomalies"
                st.rerun(scope="app")
