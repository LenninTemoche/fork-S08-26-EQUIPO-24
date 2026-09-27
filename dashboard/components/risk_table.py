"""Matriz de riesgo por activo con accesos a sus vistas de detalle."""

import html

import pandas as pd
import streamlit as st

from components.machine_detail import navigate_to_section


def render_risk_table(df_risk, selected_status, selected_criticality):
    """Renderiza los activos filtrados y permite abrir sus vistas relacionadas."""
    if "machineID" in df_risk.columns and "machine_id" not in df_risk.columns:
        df_risk = df_risk.rename(columns={"machineID": "machine_id"})

    df_filtered = df_risk[
        df_risk["risk_level"].isin(selected_status)
        & df_risk["criticality"].isin(selected_criticality)
    ].copy()

    if df_filtered.empty:
        st.warning("No hay máquinas con los filtros aplicados.")
        return

    df_filtered["priority_score_display"] = (
        df_filtered["risk_score"] * df_filtered["priority_score"]
    ).round(1)

    headers = st.columns([0.8, 0.65, 0.85, 0.9, 0.9, 0.65, 1.8], vertical_alignment="center")
    for column, label in zip(
        headers,
        ["MÁQUINA", "RIESGO", "NIVEL", "CRITICIDAD", "ACCIÓN", "PUNTUACIÓN", "ABRIR VISTA"],
    ):
        with column:
            st.markdown(
                f"<div class='eyebrow' style='padding:.45rem 0;border-bottom:1px solid rgba(126,171,255,.2)'>{label}</div>",
                unsafe_allow_html=True,
            )

    for row_index, (_, row) in enumerate(df_filtered.iterrows()):
        machine_id = row["machine_id"]
        level = str(row["risk_level"])
        tone = "critical" if level.startswith("Cr") else "moderate" if level == "Moderado" else "stable"
        criticality = str(row["criticality"])
        action = str(row["priority"])
        score = row["priority_score_display"]
        score_text = f"{float(score):.1f}" if pd.notna(score) else "—"
        key_suffix = f"{row_index}_{html.escape(str(machine_id))}"
        with st.container(key=f"matrix_alert_{tone}_{key_suffix}"):
            cells = st.columns([0.8, 0.65, 0.85, 0.9, 0.9, 0.65, 1.8], vertical_alignment="center")

            with cells[0]:
                lamp = f"<i class='alert-lamp {tone}'></i>" if tone != "stable" else ""
                st.markdown(f"{lamp}<strong>{html.escape(str(machine_id))}</strong>", unsafe_allow_html=True)
            with cells[1]:
                risk_lamp = f"<i class='alert-lamp {tone}'></i>" if tone != "stable" else ""
                st.markdown(
                    f"<span class='matrix-risk-chip {tone}'>{risk_lamp}{float(row['risk_score']):.0f}%</span>",
                    unsafe_allow_html=True,
                )
            with cells[2]:
                st.markdown(
                    f"<span class='matrix-risk-chip {tone}'>{html.escape(level)}</span>",
                    unsafe_allow_html=True,
                )
            with cells[3]:
                st.write(criticality)
            with cells[4]:
                st.write(action)
            with cells[5]:
                st.caption(score_text)
            with cells[6]:
                actions = st.columns(2, gap="small")
                with actions[0]:
                    if st.button(
                        "Telemetría",
                        key=f"matrix_telemetry_{key_suffix}",
                        help=f"Ver telemetría del activo {machine_id}",
                        width="stretch",
                    ):
                        navigate_to_section(machine_id, "telemetry")
                with actions[1]:
                    if st.button(
                        "Diagnóstico",
                        key=f"matrix_diagnostic_{key_suffix}",
                        help=f"Ver diagnóstico del activo {machine_id}",
                        width="stretch",
                    ):
                        navigate_to_section(machine_id, "anomalies")

        if row_index < len(df_filtered) - 1:
            st.markdown(
                "<div style='border-bottom:1px solid rgba(140,144,159,.13);margin:.05rem 0 .2rem'></div>",
                unsafe_allow_html=True,
            )
