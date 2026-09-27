"""Componente visual de la matriz de riesgo por activo."""

import html

import pandas as pd
import streamlit as st


def render_risk_table(df_risk, selected_status, selected_criticality):
    """Renderiza la matriz manteniendo los filtros y valores del ranking."""
    if "machineID" in df_risk.columns:
        df_risk = df_risk.rename(columns={"machineID": "machine_id"})

    df_filtered = df_risk[
        df_risk["risk_level"].isin(selected_status)
        & df_risk["criticality"].isin(selected_criticality)
    ].copy()

    if df_filtered.empty:
        st.warning("No hay máquinas con los filtros aplicados.")
        return

    # Se conserva la puntuación que ya mostraba la matriz.
    df_filtered["priority_score_display"] = (
        df_filtered["risk_score"] * df_filtered["priority_score"]
    ).round(1)

    tone_by_level = {
        "Crítico": "critical",
        "Moderado": "moderate",
        "Estable": "stable",
    }
    action_tone = {
        "Intervenir": "critical",
        "Inspeccionar": "moderate",
        "Monitorear": "monitor",
        "Ninguna": "stable",
    }

    rows = []
    for _, row in df_filtered.iterrows():
        level = str(row["risk_level"])
        criticality = str(row["criticality"])
        action = str(row["priority"])
        tone = tone_by_level.get(level, "stable")
        action_class = action_tone.get(action, "neutral")
        priority_score = row["priority_score_display"]
        score_text = f"{float(priority_score):.1f}" if pd.notna(priority_score) else "—"

        rows.append(
            "<tr>"
            f"<td><span class='matrix-machine'>{html.escape(str(row['machine_id']))}</span></td>"
            f"<td><span class='matrix-risk'>{float(row['risk_score']):.0f}%</span></td>"
            f"<td><span class='matrix-chip {tone}'><i></i>{html.escape(level)}</span></td>"
            f"<td><span class='matrix-criticality {tone}'>{html.escape(criticality)}</span></td>"
            f"<td><span class='matrix-action {action_class}'>{html.escape(action)}</span></td>"
            f"<td><span class='matrix-score'>{score_text}</span></td>"
            "</tr>"
        )

    st.markdown(
        """
        <style>
        .risk-matrix-scroll { width:100%; overflow-x:auto; border:1px solid rgba(126,171,255,.2); border-radius:8px; }
        .risk-matrix { width:100%; min-width:690px; border-collapse:separate; border-spacing:0; color:#dae2fd; background:#0f182b; font-size:.82rem; }
        .risk-matrix thead th { padding:.75rem .85rem; color:#9da6bd; background:#171f33; border-bottom:1px solid rgba(126,171,255,.24); text-align:left; font:500 .65rem 'JetBrains Mono',monospace; letter-spacing:.07em; text-transform:uppercase; white-space:nowrap; }
        .risk-matrix tbody td { padding:.72rem .85rem; border-bottom:1px solid rgba(140,144,159,.13); vertical-align:middle; }
        .risk-matrix tbody tr:last-child td { border-bottom:0; }
        .risk-matrix tbody tr { transition:background .16s ease; }
        .risk-matrix tbody tr:hover { background:rgba(77,142,255,.08); }
        .matrix-machine,.matrix-risk,.matrix-score { font-family:'JetBrains Mono',monospace; font-variant-numeric:tabular-nums; }
        .matrix-machine { color:#dae2fd; font-weight:600; }
        .matrix-risk { color:#dae2fd; font-weight:600; }
        .matrix-score { color:#9da6bd; font-size:.76rem; }
        .matrix-chip,.matrix-criticality,.matrix-action { display:inline-flex; align-items:center; gap:.4rem; width:max-content; max-width:100%; padding:.28rem .52rem; border:1px solid transparent; border-radius:999px; font-size:.72rem; font-weight:600; white-space:nowrap; }
        .matrix-chip i { width:6px; height:6px; border-radius:50%; background:currentColor; }
        .matrix-chip.critical,.matrix-criticality.critical,.matrix-action.critical { color:#ffb4ab; background:rgba(255,180,171,.1); border-color:rgba(255,180,171,.22); }
        .matrix-chip.moderate,.matrix-criticality.moderate,.matrix-action.moderate { color:#ffb690; background:rgba(255,182,144,.1); border-color:rgba(255,182,144,.22); }
        .matrix-chip.stable,.matrix-criticality.stable,.matrix-action.stable { color:#54e18c; background:rgba(84,225,140,.09); border-color:rgba(84,225,140,.2); }
        .matrix-action.monitor { color:#a4c9ff; background:rgba(77,142,255,.1); border-color:rgba(126,171,255,.22); }
        .matrix-action.neutral { color:#c2c6d6; background:rgba(194,198,214,.08); border-color:rgba(194,198,214,.16); }
        @media (max-width:800px) { .risk-matrix { min-width:640px; } .risk-matrix thead th,.risk-matrix tbody td { padding:.62rem .7rem; } }
        </style>
        """
        + "<div class='risk-matrix-scroll'><table class='risk-matrix'><thead><tr>"
        + "<th>Máquina</th><th>Riesgo</th><th>Nivel</th><th>Criticidad</th><th>Acción</th><th>Puntuación</th>"
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )
