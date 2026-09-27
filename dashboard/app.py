import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.machine_detail import render_machine_detail
from components.sensor_chart import render_sensor_chart
from components.risk_table import render_risk_table
from components.demo_simulator import (
    PERIOD_HOURS,
    render_demo_simulator,
    reset_simulation_state,
    simulation_snapshot,
)
from utils.data_loader import compute_risk_from_model, get_priority_machine, load_live_demo_data
from utils.model_loader import get_model


st.set_page_config(
    page_title="Mantenimiento predictivo",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
    :root { --bg:#0b1326; --surface:#111a2d; --surface-2:#171f33; --surface-3:#222a3d; --line:#424754; --text:#dae2fd; --muted:#9da6bd; --blue:#7eabff; --blue-strong:#4d8eff; --orange:#ffb690; --red:#ffb4ab; --green:#54e18c; }
    html, body, [class*="css"], [data-testid="stAppViewContainer"] { font-family:Inter,sans-serif; }
    [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--text); }
    [data-testid="stHeader"] { display:none; }
    [data-testid="stMainBlockContainer"] { max-width:none; padding:.65rem 1.2rem 2rem; }
    [data-testid="stMainBlockContainer"] > div[data-testid="stVerticalBlock"] { gap:.55rem; }
    [data-testid="stSidebar"] { min-width:19rem!important; width:19rem!important; background:linear-gradient(180deg,#171f33 0%,#0b1326 100%); border-right:1px solid rgba(126,171,255,.2); }
    [data-testid="stSidebar"] > div:first-child { padding:.3rem .8rem; }
    [data-testid="stSidebarHeader"] { height:1.5rem!important; min-height:1.5rem!important; margin-bottom:0!important; }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap:.45rem; }
    [data-testid="stMetric"] { background:var(--surface-2); border:1px solid rgba(126,171,255,.15); border-radius:8px; padding:1rem; }
    [data-testid="stMetricLabel"] { color:var(--muted); font-family:'JetBrains Mono',monospace; font-size:.68rem; text-transform:uppercase; letter-spacing:.06em; }
    [data-testid="stMetricValue"] { color:var(--text); font-family:'JetBrains Mono',monospace; font-weight:700; }
    [data-testid="stMetricDelta"] { font-family:'JetBrains Mono',monospace; font-size:.72rem; }
    [data-testid="stTabs"] [role="tablist"] { gap:.25rem; padding:.35rem; background:var(--surface-2); border:1px solid rgba(126,171,255,.15); border-radius:8px; }
    [data-testid="stTabs"] button[role="tab"] { color:var(--muted); border-radius:6px; font-weight:600; }
    [data-testid="stTabs"] button[role="tab"][aria-selected="true"] { color:#001a42; background:var(--blue-strong); }
    [data-testid="stVerticalBlockBorderWrapper"] { border-color:rgba(126,171,255,.14); background:rgba(23,31,51,.72); border-radius:8px; }
    [data-testid="stSidebar"] button[kind="primary"] { background:linear-gradient(100deg,#4d8eff,#70a4ff); color:#001a42; border:1px solid #91b8ff; box-shadow:0 0 0 1px rgba(77,142,255,.18),0 4px 12px rgba(0,0,0,.18); font-weight:700; }
    [data-testid="stSidebar"] button[kind="secondary"] { background:rgba(34,42,61,.62); color:#dae2fd; border:1px solid rgba(126,171,255,.18); text-align:left; }
    [data-testid="stSidebar"] button[kind="secondary"]:hover { background:rgba(77,142,255,.18); color:#fff; border-color:rgba(126,171,255,.55); }
    [data-testid="stSidebar"] [data-testid="stButton"] button { min-height:2.45rem; border-radius:6px; }
    [data-testid="stSidebar"] [data-testid="stSelectbox"], [data-testid="stSidebar"] [data-testid="stMultiSelect"] { margin-bottom:.2rem; }
    [data-testid="stDataFrame"] { border:1px solid rgba(126,171,255,.14); }
    h1, h2, h3 { font-family:'Space Grotesk',sans-serif!important; letter-spacing:0!important; }
    h1 { font-size:2rem!important; color:var(--text)!important; } h2 { font-size:1.3rem!important; } h3 { font-size:1.05rem!important; }
    .mono { font-family:'JetBrains Mono',monospace; } .eyebrow { color:var(--muted); font:500 .68rem 'JetBrains Mono',monospace; letter-spacing:.1em; text-transform:uppercase; }
    .brand { color:var(--blue); font:700 1.15rem 'Space Grotesk',sans-serif; }
    .topbar { position:relative; z-index:1; height:3.5rem; display:flex; align-items:center; justify-content:space-between; padding:0; margin-bottom:1rem; background:transparent; border-bottom:1px solid rgba(126,171,255,.15); }
    .status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--green); box-shadow:0 0 8px var(--green); margin-right:.35rem; }
    .banner { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:1rem 1.2rem; background:var(--surface-2); border:1px solid rgba(126,171,255,.14); border-radius:8px; margin-bottom:1rem; }
    .banner-title { color:var(--text); font:600 1.2rem 'Space Grotesk',sans-serif; } .banner-copy { color:var(--muted); font-size:.82rem; margin-top:.25rem; }
    .pill { display:inline-block; padding:.28rem .5rem; border-radius:4px; color:var(--blue); background:rgba(77,142,255,.16); font:600 .65rem 'JetBrains Mono',monospace; letter-spacing:.04em; }
    .pill-red { color:var(--red); background:rgba(255,80,70,.16); } .pill-green { color:var(--green); background:rgba(84,225,140,.12); } .pill-yellow { color:#ffd166; background:rgba(255,209,102,.16); }
    .ai-card { padding:1rem; border:1px solid rgba(126,171,255,.3); border-radius:8px; background:linear-gradient(110deg,rgba(77,142,255,.18),rgba(23,31,51,.8)); }
    .ai-card p { color:var(--muted); font-size:.84rem; margin:.35rem 0 0; }
    .section-head { display:flex; align-items:end; justify-content:space-between; gap:1rem; margin:1.25rem 0 .8rem; } .section-head h2 { margin:0; } .section-head p { color:var(--muted); margin:.25rem 0 0; font-size:.8rem; line-height:1.4; }
    .stitch-kpi { display:flex; flex-direction:column; justify-content:space-between; height:100%; min-height:7.1rem; padding:.9rem 1rem; background:linear-gradient(150deg,rgba(34,42,61,.9),rgba(17,26,45,.92)); border:1px solid rgba(126,171,255,.16); border-bottom:2px solid var(--accent); border-radius:8px; box-sizing:border-box; transition:border-color .18s ease,transform .18s ease; }
    .stitch-kpi:hover { border-color:var(--accent); transform:translateY(-1px); }
    .stitch-kpi-label { display:flex; align-items:center; justify-content:space-between; gap:.45rem; min-height:1.8rem; color:var(--muted); font:500 .66rem 'JetBrains Mono',monospace; letter-spacing:.06em; }
    .stitch-kpi-label > span:first-child { min-width:0; }
    .stitch-kpi-label .pill { flex:none; float:none!important; }
    .stitch-kpi-value { display:flex; align-items:center; min-height:2rem; margin:.45rem 0 .35rem; color:var(--text); font:700 1.55rem/1.1 'JetBrains Mono',monospace; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .stitch-kpi-detail { min-height:1rem; color:var(--muted); font:500 .7rem 'JetBrains Mono',monospace; line-height:1.35; }
    .risk-legend { display:flex; justify-content:flex-end; flex-wrap:wrap; gap:.5rem 1rem; margin:.15rem 0 .6rem; color:var(--muted); font:500 .67rem 'JetBrains Mono',monospace; }
    .risk-legend span { display:inline-flex; align-items:center; gap:.35rem; }
    .risk-legend b { display:inline-block; width:9px; height:9px; border-radius:2px; }
    .telemetry-head { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:1rem; margin:1.1rem 0 .8rem; background:var(--surface-2); border:1px solid rgba(126,171,255,.14); border-radius:8px; }
    .telemetry-head > div:first-child { display:flex; align-items:center; gap:.8rem; }
    .telemetry-head h2 { margin:0; font-size:1.2rem!important; } .telemetry-head p { margin:.25rem 0 0; color:var(--muted); font-size:.75rem; }
    .machine-tag { padding:.55rem .7rem; color:var(--red); background:rgba(255,180,144,.12); font:700 1.2rem 'JetBrains Mono',monospace; border-radius:3px; }
    .telemetry-head .pill + .pill { margin-left:.4rem; } .chart-label { display:flex; justify-content:space-between; align-items:center; padding:.55rem .1rem; color:var(--text); font:600 .7rem 'JetBrains Mono',monospace; }
    .anomaly-grid { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:.45rem; }
    .heat-cell { display:flex; flex-direction:column; gap:.35rem; min-height:5.7rem; padding:.55rem; background:var(--surface-2); border:1px solid rgba(126,171,255,.1); border-radius:5px; color:var(--muted); font:500 .58rem 'JetBrains Mono',monospace; text-align:center; }
    .heat-cell span { padding:.25rem .15rem; border-radius:2px; background:rgba(84,225,140,.12); } .heat-cell span.red { background:rgba(255,180,171,.16); } .heat-cell span.orange { background:rgba(255,182,144,.16); } .heat-cell strong { font-size:.68rem; } .red { color:var(--red); } .orange { color:var(--orange); } .green { color:var(--green); }
    .sidebar-card { padding:.85rem; border:1px solid rgba(126,171,255,.18); border-radius:8px; background:rgba(34,42,61,.6); margin:.65rem 0; }
    .source-card,.sidebar-info-card { padding:.65rem .75rem; border:1px solid rgba(126,171,255,.24); border-left:3px solid var(--blue); border-radius:6px; background:rgba(34,42,61,.58); margin:.55rem 0; }
    .source-card { border-color:rgba(84,225,140,.24); border-left-color:var(--green); background:rgba(24,57,54,.32); }
    .source-value { color:var(--text); font:600 .88rem 'Space Grotesk',sans-serif; margin:.35rem 0 .2rem; }
    .source-note { color:var(--muted); font-size:.65rem; line-height:1.35; }
    .sidebar-heading { margin:.7rem 0 .25rem; }
    [data-testid="stSidebar"] .st-key-sidebar_filters { padding:.55rem .7rem!important; border:1px solid rgba(126,171,255,.2); border-radius:6px; background:rgba(34,42,61,.42); }
    .st-key-sidebar_filters [data-testid="stButtonGroup"] > div { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.25rem; }
    .st-key-sidebar_filters [data-testid="stButtonGroup"] button { width:100%; min-width:0; min-height:1.8rem; padding:.2rem .35rem; border-radius:5px; font-size:.68rem; }
    .st-key-sidebar_filters [data-testid="stButtonGroup"] button[aria-pressed="true"] { border-color:rgba(126,171,255,.5); background:rgba(77,142,255,.26); color:var(--text); }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] { font-size:.8rem; }
    [data-testid="stSidebar"] hr { margin:.35rem 0; }
    [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"]:has(.topbar), [data-testid="stMainBlockContainer"] [data-testid="stMarkdownContainer"]:has(.banner) { margin-bottom:-.25rem; }
    .topbar { margin-bottom:.2rem; }
    .banner { gap:.65rem; padding:.7rem 1rem; margin-bottom:.25rem; }
    .banner > div:first-child { min-width:0; flex:1; }
    .banner-title { display:flex; align-items:center; flex-wrap:wrap; gap:.4rem; }
    .st-key-fleet_ai_panel { padding:.8rem .95rem!important; border-color:rgba(126,171,255,.24)!important; background:linear-gradient(115deg,rgba(23,31,51,.95),rgba(17,26,45,.9))!important; }
    .ai-panel-heading { display:flex; align-items:center; justify-content:space-between; gap:.75rem; }
    .ai-panel-heading::before { content:"\\2726"; display:grid; place-items:center; width:2.35rem; height:2.35rem; flex:none; border:1px solid rgba(126,171,255,.24); border-radius:7px; background:rgba(77,142,255,.14); color:var(--blue); font-size:1.2rem; }
    .ai-panel-heading > div:first-child { flex:1; min-width:0; }
    .ai-panel-heading > div:first-child::before { content:"COPILOTO PREDICTIVO"; display:block; margin-bottom:.12rem; color:var(--muted); font:500 .62rem 'JetBrains Mono',monospace; letter-spacing:.1em; }
    .ai-panel-heading strong { display:block; color:var(--text); font:600 1rem 'Space Grotesk',sans-serif; }
    .ai-panel-summary { margin:.45rem 0 0 3.05rem; color:var(--muted); font-size:.76rem; line-height:1.4; }
    .ai-diagnostic { margin-top:.7rem; padding-top:.65rem; border-top:1px solid rgba(126,171,255,.2); }
    .ai-diagnostic-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); align-items:stretch; gap:.65rem; margin-top:.5rem; }
    .ai-result-card { display:flex; flex-direction:column; min-width:0; min-height:8rem; height:100%; padding:.75rem .8rem; border:1px solid rgba(126,171,255,.16); border-left:3px solid var(--card-accent,rgba(126,171,255,.4)); border-radius:7px; background:linear-gradient(180deg,rgba(34,42,61,.66),rgba(34,42,61,.4)); box-sizing:border-box; }
    .ai-result-card .eyebrow { color:var(--card-accent,var(--muted)); }
    .ai-result-title { display:block; margin-top:.35rem; color:var(--text); font:600 .82rem 'Space Grotesk',sans-serif; overflow-wrap:anywhere; }
    .ai-result-card p,.ai-result-card li { color:var(--muted); font-size:.74rem; line-height:1.4; overflow-wrap:anywhere; }
    .ai-result-card p { margin:.35rem 0 0; }
    .ai-result-card ul { margin:.3rem 0 0; padding-left:1rem; flex:1; }
    .ai-analysis-card { padding:.8rem .85rem; border:1px solid rgba(126,171,255,.2); border-left:3px solid var(--tone,var(--blue)); border-radius:7px; background:linear-gradient(150deg,rgba(77,142,255,.13),rgba(34,42,61,.62)); margin:.55rem 0; box-shadow:0 8px 20px rgba(0,0,0,.12); }
    .ai-analysis-head { display:flex; align-items:center; justify-content:space-between; gap:.4rem; }
    .ai-analysis-head .eyebrow { color:var(--tone,var(--blue)); }
    .ai-analysis-verdict { margin:.5rem 0 .25rem; color:var(--text); font:600 .94rem 'Space Grotesk',sans-serif; }
    .ai-analysis-copy { margin:.2rem 0 .45rem; color:var(--muted); font-size:.72rem; line-height:1.45; }
    .ai-analysis-card .meta-row { border-top:1px solid rgba(126,171,255,.1); padding:.35rem 0; }
    .st-key-risk_distribution_panel,.st-key-risk_matrix_panel { padding:.9rem 1rem!important; border-color:rgba(126,171,255,.18)!important; background:linear-gradient(150deg,rgba(23,31,51,.68),rgba(17,26,45,.72))!important; }
    .st-key-live_telemetry_panel,.st-key-demo_telemetry_section { padding:.9rem 1rem!important; border-color:rgba(126,171,255,.18)!important; background:linear-gradient(150deg,rgba(23,31,51,.68),rgba(17,26,45,.72))!important; }
    .st-key-risk_distribution_panel .section-head,.st-key-risk_matrix_panel .section-head { margin:.05rem 0 .7rem; }
    .maintenance-hero { border-left-color:var(--state-color); }
    .maintenance-hero strong { color:var(--state-color); }
    .meta-row { display:flex; justify-content:space-between; gap:.5rem; padding:.25rem 0; color:var(--muted); font:.7rem 'JetBrains Mono',monospace; } .meta-row strong { color:var(--text); text-align:right; font-weight:500; }
    .priority { border-left:3px solid var(--red); padding:.9rem 1rem; background:var(--surface-2); border-radius:0 8px 8px 0; margin:.5rem 0; } .priority-title { display:flex; align-items:center; gap:.55rem; font-weight:700; color:var(--text); } .priority-copy { color:var(--muted); font-size:.82rem; margin-top:.25rem; } .priority-meta { display:flex; flex-wrap:wrap; gap:.8rem; margin-top:.5rem; color:var(--muted); font:.68rem 'JetBrains Mono',monospace; } .rank-badge { display:inline-grid; place-items:center; width:1.8rem; height:1.8rem; border-radius:5px; color:#061126; font:700 1rem 'JetBrains Mono',monospace; flex:none; }
    .maintenance-hero { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:1.15rem; border:1px solid rgba(126,171,255,.3); border-left:4px solid var(--red); border-radius:8px; background:linear-gradient(100deg,rgba(2,103,184,.55),rgba(23,31,51,.86)); }
    .maintenance-hero h2 { margin:.35rem 0; font-size:1.35rem!important; } .maintenance-hero p { color:#d6e5ff; margin:0; font-size:.82rem; } .maintenance-hero strong { color:var(--red); }
    .resource-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.75rem; margin-top:1rem; } .resource-card { display:flex; gap:.75rem; align-items:center; padding:.85rem; background:var(--surface-2); border:1px solid rgba(126,171,255,.14); border-radius:8px; } .resource-icon { width:2.2rem; height:2.2rem; display:grid; place-items:center; border-radius:6px; background:rgba(77,142,255,.18); color:var(--blue); font-size:1.15rem; } .resource-card strong { display:block; margin-top:.2rem; color:var(--text); font-size:.84rem; } .resource-card span { color:var(--muted); font:.65rem 'JetBrains Mono',monospace; }
    @media (max-width:900px) { .ai-diagnostic-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .ai-result-card:last-child { grid-column:1/-1; } }
    @media (max-width:600px) { .ai-diagnostic-grid { grid-template-columns:1fr; } .ai-result-card:last-child { grid-column:auto; } }
    @media (max-width:800px) { [data-testid="stMainBlockContainer"] { padding:.75rem 1rem 1.5rem; } .banner { align-items:flex-start; flex-direction:column; } .risk-legend { justify-content:flex-start; flex-wrap:wrap; } .stitch-kpi-value { font-size:1.35rem; } .telemetry-head { align-items:flex-start; flex-direction:column; } .anomaly-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } .resource-grid { grid-template-columns:1fr; } .maintenance-hero { align-items:flex-start; flex-direction:column; } .topbar > div:last-child { display:none; } }
    @media (max-width:800px) { .ai-panel-summary { margin-left:0; } .st-key-risk_distribution_panel,.st-key-risk_matrix_panel { padding:.7rem!important; } .stitch-kpi { min-height:6.6rem; padding:.75rem; } .stitch-kpi-label { font-size:.61rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    st.experimental_rerun()


SECTION_OPTIONS = [
    ("overview", "1. Vista General"),
    ("telemetry", "2. Telemetría de Demo"),
    ("anomalies", "3. Deteccion de Anomalias"),
    ("maintenance", "4. Plan de Mantenimiento"),
]


if "active_section" not in st.session_state:
    st.session_state.active_section = "overview"


def risk_figure(df_risk):
    chart = df_risk.sort_values("risk_score", ascending=True).copy()
    colors = {"Cr\u00edtico":"#ffb4ab", "Moderado":"#ffb690", "Estable":"#54e18c"}
    fig = go.Figure(go.Bar(
        x=chart["risk_score"], y=chart["machine_id"], orientation="h",
        marker_color=[colors.get(level, "#7eabff") for level in chart["risk_level"]],
        text=[f"{value:.0f}%" for value in chart["risk_score"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Riesgo: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(6,14,32,.65)",
        font={"family":"JetBrains Mono", "color":"#dae2fd", "size":11},
        xaxis={"range":[0,110], "gridcolor":"rgba(140,144,159,.18)", "title":"\u00cdndice de riesgo (%)"},
        yaxis={"gridcolor":"rgba(0,0,0,0)"}, margin={"l":15,"r":45,"t":15,"b":45},
        height=max(280, len(chart) * 52), showlegend=False,
    )
    fig.add_vline(x=60, line_dash="dash", line_color="#ffb4ab", annotation_text="Umbral cr\u00edtico 60%", annotation_font_color="#ffb4ab")
    return fig


def render_stitch_kpi(label, value, detail, tone="blue", badge=""):
    tone_color = {"blue": "#7eabff", "red": "#ffb4ab", "orange": "#ffb690", "green": "#54e18c"}.get(tone, "#7eabff")
    badge_html = f"<span class='pill' style='color:{tone_color}'>{html.escape(badge)}</span>" if badge else ""
    return f"<div class='stitch-kpi' style='--accent:{tone_color}'><div class='stitch-kpi-label'><span>{html.escape(label)}</span>{badge_html}</div><div class='stitch-kpi-value'>{html.escape(str(value))}</div><div class='stitch-kpi-detail'>{html.escape(detail)}</div></div>"


def risk_tone(level):
    return "red" if str(level).startswith("Cr") else "yellow" if level == "Moderado" else "green"


def risk_color(level):
    return {"red": "#ffb4ab", "yellow": "#ffd166", "green": "#54e18c"}[risk_tone(level)]


def risk_pill(level):
    return {"red": "pill-red", "yellow": "pill-yellow", "green": "pill-green"}[risk_tone(level)]


def recommended_review_window(df_risk):
    """Return an operational review window from the current fleet risk levels."""
    levels = set(df_risk["risk_level"].astype(str))
    if "Crítico" in levels:
        return "PRIORITARIA", "Hay activos críticos; revisar cuanto antes", "red"
    if "Moderado" in levels:
        return "PRÓXIMAS 24 H", "Inspección recomendada para activos moderados", "orange"
    return "PREVENTIVA", "Sin alertas prioritarias; mantener rutina habitual", "green"


def render_risk_legend():
    st.markdown(
        "<div class='risk-legend'><span><b style='background:#ffb4ab'></b>Crítico (60%+)</span><span><b style='background:#ffb690'></b>Moderado (30-59%)</span><span><b style='background:#54e18c'></b>Estable (&lt;30%)</span></div>",
        unsafe_allow_html=True,
    )


def telemetry_figure(df_selected):
    figure = go.Figure()
    series = [
        ("voltage", "Voltaje", "#ffb4ab", "solid", "y"),
        ("vibration", "Vibración (mm/s)", "#4d8eff", "solid", "y2"),
        ("pressure", "Presión (bar)", "#a4c9ff", "dash", "y3"),
    ]
    for column, label, color, dash, axis in series:
        figure.add_trace(go.Scatter(
            x=df_selected["timestamp"], y=df_selected[column], name=label,
            mode="lines", line={"color": color, "width": 2, "dash": dash}, yaxis=axis,
        ))
    latest_time = df_selected["timestamp"].max()
    start_time = df_selected["timestamp"].quantile(.82)
    figure.add_vrect(x0=start_time, x1=latest_time, fillcolor="#93000a", opacity=.13, line_width=0)
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060e20", height=360,
        margin={"l": 12, "r": 62, "t": 32, "b": 35}, hovermode="x unified",
        font={"family":"JetBrains Mono", "color":"#dae2fd", "size":10},
        legend={"orientation":"h", "y":1.1, "x":0},
        xaxis={"gridcolor":"rgba(140,144,159,.12)", "showgrid":True},
        yaxis={"title":{"text":"Voltaje", "font":{"color":"#ffb4ab"}}, "gridcolor":"rgba(140,144,159,.12)"},
        yaxis2={"title":{"text":"Vibración", "font":{"color":"#4d8eff"}}, "overlaying":"y", "side":"right", "showgrid":False},
        yaxis3={"overlaying":"y", "side":"right", "position":.96, "showgrid":False, "showticklabels":False},
    )
    return figure


def render_replayed_telemetry(chart_placeholder, chart_rows, period):
    """Replace the live plot with the simulator window in the same chart slot."""
    chart_placeholder.empty()
    with chart_placeholder.container():
        st.markdown(
            f"<div class='chart-label'><span>TELEMETRÍA REPRODUCIDA</span>"
            f"<span class='eyebrow'>VENTANA {html.escape(str(period))}</span></div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            telemetry_figure(chart_rows),
            width="stretch",
            config={"displayModeBar": False},
            key="telemetry_main_chart",
        )


def fft_figure(df_selected):
    signal = df_selected["vibration"].astype(float).to_numpy()
    if len(signal) < 4:
        return go.Figure()
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    frequency = np.fft.rfftfreq(len(signal), d=1 / 100)
    figure = go.Figure(go.Scatter(
        x=frequency, y=spectrum, mode="lines", line={"color":"#4d8eff", "width":2},
        fill="tozeroy", fillcolor="rgba(77,142,255,.12)",
        hovertemplate="Frecuencia: %{x:.0f} Hz<br>Amplitud: %{y:.2f}<extra></extra>",
    ))
    peak_index = int(np.argmax(spectrum[1:]) + 1)
    figure.add_vline(x=float(frequency[peak_index]), line_dash="dash", line_color="#ffb4ab", annotation_text=f"Pico {frequency[peak_index]:.0f} Hz", annotation_font_color="#ffb4ab")
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#060e20", height=270,
        margin={"l":10,"r":15,"t":15,"b":35}, showlegend=False,
        font={"family":"JetBrains Mono", "color":"#dae2fd", "size":10},
        xaxis={"title":"Frecuencia (Hz)", "gridcolor":"rgba(140,144,159,.12)", "range":[0, min(500, float(frequency.max()))]},
        yaxis={"title":"Amplitud", "gridcolor":"rgba(140,144,159,.12)"},
    )
    return figure


def anomaly_heatmap_html(df_risk):
    cards = []
    for _, row in df_risk.sort_values("risk_score", ascending=False).head(6).iterrows():
        level = str(row["risk_level"])
        tone = risk_tone(level)
        action = "ACCIÓN REQUERIDA" if tone == "red" else "REVISIÓN" if tone == "yellow" else "NORMAL"
        cards.append(
            f"<div class='heat-cell'><strong class='{tone}'>{html.escape(str(row['machine_id']))}</strong><span class='{tone}'>RIESGO {row['risk_score']:.0f}%</span><span class='{tone}'>{action}</span><span class='{tone}'>ESTADO {html.escape(level.upper())}</span></div>"
        )
    return "<div class='anomaly-grid'>" + "".join(cards) + "</div>"


def render_sidebar_ai_card(df_risk, selected_machine, machine_row):
    """Tarjeta de análisis IA para el activo seleccionado en el sidebar.

    Solo presentation: reutiliza el ranking ya calculado por
    `compute_risk_from_model` sin recalcular predicciones.
    """
    selected = df_risk[df_risk["machine_id"] == selected_machine].iloc[0]
    color = risk_color(selected["risk_level"])
    tone = risk_tone(selected["risk_level"])
    priority = str(selected["priority"])
    fleet_top = get_priority_machine(df_risk)
    fleet_critical = int(df_risk["risk_level"].astype(str).str.startswith("Cr").sum())
    fleet_rank = int((df_risk["priority_score"] > selected["priority_score"]).sum()) + 1
    days_since = int(machine_row["days_since_maintenance"])

    verdict = {
        "Intervenir": "Intervención inmediata",
        "Inspeccionar": "Inspección programada",
        "Monitorear": "Monitoreo reforzado",
        "Ninguna": "Sin acción requerida",
    }.get(priority, "Revisar según procedimiento")

    is_fleet_top = str(fleet_top["machine_id"]) == str(selected_machine)
    if is_fleet_top and selected["risk_level"] in {"Crítico", "Moderado"}:
        lead = "El activo encabeza la cola priorizada de la flota."
    elif fleet_critical == 0:
        lead = f"Sin alertas activas en la flota. Posición {fleet_rank} de {len(df_risk)} por score de prioridad."
    else:
        lead = f"Posición {fleet_rank} de {len(df_risk)} en la cola de intervención."

    copy = (
        f"{lead} Riesgo {selected['risk_score']:.1f}% · {fleet_critical} activo(s) "
        f"críticos en planta · {days_since} días desde el último mantenimiento."
    )

    # Añadir badge de estado en el título
    status_badge = {
        "red": "<span class='pill-red'>⚠️ CRÍTICO</span>",
        "yellow": "<span class='pill-yellow'>🟠 MODERADO</span>",
        "green": "<span class='pill-green'>✅ ESTABLE</span>"
    }.get(tone, "")

    # Indicador de criticidad con badge y mejor formato de filas de análisis
    analysis_rows = "".join(
        f"<div class='meta-row'><span>{label}</span><strong>{html.escape(str(value))}</strong></div>"
        for label, value in [
            ("Nivel", selected["risk_level"]),
            ("Puntuación", f"{selected['risk_score']:.1f}%"),
            ("Criticidad", {"Alta": "🔴 ALTA", "Media": "🟠 MEDIA", "Baja": "🟢 BAJA"}.get(selected["criticality"], selected["criticality"])),
            ("Acción", priority),
            ("Horas op.", f"{machine_row['operating_hours']} h"),
        ]
    )

    st.markdown(
        f"<div class='ai-analysis-card' style='--tone:{color}'><div class='ai-analysis-head'><span class='eyebrow'>ANÁLISIS IA · ACTIVO</span>{status_badge}</div><div class='ai-analysis-verdict'>{html.escape(verdict)}</div><p class='ai-analysis-copy'>{html.escape(copy)}</p>{analysis_rows}</div>",
        unsafe_allow_html=True,
    )


def render_sidebar(df_machines, df_risk, data_source, model_source, feature_cols):
    with st.sidebar:
        st.markdown("<div class='brand'>&#128295; Mantenimiento</div><div class='eyebrow'>S08-26-EQUIPO-24</div><div style='color:var(--blue);font:.7rem JetBrains Mono;margin-top:.4rem'><span class='status-dot'></span>DEMO PREDICTIVA</div>", unsafe_allow_html=True)
        is_remote_data = str(data_source).lower().startswith("github")
        source_name = "GitHub" if is_remote_data else "Respaldo local"
        source_detail = "live_demo.parquet · origen remoto" if is_remote_data else "live_demo.parquet · archivo local"
        st.markdown(
            f"<div class='source-card'><div class='eyebrow'>ORIGEN DE DATOS</div><div class='source-value'>Datos de prueba cargados desde: {html.escape(source_name)}</div><div class='source-note'>{html.escape(source_detail)}<br>Dataset de demostración; no transmite sensores en vivo.</div></div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("<div class='eyebrow sidebar-heading'>NAVEGACIÓN</div>", unsafe_allow_html=True)
        for section_id, label in SECTION_OPTIONS:
            sidebar_type = "primary" if st.session_state.active_section == section_id else "secondary"
            if st.button(label, key=f"sidebar_{section_id}", width="stretch", type=sidebar_type):
                st.session_state.active_section = section_id
                rerun_app()
        with st.container(key="sidebar_filters"):
            st.markdown("<div class='eyebrow sidebar-heading'>FILTROS DE FLOTA</div>", unsafe_allow_html=True)
            machine_ids = df_machines["machine_id"].tolist()
            pending_machine = st.session_state.pop("pending_selected_machine", None)
            matching_machine = next(
                (machine_id for machine_id in machine_ids if str(machine_id) == str(pending_machine)),
                None,
            )
            if matching_machine is not None:
                st.session_state.selected_machine_filter = matching_machine
            default_machine = get_priority_machine(df_risk)["machine_id"]
            if st.session_state.get("selected_machine_filter") not in machine_ids:
                st.session_state.selected_machine_filter = default_machine
            selected_machine = st.selectbox("ID de máquina", machine_ids, key="selected_machine_filter")
            status_options = ["Crítico", "Moderado", "Estable"]
            selected_status = st.pills(
                "Filtro Estado",
                status_options,
                selection_mode="multi",
                default=status_options,
                format_func=lambda level: {"Crítico": "🔴 CRÍTICO", "Moderado": "🟠 MODERADO", "Estable": "🟢 ESTABLE"}[level],
                key="risk_filter_pills",
                width="stretch",
            )
        criticality_for_status = {"Crítico": "Alta", "Moderado": "Media", "Estable": "Baja"}
        selected_criticality = [criticality_for_status[level] for level in selected_status]
        machine_row = df_machines[df_machines["machine_id"] == selected_machine].iloc[0]
        metadata = [("ID", machine_row["machine_id"]), ("Tipo", machine_row["type"]), ("Ubicacion", machine_row["location"]), ("Operacion", f"{machine_row['operating_hours']} h"), ("Ultimo mant.", machine_row["last_maintenance"])]
        rows = "".join(f"<div class='meta-row'><span>{label}</span><strong>{html.escape(str(value))}</strong></div>" for label, value in metadata)
        st.markdown(f"<div class='sidebar-info-card'><div class='eyebrow'>ACTIVO SELECCIONADO</div>{rows}</div>", unsafe_allow_html=True)
        render_sidebar_ai_card(df_risk, selected_machine, machine_row)
        model_origin = "GitHub" if str(model_source).lower().startswith("github") else "respaldo local"
        st.markdown(f"<div class='sidebar-info-card'><div class='eyebrow'>MODELO PREDICTIVO</div><div class='source-value'>{html.escape(model_origin)}</div><div class='source-note'>{len(feature_cols)} variables · cálculo de riesgo por activo</div></div>", unsafe_allow_html=True)
        if st.button("Actualizar demo", icon=":material/refresh:", width="stretch"):
            reset_simulation_state()
            st.cache_data.clear()
            st.cache_resource.clear()
            rerun_app()
    return selected_machine, selected_status, selected_criticality


try:
    with st.spinner("Cargando datos y modelo..."):
        live_df, data_source = load_live_demo_data()
        dashboard_live_df = simulation_snapshot(live_df)
        df_machines, df_risk, df_telemetry, df_errors = compute_risk_from_model(dashboard_live_df)
        model, feature_cols, meta, model_source = get_model()
except Exception as error:
    st.error(f"Error al cargar datos o modelo: {error}")
    st.stop()

selected_machine, selected_status, selected_criticality = render_sidebar(df_machines, df_risk, data_source, model_source, feature_cols)
st.markdown("<div class='topbar'><div><span class='brand'>&#128295; Mantenimiento predictivo</span><div class='eyebrow'>S08-26-EQUIPO-24 · DESCUBRIMIENTO / MVP · <span style='color:var(--green)'><span class='status-dot'></span>DEMO ACTIVA</span></div></div><div class='mono' style='color:var(--muted);font-size:.7rem'>Streamlit Core 1.63</div></div>", unsafe_allow_html=True)
st.markdown("<div class='banner'><div><div class='banner-title'>Monitor Diagnóstico Industrial <span class='pill'>PLANTA-SUR // LÍNEA-A</span></div><div class='banner-copy'>Análisis predictivo multivariante sobre el conjunto de datos de demostración.</div></div><div class='pill'><span class='status-dot'></span>FUENTE: LIVE_DEMO</div></div>", unsafe_allow_html=True)

critical_count = int(df_risk["risk_level"].astype(str).str.startswith("Cr").sum())
avg_risk = float(df_risk["risk_score"].mean())
selected_risk = df_risk[df_risk["machine_id"] == selected_machine].iloc[0]
review_window, review_detail, review_tone = recommended_review_window(df_risk)
fleet_top = get_priority_machine(df_risk)
fleet_tone = "red" if critical_count else "yellow" if (df_risk["risk_level"] == "Moderado").any() else "green"
fleet_state = {"red": "ACCIÓN REQUERIDA", "yellow": "REVISIÓN RECOMENDADA", "green": "OPERACIÓN NORMAL"}[fleet_tone]
fleet_color = {"red": "#ffb4ab", "yellow": "#ffd166", "green": "#54e18c"}[fleet_tone]
selected_priority = str(selected_risk["priority"])
selected_color = risk_color(selected_risk["risk_level"])
selected_action = {
    "Intervenir": "Programar intervención prioritaria y revisar el equipo antes de continuar la operación.",
    "Inspeccionar": "Programar una inspección técnica y validar los componentes asociados al riesgo.",
    "Monitorear": "Mantener monitoreo reforzado y revisar los errores recientes.",
    "Ninguna": "Continuar con el monitoreo preventivo habitual.",
}.get(selected_priority, "Revisar el activo según el procedimiento de mantenimiento.")
if fleet_top["risk_level"] == "Crítico":
    fleet_action = "Intervención prioritaria: detener y revisar el activo antes de continuar la operación."
elif fleet_top["risk_level"] == "Moderado":
    fleet_action = "Inspección prioritaria: programar revisión técnica y validar sus componentes."
else:
    fleet_action = "Mantener el monitoreo preventivo; no hay activos críticos o moderados."
recommended_actions = [(str(selected_machine), selected_action)]
if str(selected_machine) != str(fleet_top["machine_id"]) and fleet_top["risk_level"] in {"Crítico", "Moderado"}:
    recommended_actions.append((str(fleet_top["machine_id"]), fleet_action))
action_items_html = "".join(
    f"<li><strong>{html.escape(machine_id)}:</strong> {html.escape(action)}</li>"
    for machine_id, action in recommended_actions
)
with st.container(border=True, key="fleet_ai_panel"):
    priority_copy = f"Atención prioritaria: {fleet_top['machine_id']} · {fleet_top['risk_level']} · {fleet_top['risk_score']:.1f}%" if fleet_top["risk_level"] in {"Crítico", "Moderado"} else "Sin activos críticos o moderados en la flota."
    st.markdown(
        f"<div class='ai-panel-heading'><div><strong>Diagnóstico y prioridad de mantenimiento</strong></div><span class='pill' style='color:{fleet_color}'>{fleet_state}</span></div><p class='ai-panel-summary'>{critical_count} activo(s) críticos · {html.escape(priority_copy)}</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='ai-diagnostic'><div class='eyebrow'>DIAGNÓSTICO ACTUALIZADO</div><div class='ai-diagnostic-grid'><section class='ai-result-card' style='--card-accent:{selected_color}'><div class='eyebrow'>ACTIVO SELECCIONADO</div><strong class='ai-result-title'>{html.escape(str(selected_machine))}</strong><p>{html.escape(str(selected_risk['risk_level']))} · riesgo <strong>{selected_risk['risk_score']:.1f}%</strong></p><p>Prioridad: <strong>{html.escape(selected_priority)}</strong></p></section><section class='ai-result-card' style='--card-accent:{fleet_color}'><div class='eyebrow'>PRIORIDAD DE FLOTA</div><strong class='ai-result-title'>{html.escape(str(fleet_top['machine_id']))} · {html.escape(str(fleet_top['risk_level']))}</strong><p>{critical_count} activo(s) críticos</p><p>Riesgo: <strong>{fleet_top['risk_score']:.1f}%</strong> · {html.escape(str(fleet_top['priority']))}</p></section><section class='ai-result-card' style='--card-accent:var(--blue)'><div class='eyebrow'>ACCIONES RECOMENDADAS</div><ul>{action_items_html}</ul></section></div></div>",
        unsafe_allow_html=True,
    )

nav_columns = st.columns(4)
for nav_column, (section_id, label) in zip(nav_columns, SECTION_OPTIONS):
    with nav_column:
        button_type = "primary" if st.session_state.active_section == section_id else "secondary"
        if st.button(label, key=f"top_{section_id}", width="stretch", type=button_type):
            st.session_state.active_section = section_id
            rerun_app()

st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)


if st.session_state.active_section == "overview":
    overview_risk = df_risk.sort_values("risk_score", ascending=False).head(6)
    kpi_columns = st.columns(4)
    kpi_cards = [
        ("MAQUINAS MONITOREADAS", len(df_machines), f"{len(df_machines)} unidades IoT", "blue", "SCADA ACTIVE"),
        ("RIESGO CRITICO", critical_count, "Intervencion prioritaria" if critical_count else "Sin criticos activos", "red" if critical_count else "green", ""),
        ("VENTANA DE REVISIÓN", review_window, review_detail, review_tone, "HORIZONTE 24 H"),
        ("RIESGO PROMEDIO", f"{avg_risk:.1f}%", "Indice global", "red" if fleet_tone == "red" else "orange" if fleet_tone == "yellow" else "green", "FLEET AVG"),
    ]
    for column, card in zip(kpi_columns, kpi_cards):
        with column:
            st.markdown(render_stitch_kpi(*card), unsafe_allow_html=True)
    with st.container(border=True, key="risk_distribution_panel"):
        st.markdown(
            "<div class='section-head'><div><h2>Distribucion de riesgo operacional por activo</h2>"
            "<p>Probabilidad estimada de falla en las proximas 24 horas.</p></div>"
            "<span class='pill'>FLEET AVG</span></div>",
            unsafe_allow_html=True,
        )
        render_risk_legend()
        st.plotly_chart(risk_figure(overview_risk), width="stretch", config={"displayModeBar": False})
    with st.container(border=True, key="risk_matrix_panel"):
        st.markdown(
            "<div class='section-head'><div><h2>Matriz diagnostica de flota</h2>"
            "<p>Ranking calculado desde la ultima lectura disponible.</p></div></div>",
            unsafe_allow_html=True,
        )
        render_risk_table(overview_risk, selected_status, selected_criticality)

if st.session_state.active_section == "telemetry":
    with st.container(border=True, key="live_telemetry_panel"):
        st.markdown(
            "<div class='section-head'><div><h2>Telemetria en Vivo</h2>"
            "<p>Vista historica de las lecturas disponibles para el activo seleccionado.</p></div>"
            "<span class='pill'>HISTORICO LIVE_DEMO</span></div>",
            unsafe_allow_html=True,
        )
        df_live_selected = df_telemetry[
            df_telemetry["machine_id"] == selected_machine
        ].sort_values("timestamp")
        chart_placeholder = st.empty()
        if not df_live_selected.empty:
            simulation_time = st.session_state.get("demo_sim_datetime")
            if simulation_time is None:
                selected_period = st.session_state.get("demo_sim_period", "24H")
                live_chart_rows = df_live_selected.tail(PERIOD_HOURS.get(selected_period, 24))
                st.markdown(
                    "<div class='chart-label'><span>TELEMETRÍA EN VIVO</span>"
                    "<span class='eyebrow'>ÚLTIMAS LECTURAS HISTÓRICAS</span></div>",
                    unsafe_allow_html=True,
                )
                chart_placeholder.plotly_chart(
                    telemetry_figure(live_chart_rows),
                    width="stretch",
                    config={"displayModeBar": False},
                    key="telemetry_main_chart",
                )
            render_machine_detail(df_errors, selected_machine)

            latest = df_live_selected.iloc[-1]
            previous = df_live_selected.iloc[-2] if len(df_live_selected) > 1 else latest
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Voltaje", f"{latest['voltage']:.2f} V", f"{latest['voltage'] - previous['voltage']:+.2f} V")
            with metric_cols[1]:
                st.metric("Vibracion", f"{latest['vibration']:.2f} mm/s", f"{latest['vibration'] - previous['vibration']:+.2f} mm/s", delta_color="inverse")
            with metric_cols[2]:
                st.metric("Presion", f"{latest['pressure']:.2f} bar", f"{latest['pressure'] - previous['pressure']:+.2f} bar", delta_color="inverse")
        else:
            st.info(f"No hay lecturas historicas para {selected_machine}.")

    with st.container(border=True, key="demo_telemetry_section"):
        st.markdown(
            "<div class='section-head'><div><h2>Telemetria de Demostracion</h2>"
            "<p>Reproduccion horaria del conjunto procesado y prediccion del riesgo con el modelo.</p></div>"
            "<span class='pill'>SIMULADOR</span></div>",
            unsafe_allow_html=True,
        )
        render_demo_simulator(
            live_df,
            selected_machine,
            model,
            feature_cols,
            meta.get("decision_threshold", 0.5),
            lambda chart_rows: render_replayed_telemetry(
                chart_placeholder, chart_rows, st.session_state.get("demo_sim_period", "24H")
            ),
        )

if st.session_state.active_section == "anomalies":
    anomaly_telemetry = df_telemetry[df_telemetry["machine_id"] == selected_machine].sort_values("timestamp")
    st.markdown(f"<div class='telemetry-head'><div><span class='machine-tag' style='color:{risk_color(selected_risk['risk_level'])}'>&#128269;</span><div><h2>Módulo de diagnóstico espectral y detección de anomalías</h2><p>Modelado FFT y clasificación de anomalías sobre el activo seleccionado.</p></div></div><span class='pill {risk_pill(selected_risk['risk_level'])}'>{html.escape(str(selected_risk['risk_level']).upper())}</span></div>", unsafe_allow_html=True)
    anomaly_col, heat_col = st.columns([1, 2])
    with anomaly_col:
        confidence = float(selected_risk["risk_score"])
        st.markdown(f"<div class='sidebar-card'><div class='eyebrow'>RIESGO ESTIMADO</div><div class='mono' style='font-size:1.7rem;font-weight:700;color:{risk_color(selected_risk['risk_level'])};margin:.55rem 0'>{confidence:.1f}%</div><div style='color:var(--muted);font-size:.74rem'>{selected_risk['risk_level']} · {selected_risk['priority']}</div></div>", unsafe_allow_html=True)
        st.progress(confidence / 100, text="Probabilidad estimada")
        st.caption(f"Algoritmo: {meta.get('model_type', 'modelo predictivo')} · threshold {meta.get('decision_threshold', 0.5):.3f}")
    with heat_col:
        st.markdown("<div class='section-head'><div><h3>Mapa de estado por activo</h3><p>Vista preparada para conectar subsistemas y sensores reales.</p></div></div>", unsafe_allow_html=True)
        st.markdown(anomaly_heatmap_html(df_risk), unsafe_allow_html=True)
    fft_col, events_col = st.columns([7, 5])
    with fft_col:
        st.markdown("<div class='section-head'><div><h3>Espectro de frecuencia vibracional (FFT)</h3><p>Acelerómetro triaxial · Eje de vibración de la máquina seleccionada.</p></div><span class='pill'>BPFO ANALYSIS</span></div>", unsafe_allow_html=True)
        if not anomaly_telemetry.empty:
            st.plotly_chart(fft_figure(anomaly_telemetry), width="stretch", config={"displayModeBar": False})
    with events_col:
        st.markdown("<div class='section-head'><div><h3>Registro crítico de eventos</h3><p>Últimas señales y diagnósticos del activo.</p></div></div>", unsafe_allow_html=True)
        selected_errors = df_errors[df_errors["machine_id"] == selected_machine].sort_values("timestamp", ascending=False).head(3)
        if selected_errors.empty:
            st.success(f"Sin eventos registrados para {selected_machine}.")
        else:
            for _, event in selected_errors.iterrows():
                st.markdown(f"<div class='sidebar-card'><div class='eyebrow'>{html.escape(str(event['timestamp']))} · {html.escape(str(event['error_code']))}</div><div style='color:var(--text);font-size:.78rem;margin-top:.35rem'>{html.escape(str(event['description']))}</div></div>", unsafe_allow_html=True)

if st.session_state.active_section == "maintenance":
    ordered = df_risk.merge(df_machines[["machine_id", "type", "location", "days_since_maintenance"]], on="machine_id").sort_values("priority_score", ascending=False)
    if not ordered.empty:
        top = ordered.iloc[0]
        top_tone = risk_tone(top["risk_level"])
        top_color = risk_color(top["risk_level"])
        top_action = "ACCIÓN REQUERIDA" if top_tone == "red" else "REVISIÓN RECOMENDADA" if top_tone == "yellow" else "OPERACIÓN NORMAL"
        st.markdown(f"<div class='section-head'><div><h2>Plan de mantenimiento</h2><p>Cola priorizada según el estado real de cada activo.</p></div><span class='pill {risk_pill(top['risk_level'])}'>{top_action}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='maintenance-hero' style='--state-color:{top_color}'><div><span class='pill {risk_pill(top['risk_level'])}'>{top_action}</span><span class='eyebrow' style='margin-left:.55rem'>ESTADO DEL ACTIVO</span><h2>{html.escape(str(top['machine_id']))} · {html.escape(str(top['risk_level']))}</h2><p>Riesgo estimado: <strong>{top['risk_score']:.0f}%</strong>. Acción sugerida: <strong>{html.escape(str(top['priority']))}</strong>. {int(top['days_since_maintenance'])} días desde el último mantenimiento.</p></div><span class='pill'>LIVE_DEMO</span></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-head'><div><h2>Plan de mantenimiento</h2><p>Sin activos disponibles para priorizar.</p></div><span class='pill pill-green'>OPERACIÓN NORMAL</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='resource-grid'><div class='resource-card'><div class='resource-icon'>&#128101;</div><div><span>TECNICOS DISPONIBLES</span><strong>3 equipos de guardia</strong><span style='color:var(--green)'>Turno operativo</span></div></div><div class='resource-card'><div class='resource-icon'>&#128230;</div><div><span>REPUESTOS CRITICOS</span><strong>Inventario por conectar</strong><span>Fuente preparada para integración</span></div></div><div class='resource-card'><div class='resource-icon'>&#9201;</div><div><span>MTBF PROYECTADO</span><strong>Modelo en ejecución</strong><span>Calculado al conectar historial</span></div></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-head'><div><h2>Cola de intervención priorizada</h2><p>Ordenada por riesgo, criticidad e impacto operacional.</p></div><span class='eyebrow'>ALGORITMO RUL</span></div>", unsafe_allow_html=True)
    for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
        tone = risk_color(row["risk_level"])
        st.markdown(f"<div class='priority' style='border-left-color:{tone}'><div class='priority-title'><span class='rank-badge' style='background:{tone}'>{rank}</span>{html.escape(str(row['machine_id']))} — {html.escape(str(row['type']))} <span class='pill {risk_pill(row['risk_level'])}'>{row['risk_score']:.0f}% · {html.escape(str(row['risk_level']).upper())}</span></div><div class='priority-copy'>Acción: {html.escape(str(row['priority']))} · Ubicación: {html.escape(str(row['location']))}</div><div class='priority-meta'><span>Criticidad: {html.escape(str(row['criticality']))}</span><span>{int(row['days_since_maintenance'])} días sin mantenimiento</span><span>Prioridad: {row['priority_score']:.0f}</span></div></div>", unsafe_allow_html=True)
