import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from components.risk_table import render_risk_table
from components.sensor_chart import render_sensor_chart
from components.machine_detail import render_machine_detail
from components.priority_list import render_priority_list
from utils.data_loader import load_live_demo_data, compute_risk_from_model
from utils.model_loader import get_model

st.set_page_config(
    page_title='PredictiveMaintenance',
    page_icon='🔧',
    layout='wide',
    initial_sidebar_state='expanded'
)


def safe_rerun():
    """Intenta reiniciar el script de Streamlit de forma compatible.

    Algunas versiones de Streamlit no exponen `st.experimental_rerun`.
    Esta función intenta usarlo y, si no existe, intenta lanzar la
    excepción interna de rerun; si todo falla, muestra una advertencia
    para que el usuario recargue manualmente la página.
    """
    if hasattr(st, "experimental_rerun"):
        try:
            st.experimental_rerun()
            return
        except Exception:
            pass

    # Intentar lanzar la excepción interna de rerun (varía según versión)
    try:
        from streamlit.runtime.scriptrunner.script_runner import RerunException
        raise RerunException()
    except Exception:
        try:
            # Fallback a ubicaciones antiguas
            from streamlit.scriptrunner import RerunException
            raise RerunException()
        except Exception:
            st.warning('No es posible reiniciar programáticamente en esta versión de Streamlit. Por favor, refresca la página manualmente.')
            return

# ── Estilos CSS personalizados (tema oscuro premium) ──
st.markdown("""
<style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Fuente global */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header principal */
    h1 {
        background: linear-gradient(135deg, #3b82f6, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }

    /* KPI cards — borde según tipo */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        padding: 16px 20px;
        backdrop-filter: blur(10px);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        border-color: rgba(59, 130, 246, 0.7);
        transform: translateY(-2px);
    }

    /* Valor del metric grande */
    div[data-testid="metric-container"] > div:first-child {
        font-size: 2rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(59, 130, 246, 0.2);
    }

    /* Tabs activos */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom-color: #3b82f6 !important;
        font-weight: 600 !important;
    }

    /* Contenedores de sección */
    div[data-testid="stVerticalBlock"] > div[data-testid="element-container"] {
        border-radius: 8px;
    }

    /* Botones y elementos interactivos */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stMultiSelect"] label {
        font-weight: 500;
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Footer personalizado oculto */
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Selector de origen y control de recarga (sidebar inicial)
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 4px 0;'>
        <div style='font-size: 2rem;'>🔧</div>
        <div style='font-size: 1.1rem; font-weight: 700; color: #60a5fa; letter-spacing: -0.3px;'>PredictiveMaintenance</div>
        <div style='font-size: 0.7rem; color: #64748b; margin-top: 2px;'>S08-26-EQUIPO-24</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    source_choice = st.radio('Origen de datos', options=['GitHub', 'Local'], index=0)
    prefer_local = source_choice == 'Local'

    st.caption('Elija `Local` para forzar uso de archivos en disco (desarrollo).')

    if st.button('🔄 Recargar modelo y datos', key='reload_choice'):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        safe_rerun()

# Cargar datos y modelo (usando la preferencia seleccionada)
try:
    with st.spinner('Cargando datos y modelo...'):
        live_df, data_source = load_live_demo_data(prefer_local=prefer_local)
        df_machines, df_risk, df_telemetry, df_errors = compute_risk_from_model(live_df, prefer_local_model=prefer_local)
        model, feature_cols, meta, model_source = get_model(prefer_local=prefer_local)
except Exception as e:
    st.error(f"Error al cargar datos o modelo: {e}")
    st.stop()

# ═══════════════════════════════════════════════
# BARRA LATERAL
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"**Fuente de datos:** {data_source}")
    st.divider()

    # Info del modelo
    pr_auc = meta.get("pr_auc")
    pr_auc_text = f"{pr_auc:.4f}" if pr_auc is not None else "N/D"
    threshold_text = f"{meta['decision_threshold']:.3f}"
    st.markdown(f"""
    <div style='padding: 12px; border-radius: 8px; background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); margin-bottom: 16px;'>
        <div style='font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em;'>Modelo ML</div>
        <div style='font-weight: 600; color: #60a5fa; margin-top: 4px;'>Fuente: {model_source}</div>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 4px;'>PR-AUC: {pr_auc_text}</div>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>Threshold: {threshold_text}</div>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>Features: {len(feature_cols)}</div>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>Entrenamiento: {meta.get("train_start", "?")} → {meta.get("train_end", "?")}</div>
        <div style='color: #94a3b8; font-size: 0.85rem; margin-top: 2px;'>Test: {meta.get("test_start", "?")} → {meta.get("test_end", "?")}</div>
    </div>
    """, unsafe_allow_html=True)

    # Selector de máquina
    machine_ids = df_machines['machine_id'].tolist()
    selected_machine = st.selectbox(
        'Máquina',
        options=machine_ids,
        index=0
    )

    # Filtro de estado
    status_options = ['Crítico', 'Moderado', 'Estable']
    selected_status = st.multiselect(
        'Filtro de estado',
        options=status_options,
        default=['Crítico', 'Moderado', 'Estable']
    )

    # Filtro de criticidad
    criticality_options = ['Alta', 'Media', 'Baja']
    selected_criticality = st.multiselect(
        'Filtro de criticidad',
        options=criticality_options,
        default=criticality_options
    )

    st.divider()

    # Metadatos rápidos de la máquina seleccionada
    machine_row = df_machines[df_machines['machine_id'] == selected_machine].iloc[0]
    st.subheader('Metadatos')
    st.write(f'**ID:** {machine_row["machine_id"]}')
    st.write(f'**Tipo:** {machine_row["type"]}')
    st.write(f'**Ubicación:** {machine_row["location"]}')
    st.write(f'**Horas operación:** {machine_row["operating_hours"]} h')
    st.write(f'**Último mantenimiento:** {machine_row["last_maintenance"]}')
    st.write(f'**Días sin mantenimiento:** {machine_row["days_since_maintenance"]}')

    # Botón para forzar recarga de datos y modelo
    if st.button('🔄 Recargar modelo y datos', key='reload_sidebar'):
        try:
            st.cache_data.clear()
        except Exception:
            pass
        try:
            st.cache_resource.clear()
        except Exception:
            pass
        safe_rerun()

# ═══════════════════════════════════════════════
# CONTENEDOR PRINCIPAL
# ═══════════════════════════════════════════════
st.markdown("""
<h1 style='margin-bottom: 0;'>🔧 PredictiveMaintenance</h1>
""", unsafe_allow_html=True)
st.markdown("""
<p style='color: #64748b; font-size: 0.9rem; margin-top: 0; margin-bottom: 4px;'>
    <strong style='color: #3b82f6;'>S08-26-EQUIPO-24</strong> &nbsp;|&nbsp; Fase: Discovery / MVP Dashboard &nbsp;|&nbsp;
    <span style='color: #22c55e;'>● Sistema activo</span>
</p>
""", unsafe_allow_html=True)

st.divider()

# Fila de KPIs principales
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label='🏭 Máquinas Monitoreadas',
        value=len(df_machines),
        delta=f'{len(df_machines)} activas',
        delta_color='normal'
    )

with kpi2:
    critical_count = len(df_risk[df_risk['risk_level'] == 'Crítico'])
    st.metric(
        label='⚠️ Riesgo Crítico',
        value=critical_count,
        delta='requiere atención inmediata',
        delta_color='inverse'
    )

with kpi3:
    # Convert next_maintenance timestamp to string for display
    next_maint_date = df_machines['next_maintenance'].min().strftime('%Y-%m-%d') if not df_machines['next_maintenance'].isna().all() else 'N/A'
    st.metric(
        label='📅 Próximo Mantenimiento',
        value=next_maint_date,
        delta='más próximo programado'
    )

with kpi4:
    avg_risk = df_risk['risk_score'].mean()
    risk_delta = 'moderado' if avg_risk < 70 else 'alto'
    st.metric(
        label='📊 Riesgo Promedio',
        value=f'{avg_risk:.0f}%',
        delta=risk_delta,
        delta_color='inverse' if avg_risk >= 70 else 'normal'
    )

st.divider()

# Pestañas principales — Las 3 preguntas del producto
tab1, tab2, tab3 = st.tabs([
    '🎯 1. Identificar (Riesgo)',
    '📡 2. Comprender (Señales)',
    '🚀 3. Priorizar (Acción)'
])

# ── Pestaña 1: Identificar ──
with tab1:
    st.subheader('📊 Ranking de Riesgo por Máquina')

    # Alerta de máquinas críticas
    critical_machines = df_risk[df_risk['risk_level'] == 'Crítico']['machine_id'].tolist()
    if critical_machines:
        st.error(
            f'⚠️ **Atención inmediata:** {len(critical_machines)} máquina(s) en nivel CRÍTICO → '
            f'{", ".join(critical_machines)}'
        )

    # Tabla interactiva de riesgo con filtros
    render_risk_table(df_risk, selected_status, selected_criticality)

    st.divider()
    st.subheader('📈 Distribución de Riesgo')

    # Gráfico Plotly horizontal — ordenado por riesgo desc, colores semánticos
    df_chart = df_risk.sort_values('risk_score', ascending=True).copy()
    color_map = {
        'Crítico': '#ff4b4b',
        'Moderado': '#f97316',
        'Estable': '#22c55e'
    }
    bar_colors = df_chart['risk_level'].map(color_map).tolist()

    fig = go.Figure(go.Bar(
        x=df_chart['risk_score'],
        y=df_chart['machine_id'],
        orientation='h',
        marker_color=bar_colors,
        text=[f'{v}%' for v in df_chart['risk_score']],
        textposition='outside',
        customdata=df_chart[['risk_level', 'criticality', 'priority']].values,
        hovertemplate=(
            '<b>%{y}</b><br>'
            'Riesgo: %{x}%<br>'
            'Nivel: %{customdata[0]}<br>'
            'Criticidad: %{customdata[1]}<br>'
            'Acción: %{customdata[2]}<extra></extra>'
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(30,41,59,0.5)',
        font=dict(family='Inter, sans-serif', color='#f1f5f9'),
        xaxis=dict(
            title='Índice de Riesgo (%)',
            range=[0, 110],
            gridcolor='rgba(100,116,139,0.2)',
            color='#94a3b8'
        ),
        yaxis=dict(color='#f1f5f9'),
        margin=dict(l=20, r=40, t=20, b=20),
        height=280,
        showlegend=False,
    )
    # La categoría Crítico empieza en 60% según get_level_and_criticality.
    fig.add_vline(x=60, line_dash='dash', line_color='#ff4b4b',
                  annotation_text='Umbral crítico (60%)',
                  annotation_font_color='#ff4b4b',
                  annotation_position='top right')
    st.plotly_chart(fig, use_container_width=True)

# ── Pestaña 2: Comprender ──
with tab2:
    st.subheader(f'📡 Señales en Tiempo Real — {selected_machine}')

    # Layout de dos columnas
    col_sensors, col_detail = st.columns([2, 1])

    with col_sensors:
        st.markdown('**🌡️ Telemetría de sensores**')
        render_sensor_chart(df_telemetry, selected_machine)

    with col_detail:
        st.markdown('**🔴 Histórico de errores**')
        render_machine_detail(df_errors, selected_machine)

    st.divider()
    st.subheader('⚡ Estado de señales clave')

    # Métricas dinámicas basadas en la máquina seleccionada
    df_sel = df_telemetry[df_telemetry['machine_id'] == selected_machine]
    if not df_sel.empty:
        df_sel_sorted = df_sel.sort_values('timestamp')
        first = df_sel_sorted.iloc[0]
        latest = df_sel_sorted.iloc[-1]

        temp_delta  = latest['temperature'] - first['temperature']
        vib_delta   = latest['vibration']   - first['vibration']
        pres_delta  = latest['pressure']    - first['pressure']

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric(
                label='🌡️ Temperatura actual',
                value=f"{latest['temperature']:.1f} °C",
                delta=f"{temp_delta:+.1f} °C vs inicio",
                delta_color='inverse'
            )
        with col_s2:
            st.metric(
                label='📳 Vibración actual',
                value=f"{latest['vibration']:.2f} mm/s",
                delta=f"{vib_delta:+.2f} mm/s vs inicio",
                delta_color='inverse'
            )
        with col_s3:
            st.metric(
                label='💧 Presión actual',
                value=f"{latest['pressure']:.1f} bar",
                delta=f"{pres_delta:+.1f} bar vs inicio"
            )

# ── Pestaña 3: Priorizar ──
with tab3:
    st.subheader('🚀 Cola de Intervención — Ordenada por Prioridad')

    st.info(
        '📌 Las tareas se ordenan por **riesgo × criticidad × impacto**. '
        'Intervenir en orden descendente para maximizar disponibilidad de planta.'
    )

    # Lista mejorada con progress bars de riesgo
    df_prio = df_risk.merge(
        df_machines[['machine_id', 'type', 'location', 'days_since_maintenance']], on='machine_id'
    ).sort_values('priority_score', ascending=False)

    rank_icons = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣', '6️⃣']
    for idx, (_, row) in enumerate(df_prio.iterrows()):
        icon = rank_icons[idx] if idx < len(rank_icons) else f'{idx+1}.'
        level_color = {'Crítico': '#ff4b4b', 'Moderado': '#f97316', 'Estable': '#22c55e'}.get(row['risk_level'], '#64748b')
        action_emoji = {'Intervenir': '🔴', 'Inspeccionar': '🟠', 'Monitorear': '🟡', 'Revisar': '🟡', 'Ninguna': '🟢'}.get(row['priority'], '⚪')

        with st.container():
            c1, c2, c3, c4 = st.columns([0.3, 2.5, 2, 1.5])
            with c1:
                st.markdown(f"<div style='font-size:1.5rem; text-align:center; padding-top:8px;'>{icon}</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"**{row['machine_id']}** — {row['type']}")
                st.caption(f"📍 {row['location']} · {row['days_since_maintenance']}d sin mantenimiento")
                # `st.progress` espera un valor entre 0 y 1
                prog = float(row['risk_score']) / 100.0
                prog = max(0.0, min(1.0, prog))
                st.progress(prog)
            with c3:
                st.markdown(f"<span style='color:{level_color}; font-weight:600;'>● {row['risk_level']}</span> · Criticidad {row['criticality']}", unsafe_allow_html=True)
            with c4:
                st.markdown(f"{action_emoji} **{row['priority']}**")
            st.divider()

    # Card de recomendación principal
    top = df_prio.iloc[0]
    top_color = {'Crítico': '#ff4b4b', 'Moderado': '#f97316', 'Estable': '#22c55e'}.get(top['risk_level'], '#64748b')
    st.markdown(f"""
    <div style='
        background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(30,41,59,0.9));
        border: 1px solid #3b82f6;
        border-left: 4px solid {top_color};
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 8px;
    '>
        <div style='font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;'>🎯 RECOMENDACIÓN DE INTERVENCIÓN</div>
        <div style='font-size: 1.3rem; font-weight: 700; color: #f1f5f9; margin-bottom: 4px;'>{top['machine_id']} — {top['type']}</div>
        <div style='color: #94a3b8; margin-bottom: 12px;'>📍 {top['location']} · {top['days_since_maintenance']} días sin mantenimiento</div>
        <div style='display: flex; gap: 24px; flex-wrap: wrap;'>
            <span style='color: {top_color}; font-weight: 600;'>● Riesgo {top['risk_score']}% ({top['risk_level']})</span>
            <span style='color: #94a3b8;'>Criticidad: {top['criticality']}</span>
            <span style='color: #60a5fa; font-weight: 600;'>▶ Acción: {top['priority']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
