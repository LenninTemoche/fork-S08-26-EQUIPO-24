"""Reproduce el conjunto live ya transformado para la demo de telemetría."""

import pandas as pd
import streamlit as st

from components.machine_detail import navigate_to_diagnostic_matrix
from utils.model_loader import predict_probabilities


SIM_INDEX_KEY = "demo_sim_index"
SIM_TIME_KEY = "demo_sim_datetime"
SIM_RUNNING_KEY = "demo_sim_running"
SIM_ALERT_KEY = "demo_sim_previous_risk_levels"
PERIOD_KEY = "demo_sim_period"

PERIOD_HOURS = {
    "3H": 3,
    "6H": 6,
    "12H": 12,
    "24H": 24,
    "36H": 36,
    "72H": 72,
    "3D": 72,
    "7D": 168,
    "15D": 360,
    "30D": 720,
    "1M": 720,
    "3M": 2160,
}

FEATURE_LABELS = {
    "time_since_last_error_h": "Horas desde último error",
    "distinct_errors_last_24h": "Tipos de error · 24 h",
    "hours_since_maintenance": "Horas desde mantenimiento",
    "errors_last_24h": "Errores · 24 h",
    "has_error_recent": "Error reciente",
    "days_since_maintenance": "Días desde mantenimiento",
    "time_since_last_component_replacement_h": "Horas desde reemplazo",
    "volt_roll_mean_24h": "Voltaje medio · 24 h",
    "rotate_roll_mean_24h": "Rotación media · 24 h",
    "pressure_roll_mean_24h": "Presión media · 24 h",
    "vibration_roll_mean_24h": "Vibración media · 24 h",
}


def _risk_level(probability: float) -> str:
    if probability >= 0.60:
        return "Critico"
    if probability >= 0.30:
        return "Moderado"
    return "Estable"


def simulation_snapshot(live_df: pd.DataFrame) -> pd.DataFrame:
    """Return all prepared rows up to the current replay time, when set."""
    current_time = st.session_state.get(SIM_TIME_KEY)
    if current_time is None:
        return live_df
    return live_df[live_df["datetime"] <= pd.Timestamp(current_time)]


def _init_simulation_state(timeline_size: int) -> None:
    st.session_state.setdefault(SIM_INDEX_KEY, -1)
    st.session_state.setdefault(SIM_TIME_KEY, None)
    st.session_state.setdefault(SIM_RUNNING_KEY, False)
    if not isinstance(st.session_state.get(SIM_ALERT_KEY), dict):
        st.session_state[SIM_ALERT_KEY] = {}
    st.session_state.setdefault(PERIOD_KEY, "24H")
    if st.session_state[SIM_INDEX_KEY] >= timeline_size:
        st.session_state[SIM_INDEX_KEY] = -1
        st.session_state[SIM_TIME_KEY] = None
        st.session_state[SIM_RUNNING_KEY] = False


def _telemetry_window(live_df: pd.DataFrame, machine_id: str, current_time):
    machine_col = "machine_id" if "machine_id" in live_df.columns else "machineID"
    machine_rows = live_df[live_df[machine_col].astype(str) == str(machine_id)]
    machine_rows = machine_rows[machine_rows["datetime"] <= current_time]
    limit = PERIOD_HOURS[st.session_state[PERIOD_KEY]]
    return machine_rows.sort_values("datetime").tail(limit)


def render_demo_simulator(
    live_df: pd.DataFrame,
    machine_id: str,
    model,
    feature_cols: list[str],
    threshold: float,
    render_chart,
) -> None:
    """Render playback controls and a live, model-scored fleet timestamp."""
    timeline = pd.Index(live_df["datetime"].drop_duplicates().sort_values())
    _init_simulation_state(len(timeline))

    st.markdown(
        """
        <style>
        .st-key-demo_simulator_panel { padding:.65rem .8rem!important; border-color:rgba(126,171,255,.2)!important; background:linear-gradient(115deg,rgba(23,31,51,.92),rgba(17,26,45,.84))!important; }
        .simulator-title { display:flex; align-items:center; gap:.65rem; min-height:2.5rem; }
        .simulator-machine { display:grid; place-items:center; width:2.45rem; height:2.45rem; flex:none; border-radius:5px; background:rgba(255,180,171,.12); color:#ffb4ab; font:700 1rem 'JetBrains Mono',monospace; }
        .simulator-title strong { display:block; color:#dae2fd; font:600 .9rem 'Space Grotesk',sans-serif; }
        .simulator-title span { color:#9da6bd; font-size:.66rem; }
        .simulator-controls-label { margin:.1rem 0 .3rem; color:#9da6bd; font:500 .59rem 'JetBrains Mono',monospace; letter-spacing:.08em; }
        .st-key-demo_period_row_1,.st-key-demo_period_row_2 { gap:.2rem!important; }
        .st-key-demo_period_row_1 button,.st-key-demo_period_row_2 button { min-height:1.85rem; padding:.15rem .25rem; font-size:.66rem; }
        .st-key-demo_period_row_1 button[kind="primary"],.st-key-demo_period_row_2 button[kind="primary"] { background:rgba(77,142,255,.24); color:#dae2fd; border-color:#7eabff; }
        .simulator-reading { height:100%; min-height:5.4rem; padding:.65rem .75rem; border:1px solid rgba(126,171,255,.15); border-radius:6px; background:rgba(34,42,61,.56); box-sizing:border-box; }
        .simulator-reading-label { color:#9da6bd; font:500 .59rem 'JetBrains Mono',monospace; letter-spacing:.06em; text-transform:uppercase; }
        .simulator-reading-value { margin:.35rem 0 .2rem; color:#dae2fd; font:700 1.15rem 'JetBrains Mono',monospace; }
        .simulator-reading-note { color:#9da6bd; font-size:.65rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="demo_simulator_panel"):
        control_cols = st.columns([1.35, 2.7, 1.5, 1.0], vertical_alignment="center")
        with control_cols[0]:
            st.markdown(
                f"<div class='simulator-title'><span class='simulator-machine'>{machine_id}</span><div><strong>Telemetría de demostración</strong><span>Reproducción histórica del activo</span></div></div>",
                unsafe_allow_html=True,
            )
        with control_cols[1]:
            st.markdown("<div class='simulator-controls-label'>VENTANA DE TELEMETRÍA</div>", unsafe_allow_html=True)
            period_rows = [list(PERIOD_HOURS)[:6], list(PERIOD_HOURS)[6:]]
            for row_index, periods in enumerate(period_rows, start=1):
                period_cols = st.columns(6, gap="small")
                for column, period in zip(period_cols, periods):
                    with column:
                        selected = st.session_state[PERIOD_KEY] == period
                        if st.button(
                            period,
                            key=f"demo_period_{period}",
                            type="primary" if selected else "secondary",
                            width="stretch",
                        ):
                            st.session_state[PERIOD_KEY] = period
                            st.rerun()
        with control_cols[2]:
            button_cols = st.columns(3, gap="small")
            with button_cols[0]:
                if st.button("▶ Iniciar", key="demo_start", type="primary", width="stretch", disabled=st.session_state[SIM_RUNNING_KEY]):
                    if st.session_state[SIM_INDEX_KEY] >= len(timeline) - 1:
                        st.session_state[SIM_INDEX_KEY] = -1
                        st.session_state[SIM_TIME_KEY] = None
                        st.session_state[SIM_ALERT_KEY] = {}
                    st.session_state[SIM_RUNNING_KEY] = True
                    st.rerun()
            with button_cols[1]:
                if st.button("Ⅱ Pausar", key="demo_pause", width="stretch", disabled=not st.session_state[SIM_RUNNING_KEY]):
                    st.session_state[SIM_RUNNING_KEY] = False
                    st.rerun()
            with button_cols[2]:
                if st.button("↺ Reiniciar", key="demo_reset", width="stretch"):
                    st.session_state[SIM_INDEX_KEY] = -1
                    st.session_state[SIM_TIME_KEY] = None
                    st.session_state[SIM_RUNNING_KEY] = False
                    st.session_state[SIM_ALERT_KEY] = {}
                    st.rerun()
        status = "REPRODUCIENDO" if st.session_state[SIM_RUNNING_KEY] else "EN PAUSA"
        st.markdown(
            f"<div class='simulator-controls-label'>ESTADO</div><span class='pill {'pill-green' if st.session_state[SIM_RUNNING_KEY] else ''}'>{status}</span>",
            unsafe_allow_html=True,
        )

    @st.fragment(run_every=0.8 if st.session_state[SIM_RUNNING_KEY] else None)
    def playback_panel():
        if st.session_state[SIM_RUNNING_KEY]:
            next_index = st.session_state[SIM_INDEX_KEY] + 1
            if next_index >= len(timeline):
                st.session_state[SIM_RUNNING_KEY] = False
                st.rerun(scope="app")
            else:
                current_time = pd.Timestamp(timeline[next_index])
                st.session_state[SIM_INDEX_KEY] = next_index
                st.session_state[SIM_TIME_KEY] = current_time

                current_frame = live_df[live_df["datetime"] == current_time].copy()
                machine_col = "machine_id" if "machine_id" in current_frame.columns else "machineID"
                current_frame["machine_id"] = current_frame[machine_col].astype(str)
                current_frame["failure_probability"] = predict_probabilities(
                    model, feature_cols, current_frame
                )
                selected_rows = current_frame[
                    current_frame["machine_id"] == str(machine_id)
                ]
                current_frame["risk_level"] = current_frame["failure_probability"].map(_risk_level)
                previous_levels = st.session_state[SIM_ALERT_KEY]
                current_levels = dict(zip(current_frame["machine_id"], current_frame["risk_level"]))
                risk_level_changes = []
                alert_levels = {"Moderado", "Critico"}
                for asset_id, level in current_levels.items():
                    previous_level = previous_levels.get(asset_id, "Estable")
                    if level != previous_level and ({level, previous_level} & alert_levels):
                        risk_level_changes.append(asset_id)
                st.session_state[SIM_ALERT_KEY] = current_levels
                if risk_level_changes:
                    st.session_state[SIM_RUNNING_KEY] = False
                    st.rerun(scope="app")

        current_time = st.session_state.get(SIM_TIME_KEY)
        if current_time is None:
            st.info("La demo está lista. Inicia la reproducción para avanzar una hora de telemetría por intervalo.")
            return

        current_rows = live_df[live_df["datetime"] == pd.Timestamp(current_time)].copy()
        machine_col = "machine_id" if "machine_id" in current_rows.columns else "machineID"
        current_rows["machine_id"] = current_rows[machine_col].astype(str)
        current_rows["failure_probability"] = predict_probabilities(
            model, feature_cols, current_rows
        )
        selected_rows = current_rows[current_rows["machine_id"] == str(machine_id)]
        if selected_rows.empty:
            st.warning(f"No hay una lectura para la máquina {machine_id} en este periodo.")
            return

        selected_row = selected_rows.iloc[0]
        probability = float(selected_row["failure_probability"])
        risk_level = _risk_level(probability)
        current_rows["risk_level"] = current_rows["failure_probability"].map(_risk_level)
        critical_count = int((current_rows["risk_level"] == "Critico").sum())
        moderate_count = int((current_rows["risk_level"] == "Moderado").sum())
        if risk_level == "Critico":
            st.error(f"Alerta critica: {machine_id} | riesgo {probability:.1%} para las proximas 24 horas.")
        elif risk_level == "Moderado":
            st.warning(f"Riesgo moderado: {machine_id} | {probability:.1%}. Se recomienda revisar la tendencia.")
        else:
            st.success(f"Monitoreo estable para {machine_id} | riesgo {probability:.1%}.")
        if critical_count or moderate_count:
            st.warning(
                f"Estado de flota en esta lectura: {critical_count} critico(s) y "
                f"{moderate_count} moderado(s). La matriz general refleja este mismo instante."
            )

        reading_cols = st.columns(5)
        readings = [
            ("RIESGO | PROXIMAS 24 H", f"{probability * 100:.1f}%", risk_level.upper()),
            ("ACTIVOS CRITICOS", critical_count, "En esta hora simulada"),
            ("ACTIVOS MODERADOS", moderate_count, "En esta hora simulada"),
            ("VIBRACION", f"{selected_row['vibration']:.2f}", "Lectura del sensor"),
            ("PRESION", f"{selected_row['pressure']:.2f}", "Lectura del sensor"),
        ]
        for column, (label, value, note) in zip(reading_cols, readings):
            with column:
                st.markdown(
                    f"<div class='simulator-reading'><div class='simulator-reading-label'>{label}</div><div class='simulator-reading-value'>{value}</div><div class='simulator-reading-note'>{note}</div></div>",
                    unsafe_allow_html=True,
                )

        alert_rows = current_rows[current_rows["risk_level"].isin(["Critico", "Moderado"])]
        if not alert_rows.empty:
            with st.expander(f"Alertas de la flota en esta lectura ({len(alert_rows)})", expanded=True):
                importances = getattr(model, "feature_importances_", None)
                if importances is not None and len(importances) == len(feature_cols):
                    signal_features = sorted(
                        zip(feature_cols, importances), key=lambda item: item[1], reverse=True
                    )[:3]
                else:
                    fallback_order = [
                        "time_since_last_error_h",
                        "distinct_errors_last_24h",
                        "hours_since_maintenance",
                    ]
                    signal_features = [(name, 0.0) for name in fallback_order if name in feature_cols]

                alert_headers = st.columns([1.1, 0.7, 2.8, 1.1], vertical_alignment="center")
                for column, label in zip(
                    alert_headers,
                    ["ACTIVO", "RIESGO", "SEÑALES DEL MODELO · VALORES ACTUALES", "ACCESO"],
                ):
                    with column:
                        st.markdown(f"<div class='eyebrow'>{label}</div>", unsafe_allow_html=True)
                st.caption(
                    "Se muestran variables con mayor importancia global en el modelo y sus valores en esta lectura; "
                    "no representan una explicación causal individual."
                )
                for _, alert in alert_rows.sort_values("failure_probability", ascending=False).head(10).iterrows():
                    alert_cols = st.columns([1.1, 0.7, 2.8, 1.1], vertical_alignment="center")
                    with alert_cols[0]:
                        st.markdown(f"**{alert['machine_id']} | {alert['risk_level']}**")
                    with alert_cols[1]:
                        st.markdown(f"{alert['failure_probability']:.1%}")
                    with alert_cols[2]:
                        signals = []
                        for feature, importance in signal_features:
                            value = alert.get(feature)
                            if pd.isna(value):
                                continue
                            value_text = f"{float(value):.2f}" if isinstance(value, (int, float)) else str(value)
                            feature_label = FEATURE_LABELS.get(feature, feature.replace("_", " "))
                            signals.append(f"{feature_label}: {value_text}")
                        st.caption(" · ".join(signals) if signals else "Variables no disponibles")
                    with alert_cols[3]:
                        if st.button("Ver en matriz", key=f"sim_matrix_{alert['machine_id']}", width="stretch"):
                            navigate_to_diagnostic_matrix(alert["machine_id"])

        chart_rows = _telemetry_window(live_df, machine_id, current_time).copy()
        chart_rows = chart_rows.rename(columns={"datetime": "timestamp", "volt": "voltage"})
        render_chart(chart_rows)
        st.caption(
            f"Lectura {st.session_state[SIM_INDEX_KEY] + 1:,} de {len(timeline):,} · "
            f"Fecha simulada: {pd.Timestamp(current_time):%Y-%m-%d %H:%M} · "
            f"Umbral del modelo: {threshold:.1%}"
        )
        if st.session_state[SIM_INDEX_KEY] >= len(timeline) - 1 and not st.session_state[SIM_RUNNING_KEY]:
            st.info("La reproducción llegó al final del periodo disponible.")

    playback_panel()
