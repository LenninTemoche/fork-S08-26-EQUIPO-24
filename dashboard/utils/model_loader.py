"""Carga del modelo ML serializado (joblib) desde GitHub.

El archivo `models/baseline_model.joblib` se aloja en la rama
`feat/modeling_integration` del repositorio y contiene:

- model: RandomForestClassifier entrenado.
- feature_cols: lista de 46 features en el orden esperado.
- model_type, decision_threshold, pr_auc, roc_auc, precision, recall, f1_score.
- metadatos temporales (train_start, train_end, test_start, test_end).

No se reentrena el modelo en el dashboard; solo se carga y se usa.
"""

import io
import os
import warnings

import joblib
import pandas as pd
import streamlit as st
from urllib.request import Request, urlopen

# ── Configuración ──────────────────────────────────────────────
MODEL_URL = (
    "https://raw.githubusercontent.com/No-Country-simulation/"
    "S08-26-EQUIPO-24/feat/modeling_integration/models/"
    "baseline_model.joblib"
)

# Ruta local de respaldo (útil para desarrollo sin internet).
LOCAL_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "models", "baseline_model.joblib",
)


def _load_from_url(url: str) -> dict:
    """Descarga y deserializa el joblib desde una URL HTTP."""
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=120) as response:
        model_bytes = response.read()
    return joblib.load(io.BytesIO(model_bytes))


def _load_from_local(path: str) -> dict:
    """Carga el joblib desde el sistema de archivos local."""
    return joblib.load(path)


def load_model_artifact(prefer_local: bool = False):
    """Carga el artefacto del modelo (con cacheo en Streamlit).

    Estrategia:
    1. Intenta descargar desde GitHub (fuente de verdad en producción).
    2. Si falla, usa la copia local `models/baseline_model.joblib`.

    Returns:
        dict con al menos las claves: model, feature_cols, model_type,
        decision_threshold y las métricas de evaluación.
    """
    # Si el usuario prefiere local, intentar cargar desde disco primero.
    if prefer_local:
        local = os.path.normpath(LOCAL_MODEL_PATH)
        if os.path.exists(local):
            artifact = _load_from_local(local)
            source = f"Local ({local})"
        else:
            raise RuntimeError(
                "Se solicitó modo local pero no se encontró el artefacto en: "
                f"{LOCAL_MODEL_PATH}."
            )
    else:
        try:
            artifact = _load_from_url(MODEL_URL)
            source = "GitHub (feat/modeling_integration)"
        except Exception:
            # Fallback a archivo local si existe.
            local = os.path.normpath(LOCAL_MODEL_PATH)
            if os.path.exists(local):
                artifact = _load_from_local(local)
                source = f"Local ({local})"
            else:
                raise RuntimeError(
                    "No se pudo cargar el modelo desde GitHub ni desde "
                    f"{LOCAL_MODEL_PATH}. Verifica la conexión a internet."
                )

    # Validaciones básicas del artefacto.
    required_keys = {"model", "feature_cols", "model_type"}
    missing = required_keys - set(artifact.keys())
    if missing:
        raise ValueError(
            f"El artefacto del modelo no contiene las claves requeridas: {missing}"
        )

    return artifact, source


@st.cache_resource(ttl=3600, show_spinner=False)
def _cached_load():
    """Wrapper cacheado para evitar descargas repetidas.

    Uso `st.cache_resource` porque el artefacto del modelo es un
    recurso (objeto pesado) y conviene almacenarlo en memoria entre
    ejecuciones para mejorar latencia en el dashboard.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_model_artifact()


def get_model(prefer_local: bool = False):
    """Retorna el modelo cacheado y sus metadatos.

    Returns:
        model: estimator entrenado.
        feature_cols: list[str] de features.
        meta: dict con threshold, métricas y metadatos temporales.
        source: str con el origen del archivo cargado.
    """
    # Si se solicita prefer_local, evitar el cache (uso en desarrollo)
    if prefer_local:
        artifact, source = load_model_artifact(prefer_local=True)
    else:
        artifact, source = _cached_load()

    model = artifact["model"]
    feature_cols = list(artifact["feature_cols"])

    meta = {
        "model_type": artifact.get("model_type", type(model).__name__),
        "decision_threshold": float(artifact.get("decision_threshold", 0.5)),
        "pr_auc": artifact.get("pr_auc"),
        "roc_auc": artifact.get("roc_auc"),
        "precision": artifact.get("precision"),
        "recall": artifact.get("recall"),
        "f1_score": artifact.get("f1_score"),
        "train_start": artifact.get("train_start"),
        "train_end": artifact.get("train_end"),
        "test_start": artifact.get("test_start"),
        "test_end": artifact.get("test_end"),
        "positive_rate_train": artifact.get("positive_rate_train"),
        "positive_rate_test": artifact.get("positive_rate_test"),
    }

    return model, feature_cols, meta, source


def predict_probabilities(model, feature_cols, df: pd.DataFrame) -> pd.Series:
    """Calcula la probabilidad de falla para cada fila de `df`.

    Args:
        model: estimator cargado.
        feature_cols: lista de features esperadas.
        df: DataFrame con las features (se extraen las columnas esperadas).

    Returns:
        pd.Series con la probabilidad de la clase positiva (falla).
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Faltan features requeridas por el modelo: {missing}"
        )

    X = df[feature_cols]
    classes = list(getattr(model, "classes_", []))
    if classes:
        try:
            positive_index = classes.index(1)
        except ValueError as exc:
            raise ValueError(
                f"El modelo no contiene la clase positiva 1 (clases: {classes})."
            ) from exc
    else:
        positive_index = 1

    probabilities = model.predict_proba(X)
    if probabilities.ndim != 2 or probabilities.shape[1] <= positive_index:
        raise ValueError("El modelo devolvió probabilidades con forma inesperada.")
    probs = probabilities[:, positive_index]
    return pd.Series(probs, index=df.index, name="failure_probability")


def predict_binary(model, feature_cols, df: pd.DataFrame, threshold: float) -> pd.Series:
    """Predicción binaria usando el umbral de decisión del modelo."""
    if not 0 <= threshold <= 1:
        raise ValueError(f"El umbral debe estar entre 0 y 1; recibido: {threshold}.")
    probs = predict_probabilities(model, feature_cols, df)
    return (probs >= threshold).astype(int)
