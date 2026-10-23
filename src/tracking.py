"""Utilidades delgadas sobre MLflow para loguear experimentos de forma consistente.

Por defecto apunta a un tracking local (./mlruns). Para usar DagsHub como
backend remoto basta con exportar las variables de entorno estandar de
MLflow antes de correr los notebooks/scripts, sin tocar este modulo:

    export MLFLOW_TRACKING_URI=https://dagshub.com/<user>/<repo>.mlflow
    export MLFLOW_TRACKING_USERNAME=<user>
    export MLFLOW_TRACKING_PASSWORD=<token>
"""

import mlflow

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI


def init_mlflow() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def log_classification_metrics(prefix: str, metrics: dict) -> None:
    """Loguea un dict de metricas con un prefijo (p.ej. 'offline_' u 'online_')."""
    for name, value in metrics.items():
        mlflow.log_metric(f"{prefix}{name}", float(value))
