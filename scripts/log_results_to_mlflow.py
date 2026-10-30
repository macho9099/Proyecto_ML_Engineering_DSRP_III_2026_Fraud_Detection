"""Registra en MLflow los resultados reales ya obtenidos en
notebooks/2.0-entrenamiento-modelo.ipynb (comparacion de balanceo,
baseline, modelo campeon), sin volver a correr el entrenamiento.

Util para publicar los experimentos en un backend remoto (p.ej. DagsHub)
despues de haber corrido el pipeline localmente: exportar las variables de
entorno estandar de MLflow antes de correr este script.

    export MLFLOW_TRACKING_URI=https://dagshub.com/<usuario>/<repo>.mlflow
    export MLFLOW_TRACKING_USERNAME=<usuario>
    export MLFLOW_TRACKING_PASSWORD=<token>
    python scripts/log_results_to_mlflow.py

Los valores de abajo son los resultados reales obtenidos al correr el
pipeline completo sobre el dataset de PaySim (ver README.md, seccion f).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import mlflow

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MODEL_PATH

BALANCING_COMPARISON = {
    "Original": {"precision": 1.000000, "recall": 0.994612, "f1": 0.997295, "pr_auc": 0.996068, "roc_auc": 0.998195},
    "Reweighing": {"precision": 0.984427, "recall": 0.994612, "f1": 0.989381, "pr_auc": 0.996348, "roc_auc": 0.998898},
    "Oversampling": {"precision": 0.994652, "recall": 0.994612, "f1": 0.994616, "pr_auc": 0.996238, "roc_auc": 0.998806},
}
BALANCING_STRATEGY_CHOSEN = "Reweighing"

BASELINE_OFFLINE_METRICS = {
    "accuracy": 0.33974444444444446,
    "precision": 0.0011598344294328998,
    "recall": 0.989247311827957,
    "f1": 0.0023169523681603734,
    "roc_auc": 0.7637824481236999,
    "pr_auc": 0.030086300399671095,
    "log_loss": 0.5574567914009094,
}

CHAMPION_PARAMS = {
    "subsample": 0.6,
    "n_estimators": 200,
    "min_child_weight": 3,
    "max_depth": 3,
    "learning_rate": 0.05,
    "colsample_bytree": 0.8,
}
CHAMPION_OFFLINE_METRICS = {
    "accuracy": 0.9999361111111111,
    "precision": 0.935374149659864,
    "recall": 0.985663082437276,
    "f1": 0.9598603839441536,
    "roc_auc": 0.9961009407938305,
    "pr_auc": 0.9857329016144031,
    "log_loss": 0.00048263807548210025,
}
CHAMPION_ONLINE_METRICS = {
    "accuracy": 0.9999583036536424,
    "precision": 0.9986566362170876,
    "recall": 0.994914346895075,
    "f1": 0.996781979082864,
    "roc_auc": 0.999970034354391,
    "pr_auc": 0.9990374842796104,
    "log_loss": 0.00039267324609681964,
}


def log_metrics(prefix: str, metrics: dict) -> None:
    for name, value in metrics.items():
        mlflow.log_metric(f"{prefix}{name}", float(value))


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print("Tracking URI:", mlflow.get_tracking_uri())

    with mlflow.start_run(run_name="class_balancing_comparison"):
        mlflow.log_param("balancing_strategy_chosen", BALANCING_STRATEGY_CHOSEN)
        for estrategia, metrics in BALANCING_COMPARISON.items():
            log_metrics(f"{estrategia.lower()}_", metrics)
    print("Run 'class_balancing_comparison' registrado.")

    with mlflow.start_run(run_name="baseline_logreg"):
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("balancing_strategy", BALANCING_STRATEGY_CHOSEN)
        log_metrics("offline_", BASELINE_OFFLINE_METRICS)
    print("Run 'baseline_logreg' registrado.")

    with mlflow.start_run(run_name="champion_xgboost_random_search"):
        mlflow.log_param("model", "XGBClassifier")
        mlflow.log_param("hpo_method", "random_search")
        mlflow.log_param("balancing_strategy", BALANCING_STRATEGY_CHOSEN)
        for k, v in CHAMPION_PARAMS.items():
            mlflow.log_param(k, v)
        log_metrics("offline_", CHAMPION_OFFLINE_METRICS)
        log_metrics("online_", CHAMPION_ONLINE_METRICS)

        payload = joblib.load(MODEL_PATH)
        mlflow.xgboost.log_model(payload["model"], artifact_path="model")
    print("Run 'champion_xgboost_random_search' registrado (con modelo).")


if __name__ == "__main__":
    main()
