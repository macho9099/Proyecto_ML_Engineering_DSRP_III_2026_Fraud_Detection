"""Configuracion central del proyecto: rutas, semillas y constantes del negocio."""

import os
from pathlib import Path

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = Path(
    os.environ.get("RAW_DATA_PATH", DATA_DIR / "raw" / "PS_20174392719_1491204439457_log.csv")
)
PROCESSED_DIR = DATA_DIR / "processed"
REFERENCE_DATA_PATH = PROCESSED_DIR / "reference.csv"   # step <= 400 (train/validacion offline)
PRODUCTION_DATA_PATH = PROCESSED_DIR / "production.csv"  # step > 400 (simulacion online)
MODEL_DIR = ROOT_DIR / "artifacts" / "models"
MODEL_PATH = MODEL_DIR / "fraud_model.pkl"
REPORTS_DIR = ROOT_DIR / "app" / "reports"

# --- Split temporal segun Anexo 1 del proyecto final ---
REFERENCE_MAX_STEP = 400  # step <= 400 -> referencia/entrenamiento ; step > 400 -> produccion

# --- Reproducibilidad ---
RANDOM_STATE = 42

# --- Target y features ---
TARGET_COL = "isFraud"
DROP_COLS = ["isFlaggedFraud", "nameOrig", "nameDest"]  # se descartan tras derivar features
CATEGORICAL_COLS = ["type"]

# --- MLflow ---
# Local por defecto (SQLite: el backend de archivos "./mlruns" quedo en modo
# mantenimiento desde MLflow 3.x). Para apuntar a DagsHub en el futuro solo
# hace falta exportar MLFLOW_TRACKING_URI (y credenciales) sin tocar el codigo.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-detection-paysim")
