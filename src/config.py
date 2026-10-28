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

# dtypes explicitos para leer reference.csv/production.csv: sobre el dataset
# completo (~5.8M filas de referencia) el default de pandas (float64/int64)
# practicamente duplica el uso de memoria vs. float32/int8. Columnas nuevas
# que no esten aqui (p.ej. fraud_proba/fraud_pred agregadas en el notebook
# de entrenamiento) se infieren normalmente -- pd.read_csv(dtype=...) solo
# aplica el tipo a las columnas presentes en el dict.
PROCESSED_DTYPES = {
    "step": "int32",
    "amount": "float32",
    "oldbalanceOrg": "float32",
    "newbalanceOrig": "float32",
    "oldbalanceDest": "float32",
    "newbalanceDest": "float32",
    "isFraud": "int8",
    "errorBalanceOrig": "float32",
    "errorBalanceDest": "float32",
    "amountToOldBalanceOrgRatio": "float32",
    "origAccountEmptied": "int8",
    "isDestMerchant": "int8",
    "destBalanceUnknown": "int8",
    "type_CASH_IN": "int8",
    "type_CASH_OUT": "int8",
    "type_DEBIT": "int8",
    "type_PAYMENT": "int8",
    "type_TRANSFER": "int8",
}

# --- MLflow ---
# Local por defecto (SQLite: el backend de archivos "./mlruns" quedo en modo
# mantenimiento desde MLflow 3.x). Para apuntar a DagsHub en el futuro solo
# hace falta exportar MLFLOW_TRACKING_URI (y credenciales) sin tocar el codigo.
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT_DIR / 'mlflow.db'}")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "fraud-detection-paysim")
