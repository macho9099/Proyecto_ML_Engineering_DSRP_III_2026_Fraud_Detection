"""Feature engineering para el dataset PaySim de deteccion de fraude.

Centraliza la logica de transformacion para que el notebook de
preprocesamiento, el de entrenamiento y la app de monitoreo apliquen
exactamente las mismas reglas (evita entrenar/servir con features distintas).
"""

import numpy as np
import pandas as pd

EPS = 1e-6


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features derivadas sugeridas en el Anexo 1 del proyecto final.

    No modifica el DataFrame de entrada; retorna una copia.
    """
    out = df.copy()

    # Consistencia de saldos del originador y del destinatario.
    out["errorBalanceOrig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["errorBalanceDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]

    # Que porcentaje del saldo origen se mueve en la transaccion.
    out["amountToOldBalanceOrgRatio"] = out["amount"] / (out["oldbalanceOrg"] + EPS)

    # Si la cuenta origen queda en cero tras la transaccion (vaciado de cuenta).
    out["origAccountEmptied"] = (
        (out["oldbalanceOrg"] > 0) & (np.isclose(out["newbalanceOrig"], 0))
    ).astype(int)

    # El destinatario es un comercio (merchant) si su id empieza con "M".
    out["isDestMerchant"] = out["nameDest"].astype(str).str.startswith("M").astype(int)

    # PaySim no reporta saldo de merchants -> lo marcamos explicitamente.
    out["destBalanceUnknown"] = (
        (out["oldbalanceDest"] == 0) & (out["newbalanceDest"] == 0) & (out["isDestMerchant"] == 1)
    ).astype(int)

    return out


def select_model_features(df: pd.DataFrame) -> list[str]:
    """Columnas numericas/dummy que entran al modelo (excluye ids e isFlaggedFraud)."""
    exclude = {"isFraud", "isFlaggedFraud", "nameOrig", "nameDest", "step"}
    dummy_cols = [c for c in df.columns if c.startswith("type_")]
    base_cols = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "errorBalanceOrig",
        "errorBalanceDest",
        "amountToOldBalanceOrgRatio",
        "origAccountEmptied",
        "isDestMerchant",
        "destBalanceUnknown",
    ]
    return [c for c in base_cols + dummy_cols if c in df.columns and c not in exclude]


def build_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encoding de `type` (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER).

    `pd.get_dummies` genera columnas dtype `bool` desde pandas 2.x; se
    castean a `int` (0/1) para evitar valores `True`/`False` aguas abajo
    (p.ej. DiCE los serializa a texto y rompe la prediccion de XGBoost).
    """
    dummies = pd.get_dummies(df, columns=["type"], prefix="type")
    dummy_cols = [c for c in dummies.columns if c.startswith("type_")]
    dummies[dummy_cols] = dummies[dummy_cols].astype(int)
    return dummies
