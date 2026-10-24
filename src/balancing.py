"""Tecnicas de balanceo de clases vistas en la Sesion 8 del curso
(`mod0_Reweighing.ipynb`, `mod1_Oversampling.ipynb`), generalizadas de su
contexto original (equilibrar un grupo sensible para fairness) al
contexto de este proyecto: equilibrar la clase minoritaria del target
(`isFraud`).

No se usa ninguna libreria de balanceo externa (p.ej. `imbalanced-learn`)
porque no fue parte del curso; solo `numpy`/`sklearn.utils`, igual que en
los notebooks originales.
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample


def reweighting_sample_weights(y: pd.Series) -> np.ndarray:
    """Pesos por muestra inversamente proporcionales a la frecuencia de su clase.

    Misma mecanica que `mod0_Reweighing.ipynb` (pesos pasados como
    `sample_weight` a `.fit()`), generalizando el peso fijo 2.0/1.0 del
    ejemplo de fairness a la razon real de desbalance `n_negativos / n_positivos`.
    """
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    weight_pos = n_neg / n_pos
    return np.where(y == 1, weight_pos, 1.0)


def oversample_minority(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """Duplica (con reemplazo) la clase minoritaria hasta igualar a la mayoritaria.

    Misma tecnica que `mod1_Oversampling.ipynb` (`sklearn.utils.resample`),
    aplicada aqui sobre la clase objetivo `isFraud` en vez de un grupo
    demografico.
    """
    df = X.copy()
    df["_target"] = y.values

    majority = df[df["_target"] == 0]
    minority = df[df["_target"] == 1]

    minority_upsampled = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=random_state,
    )

    balanced = pd.concat([majority, minority_upsampled]).sample(
        frac=1, random_state=random_state
    )
    y_balanced = balanced.pop("_target")
    return balanced, y_balanced
