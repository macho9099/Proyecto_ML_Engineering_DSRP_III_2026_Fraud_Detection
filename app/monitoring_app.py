"""App de monitoreo de modelos (Streamlit + Evidently).

Replica el estilo del dashboard "Report: Data drifts" de Evidently AI
sugerido en el enunciado del proyecto: tabla por feature con tipo,
distribucion referencia/actual, flag de drift, test estadistico usado y
drift score, mas un "semaforo ejecutivo" (Sesion 10) con el resumen global.

Ejecutar desde la raiz del repo:

    streamlit run app/monitoring_app.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st

# Nota: en evidently >= 0.6 la API "clasica" (Report + metric_preset) vive
# bajo el namespace evidently.legacy; es el shim oficial de compatibilidad
# hacia atras, mismo API que se vio en el curso (Sesion 9-10).
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.report import Report

from src.config import PROCESSED_DIR, REFERENCE_DATA_PATH, TARGET_COL
from src.features import select_model_features
from src.metrics import classification_metrics

st.set_page_config(page_title="Monitoreo — Fraud Detection", layout="wide")


@st.cache_data
def load_data():
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    scored_path = PROCESSED_DIR / "production_scored.csv"
    production = pd.read_csv(scored_path if scored_path.exists() else PROCESSED_DIR / "production.csv")
    return reference, production


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[features], current_data=current[features])
    # DataDriftPreset expande a 2 metricas (DatasetDriftMetric + DataDriftTable);
    # la tabla por columna vive en la que trae "drift_by_columns".
    metrics_dict = report.as_dict()["metrics"]
    result = next(m["result"] for m in metrics_dict if "drift_by_columns" in m["result"])

    rows = []
    for feature, info in result["drift_by_columns"].items():
        rows.append(
            {
                "Feature": feature,
                "Tipo": info.get("column_type"),
                "Drift Score": round(info.get("drift_score", 0.0), 4),
                "Drift Detectado": "🔴 Si" if info.get("drift_detected") else "🟢 No",
                "Test estadistico": info.get("stattest_name"),
            }
        )
    return pd.DataFrame(rows).sort_values("Drift Score", ascending=False)


def semaforo(pct_drifted: float) -> tuple[str, str]:
    if pct_drifted < 0.10:
        return "🟢", "Estable — sin accion requerida"
    if pct_drifted < 0.30:
        return "🟡", "Atencion — monitorear de cerca / evaluar re-entrenamiento"
    return "🔴", "Critico — drift extendido, recomendar retraining / challenger"


def main():
    st.title("📊 Monitoreo de Modelo — Deteccion de Fraude (PaySim)")
    st.caption(
        "Referencia: step ≤ 400 (entrenamiento) · Produccion: step > 400 (simulacion online)"
    )

    reference, production = load_data()
    features = select_model_features(reference)

    with st.sidebar:
        st.header("Filtros")
        if "step" in production.columns:
            step_min, step_max = int(production["step"].min()), int(production["step"].max())
            step_range = st.slider(
                "Ventana de produccion (step)", step_min, step_max, (step_min, step_max)
            )
            current = production[production["step"].between(*step_range)]
        else:
            current = production
        st.markdown("---")
        st.caption(f"{len(current):,} transacciones en la ventana seleccionada")

    tab_drift, tab_perf, tab_about = st.tabs(
        ["🔎 Data Drift", "🎯 Desempeno del modelo", "ℹ️ Acerca de"]
    )

    with tab_drift:
        st.subheader("Report: Data Drifts")
        table = drift_table(reference, current, features)
        pct_drifted = (table["Drift Detectado"] == "🔴 Si").mean()

        emoji, mensaje = semaforo(pct_drifted)
        c1, c2, c3 = st.columns(3)
        c1.metric("Features con drift", f"{int(pct_drifted * len(table))}/{len(table)}")
        c2.metric("% features con drift", f"{pct_drifted:.1%}")
        c3.markdown(f"### {emoji} {mensaje}")

        search = st.text_input("🔍 Buscar feature", "")
        view = table[table["Feature"].str.contains(search, case=False)] if search else table
        st.dataframe(view, use_container_width=True, hide_index=True)

    with tab_perf:
        st.subheader("Desempeno online vs offline")
        if TARGET_COL in current.columns and "fraud_pred" in current.columns:
            m = classification_metrics(
                current[TARGET_COL], current["fraud_pred"], current["fraud_proba"]
            )
            cols = st.columns(len(m))
            for col, (name, value) in zip(cols, m.items()):
                col.metric(name.upper(), f"{value:.4f}")
        else:
            st.info(
                "Corre `notebooks/2.0-entrenamiento-modelo.ipynb` para generar "
                "`data/processed/production_scored.csv` con las predicciones del modelo."
            )

    with tab_about:
        st.markdown(
            """
            Esta app acompana al **Notebook de monitoreo**
            (`notebooks/4.0-monitoreo-drift.ipynb`) y a los reportes HTML
            generados en `app/reports/`. Calcula el drift de distribuciones
            (test de Kolmogorov-Smirnov / PSI segun el tipo de variable,
            via Evidently) comparando la ventana de referencia usada para
            entrenar contra la ventana de produccion seleccionada.
            """
        )


if __name__ == "__main__":
    main()
