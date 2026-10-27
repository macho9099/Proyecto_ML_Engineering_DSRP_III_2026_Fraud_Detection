# Model Card — Fraud Detection (PaySim)

Formato inspirado en [Kaggle Model Cards](https://www.kaggle.com/code/var0101/model-cards).

## 1. Detalles del modelo

| Campo | Valor |
|---|---|
| Desarrollado por | Juan Carlos Aguilar Alfaro — DataScienceResearchPeru, Especializacion MLE, Curso III |
| Fecha | _(completar tras el entrenamiento final)_ |
| Tipo de modelo | `XGBClassifier` (gradient boosted trees), binario |
| Version | v1.0.0 |
| Hiperparametros | Optimizados con Random Search (`RandomizedSearchCV`) — ver `notebooks/2.0-entrenamiento-modelo.ipynb` y el experimento registrado en MLflow |
| Licencia | MIT (ver `LICENSE`) |

## 2. Uso previsto

- **Uso primario**: priorizar transacciones de dinero movil (`TRANSFER`,
  `CASH_OUT`) para revision por un analista de fraude, a partir de una
  probabilidad de fraude (`fraud_proba`).
- **Usuarios previstos**: equipo de riesgo/fraude, ingenieros de ML que dan
  mantenimiento al pipeline.
- **Fuera de alcance**: no esta pensado como bloqueador automatico de
  transacciones sin revision humana; no se ha validado sobre datos reales
  de produccion (el dataset es una simulacion, PaySim).

## 3. Datos de entrenamiento y evaluacion

- Dataset: PaySim (`PS_20174392719_1491204439457_log.csv`).
- Entrenamiento/validacion offline: `step <= 400`.
- Evaluacion online (produccion simulada): `step > 400`.
- Ver diccionario completo en [`data/README.md`](../data/README.md).

## 4. Metricas

Ver tabla de resultados en el `README.md` principal (seccion *Resultados*).
Se reporta **PR-AUC** como metrica principal (el target esta fuertemente
desbalanceado, ROC-AUC puede ser optimista), ademas de precision, recall,
F1, ROC-AUC y log loss, offline y online.

## 5. Consideraciones eticas y limitaciones

- El dataset es una **simulacion**; los patrones de fraude pueden no
  generalizar 1:1 a un sistema financiero real.
- Un falso positivo bloquea/retrasa una transaccion legitima de un
  usuario real — el umbral de decision (0.5 por defecto) deberia
  calibrarse segun el costo de negocio real de falsos positivos vs.
  falsos negativos (ver curva Precision-Recall en el notebook de
  entrenamiento).
- No se evaluo fairness entre subgrupos de usuarios porque el dataset no
  trae atributos demograficos; si se integrara con datos reales deberia
  auditarse con las tecnicas vistas en la Sesion 7-8 del curso
  (Fairlearn/AIF360, reweighing, equalized odds).

## 6. Interpretabilidad

Analisis global (SHAP summary, PDP) y local (SHAP waterfall, LIME,
contrafactuales DiCE) en
[`notebooks/3.0-interpretabilidad.ipynb`](../notebooks/3.0-interpretabilidad.ipynb).

## 7. Monitoreo

Data drift, target drift y degradacion de desempeno a lo largo del tiempo
en [`notebooks/4.0-monitoreo-drift.ipynb`](../notebooks/4.0-monitoreo-drift.ipynb)
y en la app [`app/monitoring_app.py`](../app/monitoring_app.py).
