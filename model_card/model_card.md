# Model Card — Fraud Detection (PaySim)

Formato inspirado en [Kaggle Model Cards](https://www.kaggle.com/code/var0101/model-cards).

## 1. Detalles del modelo

| Campo | Valor |
|---|---|
| Desarrollado por | Juan Carlos Aguilar Alfaro — DataScienceResearchPeru, Especializacion MLE, Curso III |
| Fecha | 2026-10-27 |
| Tipo de modelo | `XGBClassifier` (gradient boosted trees), binario |
| Version | v1.0.0 |
| Hiperparametros | `n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.6, colsample_bytree=0.8, min_child_weight=3` — elegidos con Random Search (`RandomizedSearchCV`, Sesion 3) — ver `notebooks/2.0-entrenamiento-modelo.ipynb` y el experimento registrado en MLflow |
| Balanceo de clases | Reweighing (Sesion 8) — ganador sobre Original y Oversampling en comparacion 3-fold CV |
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

- Dataset: PaySim (`PS_20174392719_1491204439457_log.csv`, 6.36M filas).
- Entrenamiento/validacion offline: `step <= 400` — por limite de computo
  del equipo de desarrollo se usa una submuestra estratificada de 1.8M de
  las ~5.79M filas de referencia (misma tasa real de fraude, ~0.077%).
- Evaluacion online (produccion simulada): `step > 400`, **100%** de las
  575,590 filas (sin submuestrear).
- Ver diccionario completo en [`data/README.md`](../data/README.md).

## 4. Metricas

| Metrica | Offline (holdout) | Online (produccion completa) |
|---|---|---|
| PR-AUC (principal) | 0.9857 | 0.9990 |
| Recall | 0.9857 | 0.9949 |
| Precision | 0.9354 | 0.9987 |
| F1 | 0.9599 | 0.9968 |
| ROC-AUC | 0.9961 | 0.99997 |

Ver tabla completa (incluye accuracy/log loss y el baseline lineal) en el
`README.md` principal (seccion *f. Resultados*). Se reporta **PR-AUC**
como metrica principal (el target esta fuertemente desbalanceado, ~0.08%
de fraude en referencia; ROC-AUC/accuracy pueden ser optimistas).

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

**Hallazgo clave**: sin data drift en las features de entrada, pero con
un target drift fuerte (tasa de fraude ~8.4x mayor en produccion que en
referencia). El desempeno del modelo se mantiene estable pese a eso — se
recomienda monitoreo activo de la tasa de fraude y el PR-AUC por ventana
de tiempo en vez de retraining inmediato (ver seccion *h. Conclusiones*
del README).
