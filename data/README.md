# Datos — Diccionario y origen

## Origen

Dataset **PaySim** (simulador de transacciones de dinero movil), archivo
`PS_20174392719_1491204439457_log.csv`, indicado como obligatorio en el
enunciado del proyecto final (Anexo 1). No se versiona en el repo por su
tamano (~470 MB) — ver [`.gitignore`](../.gitignore) y las instrucciones de
descarga en el `README.md` principal.

## Split temporal usado en el proyecto

| Archivo | Filtro | Uso |
|---|---|---|
| `data/processed/reference.csv` | `step <= 400` | Entrenamiento + validacion offline |
| `data/processed/production.csv` / `production_scored.csv` | `step > 400` | Simulacion de produccion, evaluacion online y monitoreo (drift) |

`step` = 1 hora de simulacion; el dataset cubre 744 steps (30 dias).

## Diccionario de datos (columnas originales)

| Columna | Descripcion |
|---|---|
| `step` | Unidad de tiempo (1 step = 1 hora). Rango 1-744. |
| `type` | Tipo de transaccion: `CASH-IN`, `CASH-OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`. |
| `amount` | Monto de la transaccion, en moneda local. |
| `nameOrig` | Cliente que origina la transaccion. |
| `oldbalanceOrg` | Saldo del originador antes de la transaccion. |
| `newbalanceOrig` | Saldo del originador despues de la transaccion. |
| `nameDest` | Cliente/merchant destinatario. |
| `oldbalanceDest` | Saldo del destinatario antes de la transaccion (no disponible para merchants, prefijo `M`). |
| `newbalanceDest` | Saldo del destinatario despues de la transaccion (no disponible para merchants). |
| `isFraud` | **Target.** 1 si la transaccion fue fraudulenta (toma de control de cuenta + vaciado via transferencia). |
| `isFlaggedFraud` | Regla de negocio existente: marca intentos de transferir > 200,000 en una sola transaccion. Se excluye de los features del modelo para evitar fuga de informacion (es una regla, no una senal predictiva independiente). |

## Features derivadas (`src/features.py`)

| Feature | Definicion | Motivacion |
|---|---|---|
| `errorBalanceOrig` | `newbalanceOrig + amount - oldbalanceOrg` | Inconsistencias contables en la cuenta origen suelen aparecer en fraude. |
| `errorBalanceDest` | `oldbalanceDest + amount - newbalanceDest` | Idem para la cuenta destino. |
| `amountToOldBalanceOrgRatio` | `amount / (oldbalanceOrg + eps)` | Que porcentaje del saldo se mueve de una vez (vaciado de cuenta). |
| `origAccountEmptied` | 1 si `oldbalanceOrg > 0` y `newbalanceOrig ≈ 0` | Patron tipico descrito en el Anexo 1. |
| `isDestMerchant` | 1 si `nameDest` empieza con `M` | El fraude en PaySim no ocurre hacia merchants (sirve como filtro de negocio). |
| `destBalanceUnknown` | 1 si los saldos del destino son 0 y es merchant | Marca explicita del missing-by-design de PaySim. |
| `type_*` | One-hot de `type` | El fraude solo ocurre en `TRANSFER`/`CASH_OUT`. |

`nameOrig` y `nameDest` (ids de alta cardinalidad) no entran directo al
modelo; solo se usan para derivar `isDestMerchant`.
