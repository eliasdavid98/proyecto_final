# Registro de Cambios y Decisiones Actualizadas (PSets #1 a #4)

Este documento registra las correcciones aplicadas a las observaciones de los PSets #1, #2 y #3, así como las decisiones de negocio, datos y operación que evolucionaron a lo largo del proyecto end-to-end.

## 1. Decisiones actualizadas entre PSets

| Dimensión | Decisión inicial (PSet #1 / #2 / #3) | Decisión vigente (PSet #4) | Archivos actualizados |
| :--- | :--- | :--- | :--- |
| **Dominio y fuente de datos** | Propuesta inicial enfocada en churn B2B de telecomunicaciones (IBM Telco). | Se formaliza el dominio de **churn bancario** correspondiente a la OBT real en Snowflake (`CHURN_DB.GOLD.OBT_CUSTOMER_CHURN`, 25.000 filas: 15.000 `train` etiquetadas y 10.000 `test` sin etiqueta). | `README.md`, `dbt/`, `spark/build_obt.py`, `docs/PSet4_memo_pogo_torres.pdf` |
| **Línea base de negocio** | Regla heurística de telecomunicaciones (>2 tickets técnicos/mes o mora >30 días). | Regla determinística bancaria sin ML: contactar si `number_of_products >= 3` o (`age >= 40` y `is_active_member == 0`). Opera también como respaldo automático y manual (`PREDICTION_MODE=business_rule`). | `api/main.py`, `scripts/run_production_batch.py`, `tests/test_comportamiento.py` |
| **Métrica de negocio y ROI** | Proyección teórica del PSet #1 (reducción de 15% del churn) sin costos unitarios en el dataset. | Función explícita de valor neto por lote: `$300 USD` por retención exitosa (`TP`) menos `$25 USD` por contacto (`TP + FP`). En los 8 lotes simulados el modelo generó **$111.550 USD** vs. **$100.100 USD** de la regla base (**+$11.450 USD / +11,44%**) y **$0 USD** de no hacer nada. | `scripts/run_production_batch.py`, `src/reentrenar.py`, `dashboard/app.py`, `docs/monitoreo.csv` |
| **Umbral de decisión** | Umbral propuesto de `0,60` en el PSet #1 sin validación empírica. | Umbral operativo fijado en `0,50` (`CHURN_DECISION_THRESHOLD=0.5`), coherente con el `F2 = 0,6936` auditado en el holdout del PSet #3 y usado en la API y el monitoreo. | `api/main.py`, `src/reentrenar.py`, `scripts/run_production_batch.py` |
| **Grano de la OBT y clave primaria** | Granularidad a nivel de cliente individual (`CustomerId`). | Grano fijado en una fila por observación (`observation_id`), dado que `CustomerId` presenta 3.454 valores repetidos con atributos inconsistentes y 1 valor no entero (`id=5597`, `156917.93`) en `train`. | `dbt/models/silver/`, `dbt/models/gold/`, `spark/build_obt.py` |
| **Infraestructura y orquestación** | Kestra y Spark en el PSet #2 separados del servidor local de MLflow del PSet #3. | Un único `docker-compose.yml` levanta Postgres, Kestra v2.0.10, MLflow, Spark (perfil `spark`), API FastAPI y Dashboard Streamlit. | `docker-compose.yml`, `.env.example` |

---

## 2. Correcciones a observaciones de PSets anteriores

### PSet #1 (Propuesta y Pitch)
* **Observación:** Desajuste entre el caso de negocio B2B telecom planteado en el pitch y los datos bancarios disponibles, así como cifras proyectadas sin respaldo empírico.
* **Cambio aplicado:** Se reemplazaron todas las cifras proyectadas del pitch por los resultados reales de la simulación de 8 lotes (`+$11.450 USD`, `+11,44%` sobre la regla base) y se ajustó la presentación ejecutiva final al dominio bancario real.
* **Archivos:** `docs/PSet4_presentacion_pogo_torres.pdf`, `docs/PSet4_memo_pogo_torres.pdf`.

### PSet #2 (Ingesta, Calidad, dbt y Spark)
* **Observación:** El documento técnico del PSet #2 se encontraba en formato `.docx` en el historial y la orquestación entre etapas estaba fragmentada.
* **Cambio aplicado:** Se convirtió `docs/historial/PSet2_memo_pogo_torres.pdf` a PDF, se unificaron los flujos de Kestra en `kestra/flows/` y se mantuvo el linaje completo (`RAW` → `BRONZE` → `SILVER` → `GOLD` → `OBT`) con trazabilidad de `OBT_SNAPSHOT_ID` (`HASH_AGG(*)`) en cada predicción y lote monitoreado.
* **Archivos:** `docs/historial/PSet2_memo_pogo_torres.pdf`, `kestra/flows/ingesta.yml`, `scripts/run_production_batch.py`, `api/main.py`.

### PSet #3 (Experimentación, Tracking y Registro)
* **Observación 1:** MLflow corría fuera de Docker Compose y las corridas iniciales no capturaban automáticamente el snapshot de datos ni el ciclo de reentrenamiento.
* **Cambio aplicado:** Se integró MLflow en `docker-compose.yml` (puerto `5001` en el host, `MLFLOW_SERVER_ALLOWED_HOSTS` habilitado para `host.docker.internal`), se preservó el historial consolidado de las 103 corridas originales más la corrida de reentrenamiento (`fase='reentrenamiento'`) en `docs/corridas.csv` y `docs/reentrenamientos.csv`.
* **Archivos:** `docker-compose.yml`, `src/reentrenar.py`, `scripts/export_runs.py`, `docs/corridas.csv`, `docs/reentrenamientos.csv`.
* **Observación 2:** Los notebooks requerían ejecución limpia de principio a fin con salidas guardadas y configuración compatible con el servidor MLflow de Compose.
* **Cambio aplicado:** Se ejecutaron todos los notebooks en `notebooks/` con *Restart & Run All* apuntando a `http://localhost:5001` y se guardaron con todas sus salidas.
* **Archivos:** `notebooks/*.ipynb`.

---

## 3. Cambios implementados durante el cierre del PSet #4

* **Simulación etiquetada y monitoreo de deriva conceptual (`scripts/run_production_batch.py`, `monitoreo/alertas.py`):** Se migró la simulación al holdout etiquetado en 8 lotes semanales sintéticos (`n = 375` filas/lote), induciendo deriva de datos y conceptual en los lotes 6 y 7 (`balance += 50.000`, `geography = 'Germany'`, inversión de 33% en `is_active_member` y `exited`). Esto permitió medir `ROC-AUC`, `Average Precision`, `F2`, `MAX_PSI` y valor neto de negocio lote por lote.
* **Respaldo dual e interruptor de retirada (`api/main.py`):** Se incorporó la variable `PREDICTION_MODE` (`model` | `business_rule`) para conmutar a la regla de negocio sin ML de forma controlada ante incidentes, además del respaldo automático ante fallos de carga del modelo.
* **Ciclo completo de reentrenamiento con aprobación humana (`src/reentrenar.py`, `kestra/flows/reentrenamiento.yml`):** Se implementó el reentrenamiento sobre la ventana `development` + lotes 1–7, evaluando `champion` (v1) vs. `challenger` (v3, `run_id: 15d2b1befb6d42e598aa0b6d96a87fd5`) en el lote 8. Tras la pausa humana (`Pause` con input `BOOL`) en Kestra, v3 fue **rechazado** porque, aunque subió `ROC-AUC` (`+0,0029`), cayó en `F2` (`-0,0125`) y redujo el valor neto en `-$300 USD` (`$13.725` vs. `$14.025`).
* **Visualización lote por lote (`dashboard/app.py`):** Se añadieron 4 gráficos en Streamlit (latencia p50/p95, desempeño `ROC-AUC`/`AP`/`F2` vs. límite `CV - 2*SD`, deriva `MAX_PSI` vs. umbral `0,25` y valor neto de negocio Modelo vs. Regla base).
* **Higiene del repositorio (`.gitignore`):** Se eliminaron archivos `.DS_Store` y se excluyeron en `.gitignore`.