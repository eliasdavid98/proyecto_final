# Proyecto final PSet 1–4: churn

Este proyecto reúne los artefactos de ingesta, transformación, entrenamiento y servicio de los PSet anteriores, y añade los componentes operativos implementados en el PSet 4.

## Alcance y límites de interpretación

El caso de negocio propuesto en el PSet 1 fue retención B2B en telecomunicaciones. Por decisión del autor, PSet 2, PSet 3 y este proyecto conservan la OBT bancaria `CHURN_DB.GOLD.OBT_CUSTOMER_CHURN`, con la etiqueta `exited`. Por ello:

- las métricas y simulaciones de este repositorio son una demostración técnica sobre churn bancario; no validan predicción a 30–60 días para contratos de telecomunicaciones ni permiten afirmar reducción de churn o ROI en telecom;
- la regla comercial original (más de dos tickets técnicos al mes o una factura vencida por más de 30 días) no está en la OBT y no se puede evaluar con estos datos;
- la OBT no contiene timestamps de eventos. Los lotes semanales de la simulación se construyen sintéticamente y sirven para ejercitar API, logging, monitoreo, alertas y reentrenamiento; no prueban deriva temporal en producción;
- la etiqueta del dataset no define explícitamente un horizonte futuro de 30–60 días. Los resultados técnicos no se interpretan como evidencia de ese horizonte;
- el holdout de PSet 3 ya se utilizó para seleccionar el champion. PSet 4 no lo vuelve a presentar como una evaluación independiente.

Las decisiones, brechas entre PSets y limitaciones están registradas en [`docs/cambios.md`](docs/cambios.md). No se inventan tickets, timestamps, etiquetas, beneficios económicos ni resultados de intervención.

## Estructura integrada

```text
proyecto_final/
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── data.py
├── dbt/
├── kestra/
├── spark/
├── src/
├── notebooks/
├── api/                 # servicio de inferencia del PSet 4
├── dashboard/           # tablero de estado, lotes y alertas
├── monitoreo/           # PSI, pruebas estadísticas y alertas
├── tests/
├── .github/workflows/   # CI y publicación de imagen
└── docs/
    ├── cambios.md
    ├── etica_operativa.md
    ├── predicciones.csv
    ├── monitoreo.csv
    ├── alertas.csv
    ├── reentrenamientos_operativos.csv
    ├── historial/
    └── corridas.csv
```

La ingesta y el modelado conservan el pipeline del PSet 2; las particiones, el pipeline sklearn y la selección champion/challenger conservan el código del PSet 3. El PSet 4 incorpora la API versionada por alias MLflow, procesamiento batch,
persistencia y monitoreo en Snowflake, una compuerta de reentrenamiento y un
tablero Streamlit.

## Configuración local

Requisitos: Docker Desktop, Python 3.12, acceso a Snowflake y acceso a los archivos del dataset de la competencia. Los CSV locales se excluyen de Git: el dataset se obtiene con `data.py`/KaggleHub y sus credenciales se configuran localmente si Kaggle las solicita.

```bash
cp .env.example .env
```

Completa `.env` con valores locales para `POSTGRES_PASSWORD`, `SNOWFLAKE_URL`, `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USERNAME`, `SNOWFLAKE_PASSWORD`, `SNOWFLAKE_DATABASE` y `SNOWFLAKE_WAREHOUSE`. No subas `.env`, tokens ni contraseñas. El `.env.example` contiene únicamente valores de muestra. Para Kestra Open Source, define también `SECRET_SNOWFLAKE_PASSWORD` con el valor Base64 de `SNOWFLAKE_PASSWORD`; `secret()` lo decodifica al iniciar Kestra. Base64 es codificación, no cifrado: mantén `.env` local y excluido de Git. Confirma que `API_IMAGE` refiere a la misma imagen construida por Compose.

Para instalar el entorno de Python:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Reproducción del flujo

1. Obtén los CSV con `python data.py`; configura las variables Snowflake y
   `SECRET_SNOWFLAKE_PASSWORD` en `.env`, y carga los archivos con
   `kestra/flows/ingest_churn.yml`.
2. Ejecuta `dbt build` desde `dbt/` con el perfil local configurado.
3. Construye la OBT validada con `docker compose --profile spark run --rm spark`.
4. Los notebooks existentes de PSet 3 conservan su numeración histórica; deben
   ejecutarse desde Restart y Run All antes de la entrega. Sus resultados
   exportados se resumen en `docs/corridas.csv`.
5. Inicia los servicios: `docker compose up --build -d`.
   MLflow de PSet 4 se publica en `http://localhost:5001` para no chocar con
   el servidor del PSet 3 en el puerto 5000; dentro de Compose sigue usando 5000.
   Kestra se publica en `http://localhost:8081` para coexistir con Kestra de
   PSet 2 en el puerto 8080. Las variables disponibles en expresiones Kestra
   llevan el prefijo `ENV_` en Compose y se referencian en minúsculas, por
   ejemplo `ENV_SNOWFLAKE_ACCOUNT` como `envs.snowflake_account`. Los tasks de
   producción y exportación usan el runner Docker como `0:0` para escribir en
   el `workingDir` temporal montado por Kestra.
6. Importa los modelos históricos en el MLflow local:
   `docker compose --profile bootstrap run --rm registry-bootstrap`. Comprueba
   `/salud`, `/modelo` y `/docs` en `http://localhost:8000`; un modelo no
   disponible se reporta explícitamente como fallback degradado. Para inferir
   una observación, `POST /predecir` recibe `{"features": {...}}` conforme al
   esquema OpenAPI. Para cambiar el modelo, mueve el alias `champion` en MLflow
   y consulta `/modelo`: la API carga la versión apuntada sin reiniciarse. Si la
   nueva versión falla al cargar, informa estado degradado y sigue usando la
   última versión válida, con su versión y error visibles.
   Para suspender las decisiones del modelo y derivar todos los casos a revisión
   humana, configura `PREDICTION_MODE=manual_review_all` en `.env` y recrea solo
   la API con `docker compose up -d --force-recreate api`. El modo responde con
   `decision=revision_humana`, `used_fallback=true` y el prior global, que no es
   un score individual. Para volver al modelo, restablece `PREDICTION_MODE=model`.
7. Importa y ejecuta `kestra/flows/produccion_lotes.yml` desde `http://localhost:8081`.
   El flujo toma la OBT bancaria, la divide en ocho cohortes sintéticas por
   orden de `observation_id`, llama `POST /predecir/lote`, guarda resultados en
   Snowflake y adjunta CSV sin features ni identificadores personales.
   `HASH_AGG(*)` identifica el contenido observado de la OBT; no representa un
   snapshot transaccional inmutable de Snowflake. Si la ejecución falla después
   de guardar resultados en Snowflake, verifica sus `BATCH_ID` antes de
   reintentar para evitar duplicados.
8. Revisa las tablas `MONITORING.PREDICCIONES`, `MONITOREO` y `ALERTAS`, además
   del tablero en `http://localhost:8501`. La última cohorte modifica balance y
   geografía de forma sintética. Las fechas semanales son etiquetas de demo, no
   fechas de eventos.
9. Ejecuta `kestra/flows/monitoreo.yml` para exportar CSV de predicciones,
   monitoreo, alertas y estado operativo de reentrenamientos. El CSV de
   predicciones omite el ID de observación y las variables de entrada. El
   estado operativo se nombra `reentrenamientos_operativos.csv` para no
   sobrescribir el CSV de corridas de MLflow.
10. Ejecuta `kestra/flows/reentrenamiento.yml` para un `trigger_batch_id`
   registrado. Sin outcomes etiquetados el gate bloquea el entrenamiento; aun
   con etiquetas, requiere aprobación humana y evaluación independiente antes
   de promover o revertir un modelo.
11. La exportación exitosa de Kestra `4eOcHGoRTiqI2Z1WceO0Af` se conserva en
    `docs/`: `predicciones.csv`, `monitoreo.csv`, `alertas.csv` y
   `reentrenamientos_operativos.csv`. No repitas producción: los resultados
   ya están persistidos en Snowflake. Completa los entregables con outcomes,
   acciones y costos verificados; no se crean etiquetas ni resultados
   económicos.

## Criterios 8–10 de la consigna

- **Ética operativa y retirada:** el diagnóstico por subgrupo, costos cualitativos
  de error, supervisión y condiciones de respaldo están en
  [`docs/etica_operativa.md`](docs/etica_operativa.md). Los cortes de desempeño
  usan el holdout ya consumido por PSet 3 y no son evidencia independiente.
- **Resultado de negocio:** pendiente de outcomes etiquetados, acciones de
  intervención y supuestos de costos. El test de competencia no tiene etiquetas;
  por eso no se reporta ROI, churn evitado ni beneficio como si fueran medidos.
- **Integración end-to-end:** el orden reproducible y los límites entre PSet 1
  y los datos bancarios están descritos arriba y en `docs/cambios.md`. La
  trazabilidad de las predicciones conserva versión/run_id y metadatos de OBT;
  `HASH_AGG(*)` identifica el contenido observado, no un snapshot transaccional
  inmutable.

## Estado parcial frente a la consigna

- Para el commit `d101d5d`, CI pasó las pruebas y la construcción de la imagen;
  CD publicó la imagen en GHCR. Los resultados de ejecuciones posteriores se
  deben revisar en [GitHub Actions](https://github.com/eliasdavid98/proyecto_final/actions).
- El flujo de reentrenamiento registra el gate por falta de outcomes y exige
  aprobación; no ejecuta todavía el entrenamiento, evaluación y promoción
  challenger descritos en 7.1–7.4.
- `docs/corridas.csv` contiene 103 corridas de PSet 3, sin corridas etiquetadas
  como reentrenamiento. `docs/reentrenamientos.csv` queda reservado para las
  corridas MLflow; el estado exportado desde Snowflake está en
  `docs/reentrenamientos_operativos.csv`.
- Los notebooks no tienen salidas guardadas y su numeración difiere de la
  estructura de referencia. Los diagramas, memo técnico, presentación
  ejecutiva y video de PSet 4 aún no están incluidos.
- El resultado de negocio del paso 9 sigue pendiente de outcomes, acciones,
  comparador y costos. No se reportan ROI ni churn evitado.

## Pruebas

```bash
source .venv/bin/activate
pytest
```

Los flujos con Snowflake y la API requieren credenciales válidas y el stack local
activo; sus resultados operativos no se simulan como si fueran corridas reales.
