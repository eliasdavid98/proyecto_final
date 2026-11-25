# PSet 3: experimentación y selección del modelo

Este repositorio continúa el proyecto de churn del PSet 2. El primer paso analiza la One Big Table (OBT) publicada en Snowflake por el pipeline anterior.

## Estructura

```text
PSet3/
├── .env.example
├── README.md
├── requirements.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_preparacion.ipynb
│   ├── 05_baseline.ipynb
│   ├── 06_modelos_candidatos.ipynb
│   ├── 07_ajuste_hiperparametros.ipynb
│   └── 08_seleccion_modelo.ipynb
├── src/
│   ├── ajuste.py
│   ├── seleccion.py
│   ├── datos.py
│   ├── candidatos.py
│   ├── experimentos.py
│   ├── pipeline.py
│   ├── baseline.py
│   ├── features.py
│   └── preparacion.py
├── scripts/
│   ├── start_mlflow.sh
│   ├── register_challenger.py
│   └── export_runs.py
├── docs/
│   ├── PSet3_memo_pogo_torres.pdf
│   └── corridas.csv
└── tests/
    ├── test_export_runs.py
    ├── test_ajuste.py
    ├── test_features.py
    └── test_preparacion.py
```

## Paso 1: exploración de datos

1. Asegúrate de haber ejecutado el pipeline del PSet 2: la tabla `GOLD.OBT_CUSTOMER_CHURN` debe existir en Snowflake.
2. Crea y activa un entorno Python 3.12, instala dependencias y configura las credenciales:

   ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   python -m ensurepip --upgrade
   python -m pip install -r requirements.txt
   cp .env.example .env
   ```

3. Completa `.env` con el identificador de cuenta, usuario, contraseña, base de datos, warehouse y esquema Gold. No subas `.env` al repositorio.
4. Abre `notebooks/01_eda.ipynb` en VS Code o Jupyter y ejecuta todas las celdas desde el inicio.

El notebook inspecciona el tamaño, tipos, valores faltantes, estadística descriptiva, distribución del target, valores atípicos por IQR y Z-score, correlaciones de Spearman, tasas de churn por categorías y hallazgos que afectan el diseño del experimento.

La OBT contiene filas `train` etiquetadas y `test` sin etiqueta. El análisis del target, sus asociaciones y los outliers para modelado se calcula solo con `source_split == "train"`; la tabla de test se usa únicamente para revisar estructura y completitud. El hold-out interno para la evaluación final se definirá en el paso de preparación, sin usar el CSV de competencia sin etiquetas como conjunto de evaluación.

Los hallazgos generados por el notebook son evidencia exploratoria; no implican eliminar filas ni aplicar transformaciones antes de justificarlas en los pasos de features y pipeline.

## Paso 2: diseño de features

Después de ejecutar `01_eda.ipynb`, abre y ejecuta `notebooks/02_features.ipynb`. El notebook recalcula la evidencia en train y evalúa dos variables derivadas implementadas en `src/features.py`. La validación con la OBT real de Snowflake devolvió 25.000 filas: 15.000 de train etiquetado y 10.000 de test sin etiqueta; no hubo valores faltantes en `age`, `balance` ni `exited` de train. Las columnas transformadas de train y test quedaron alineadas.

- `age_40_plus`: distingue edades menores de 40 de edades 40–100. En la OBT real, el churn fue 44,79% para el grupo 40+ (5.318 filas) frente a 7,37% para menores de 40 (9.682 filas).
- `balance_status`: distingue balance cero de balance positivo. El 64,76% del train tenía balance cero (9.714 filas); las tasas de churn observadas fueron 16,18% para saldo cero y 28,83% para saldo positivo (5.286 filas).

Ambas variables se derivan de `age` y `balance`, disponibles también para las filas test antes de predecir. La transformación no recibe ni consulta `Exited`, no estima tasas del target y no calcula estadísticas de la muestra. La banda de edad precomputada se elimina para evitar duplicar esa representación; la codificación one-hot de las variables categóricas se hará en el `ColumnTransformer` ajustado dentro de cada fold en el paso de preparación. La asociación de EDA solo justifica probar las features; la CV determinará si aportan al modelo.

## Paso 3: preparación y esquema de validación

Ejecuta `notebooks/03_preparacion.ipynb` después del notebook de features. `src/preparacion.py` divide únicamente el train etiquetado en desarrollo (80%) y holdout estratificado (20%), usando semilla 42. El test de competencia sin etiquetas permanece aparte y no participa en selección ni evaluación.

El desarrollo utiliza validación cruzada estratificada de 3 folds, también con semilla 42. `src/pipeline.py` mantiene `ChurnFeatureEngineer`, imputación, escalado y codificación one-hot dentro de un único `Pipeline`; por lo tanto, cada fold ajusta esas transformaciones solamente con su porción de entrenamiento. El holdout se reserva para una evaluación final después de seleccionar el modelo. Los números de Snowflake (`Decimal`) se convierten a tipos numéricos antes de la selección de columnas.

## Paso 4: servidor de tracking MLflow

Instala las dependencias desde el entorno virtual y, desde una terminal en la raíz de PSet3, inicia el servidor local:

```bash
source .venv/bin/activate
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
MLFLOW_PORT=5001 bash scripts/start_mlflow.sh
```

En esta máquina el puerto 5000 está ocupado, por eso se usa 5001. Abre `http://127.0.0.1:5001` para consultar la interfaz y configura `MLFLOW_TRACKING_URI=http://127.0.0.1:5001` al ejecutar los notebooks/scripts. El backend persistente usa `mlflow.db` (SQLite) y guarda artefactos en `mlartifacts/`; ambos se excluyen de Git. Mantén esta terminal abierta mientras ejecutas los experimentos. El servidor escucha solo en la interfaz local.

## Paso 5: línea base

Con el servidor activo, ejecuta `notebooks/05_baseline.ipynb`. `src/baseline.py` compara `DummyClassifier(strategy="prior")` con regresión logística balanceada, usando el mismo pipeline y los mismos tres folds sobre el split de desarrollo. Registra en el experimento `pset3-churn-baselines` las medias y desviaciones estándar de ROC-AUC, average precision, F1, precision, recall y balanced accuracy. La evaluación no ajusta modelos con el holdout ni con el test de competencia.

Validación inicial con la OBT real: regresión logística obtuvo ROC-AUC 0,8806, average precision 0,7172, F1 0,6362, precision 0,5327, recall 0,7897 y balanced accuracy 0,8047 como medias de CV. `DummyClassifier` obtuvo ROC-AUC 0,5000 y average precision 0,2064. Son referencias de desarrollo, no resultados del holdout ni una selección del modelo final.

## Paso 6: modelos candidatos

Ejecuta `notebooks/06_modelos_candidatos.ipynb` con el servidor MLflow activo. `src/candidatos.py` compara cinco familias sobre los mismos folds: árbol de decisión, random forest, gradient boosting, SVC con kernel RBF y K-nearest neighbors. Las configuraciones son puntos de partida, no hiperparámetros finales. `src/experimentos.py` registra los parámetros y las medias/desviaciones de las mismas seis métricas en `pset3-churn-candidates`. Selecciona los tres candidatos con mejor evidencia de CV de desarrollo para el ajuste del siguiente paso; no uses el holdout para ordenarlos.

En la primera ejecución con la OBT real, los resultados por ROC-AUC / average precision fueron: gradient boosting 0,9303 / 0,8170; random forest 0,9235 / 0,7911; SVC RBF 0,9184 / 0,7759; KNN 0,9045 / 0,7549; árbol de decisión 0,9004 / 0,7651. ROC-AUC y average precision coinciden en los tres primeros: gradient boosting, random forest y SVC RBF, candidatos para el ajuste del siguiente paso.

## Paso 7: ajuste de hiperparámetros

Con el servidor MLflow activo, ejecuta `notebooks/07_ajuste_hiperparametros.ipynb`. `src/ajuste.py` ajusta los tres mejores candidatos del paso anterior con exactamente 30 configuraciones por modelo y los mismos 3 folds sobre desarrollo. Grid search selecciona por ROC-AUC; cada búsqueda, cada configuración y sus métricas se registran en `pset3-churn-fine-tuning`. El holdout no se usa en ninguna búsqueda; se reserva para comparar los finalistas en el paso de evaluación final.

Con la OBT real, las mejores medias de CV fueron: gradient boosting ROC-AUC 0,9317 (average precision 0,8198; `learning_rate=0.07`, `max_depth=4`, `n_estimators=120`); random forest ROC-AUC 0,9238 (average precision 0,7921; `max_depth=15`, `min_samples_leaf=5`, `n_estimators=200`); SVC RBF ROC-AUC 0,9214 (average precision 0,7817; `C=10`, `gamma=0.01`, `class_weight=balanced`). Gradient boosting y random forest son los dos finalistas por ROC-AUC para el holdout; esta decisión se tomó sin consultarlo.

## Paso 8: selección final y registro

Ejecuta `notebooks/08_seleccion_modelo.ipynb` una sola vez, con MLflow activo. `src/seleccion.py` recupera del tracking los dos finalistas del paso 7, los ajusta sobre desarrollo y los compara en el holdout con ROC-AUC, average precision, F1, precision, recall y balanced accuracy. Selecciona por ROC-AUC de holdout y registra únicamente el ganador, con firma, ejemplo de entrada y código del transformador empaquetado.

El modelo queda en el Model Registry como `pset3-churn-classifier` con el alias `champion`; se puede cargar mediante `models:/pset3-churn-classifier@champion`. La evaluación registra `holdout_used=true` y el notebook no debe volver a ejecutarse una vez consumido el holdout.

Resultado de la ejecución final: ganó gradient boosting y quedó registrado como versión 1 con el alias `champion`. Sus métricas de holdout fueron ROC-AUC 0,9323, average precision 0,8301, F1 0,7258, precision 0,7868, recall 0,6737 y balanced accuracy 0,8131. El modelo registrado se cargó desde el alias y se probó con tres filas de desarrollo.

En el primer intento, MLflow rechazó tipos de sklearn al empaquetar el modelo; la corrida guardó la selección y las métricas de gradient boosting, pero el error interrumpió el registro antes de guardar las métricas del random forest. Se añadió una lista explícita de tipos confiables para el serializador seguro y se completó el registro sin volver a puntuar el holdout. Por ello, no se reportan aquí métricas de holdout del random forest; el experimento y el registro en MLflow conservan la evidencia disponible.

El segundo finalista, Random Forest, quedó registrado como versión 2 con el alias `challenger`; se ajustó solo con desarrollo y no se evaluó otra vez en holdout. Sus métricas de CV fueron ROC-AUC 0,9238 y average precision 0,7921. El archivo [docs/corridas.csv](./docs/corridas.csv) contiene 103 corridas de los cuatro experimentos PSet 3. Para volver a registrar el challenger o regenerar el CSV:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
  .venv/bin/python scripts/register_challenger.py --env-file .env
MLFLOW_TRACKING_URI=http://127.0.0.1:5001 \
  .venv/bin/python scripts/export_runs.py
```

Las rutas de artefactos exportadas en el CSV son relativas a la carpeta del proyecto para evitar registrar rutas locales de una máquina.

El alias `challenger` se carga como `models:/pset3-churn-classifier@challenger`; champion se carga como `models:/pset3-churn-classifier@champion`. El CSV incluye `experiment_name` para identificar de cuál experimento proviene cada corrida.
