# Sistema Predictivo de Churn Bancario — Proyecto Final (PSets #1 a #4)

Repositorio end-to-end de ciencia de datos y MLOps para predicción y retención proactiva de churn bancario. Integra ingesta orquestada con **Kestra**, almacenamiento y modelado medallón/dimensional con **Snowflake** y **dbt**, construcción de OBT con **Apache Spark**, experimentación y Model Registry con **MLflow**, servicio de inferencia en **FastAPI**, monitoreo de deriva y negocio en **Streamlit**, y CI/CD con **GitHub Actions + GHCR**.

---

## 1. Requisitos previos y configuración de entorno

1. **Requisitos:** Docker Desktop (con Docker Compose v2), Python 3.12 y una cuenta activa de Snowflake con base de datos y warehouse creados.
2. **Variables de entorno:** Copia `.env.example` a `.env` y completa tus credenciales (nunca subas `.env` a Git):
   ```bash
   cp .env.example .env