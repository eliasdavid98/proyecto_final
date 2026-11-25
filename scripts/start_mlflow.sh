#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/mlartifacts"

exec "$PROJECT_ROOT/.venv/bin/mlflow" server \
  --host 127.0.0.1 \
  --port "$MLFLOW_PORT" \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root "$PROJECT_ROOT/mlartifacts"
