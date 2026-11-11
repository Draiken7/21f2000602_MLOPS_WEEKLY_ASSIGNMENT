#!/usr/bin/env bash
# Runs ON THE VM (instance-week-5-graded), inside `screen -S mlflow_server`.
# Copy kept in the repo for reproducibility - the VM itself is not in Git.
#
# MLflow tracking server + Model Registry for the IRIS pipeline.
#   Backend store  : SQLite on the VM's disk (runs, params, metrics, registry)
#   Artifact store : GCS bucket, proxied through this server (model files)
#   Listens on     : 127.0.0.1:5000 only - reached through an SSH tunnel
#   Browser access : Cloud Shell Web Preview (*.cloudshell.dev)
set -euo pipefail
source "$HOME/mlflow-venv/bin/activate"
mkdir -p "$HOME/mlflow"
exec mlflow server \
  --backend-store-uri "sqlite:///$HOME/mlflow/mlflow.db" \
  --artifacts-destination "gs://21f2000602-dvc-week2/mlflow-artifacts" \
  --host 127.0.0.1 \
  --port 5000 \
  --allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,*.cloudshell.dev" \
  --cors-allowed-origins "https://*.cloudshell.dev"
