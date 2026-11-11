#!/usr/bin/env bash
# Open (or re-open) the SSH tunnel to the MLflow server on the VM:
#   Cloud Shell 127.0.0.1:5000  ==SSH==>  VM 127.0.0.1:5000 (mlflow server)
# Safe to run any time: does nothing if the tunnel already answers.
set -euo pipefail
VM=instance-week-5-graded
ZONE=us-central1-a
PORT=5000

if curl -s -m 3 "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "Tunnel already up: http://127.0.0.1:${PORT}"
  exit 0
fi

# Remove a stale tunnel that still holds the port but no longer works
pkill -f "127.0.0.1:${PORT}:127.0.0.1:${PORT}" || true

gcloud compute ssh "$VM" --zone "$ZONE" -- \
  -f -N \
  -L "127.0.0.1:${PORT}:127.0.0.1:${PORT}" \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes

sleep 2
curl -s -m 5 "http://127.0.0.1:${PORT}/health" && echo "  <- MLflow reachable through the tunnel"
