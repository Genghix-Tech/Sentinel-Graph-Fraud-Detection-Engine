#!/bin/sh
set -e
# Seed the demo world and train model v1 on first start (skipped if a model already exists).
if [ ! -f "${ARTIFACT_DIR:-/app/artifacts}/v1/meta.json" ]; then
  python -m app.cli seed
fi
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
