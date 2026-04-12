#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
echo "=== black ==="
black --check src/pipeline/ src/training/ tests/ generate.py train.py
echo "=== pylint ==="
pylint src/pipeline/ src/training/ generate.py train.py
echo "=== pytest ==="
pytest tests/ -v
