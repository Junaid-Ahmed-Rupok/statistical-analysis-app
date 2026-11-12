#!/usr/bin/env bash
# One-command launcher for macOS / Linux. Creates a local .venv on first run.
set -e
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.13 || command -v python3 || true)
[ -z "$PY" ] && { echo "Python 3.11+ is required: https://python.org/downloads"; exit 1; }
"$PY" -c 'import sys; sys.exit(sys.version_info < (3,11))' || { echo "Python 3.11+ is required (found $($PY -V))"; exit 1; }
[ -d .venv ] || "$PY" -m venv .venv
. .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
exec streamlit run app.py
