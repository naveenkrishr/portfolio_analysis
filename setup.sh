#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Portfolio Analysis — Setup ==="

if [ ! -d "venv" ]; then
    echo "[1/2] Creating virtual environment..."
    python3 -m venv venv
else
    echo "[1/2] venv exists — skipping"
fi

echo "[2/2] Installing dependencies..."
./venv/bin/pip install --quiet --upgrade pip
./venv/bin/pip install --quiet -r requirements.txt

echo ""
echo "=== Setup complete ==="
echo "Run: ./venv/bin/python3 main.py"
