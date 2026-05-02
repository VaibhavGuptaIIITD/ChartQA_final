#!/bin/bash
# ================================================================
# GPU Server Virtual Environment Setup
# Run this once on the shared GPU server to create an isolated
# environment under your own subdirectory.
# Usage: bash scripts/setup_venv.sh <YOUR_NAME>
# ================================================================

YOUR_NAME=${1:-"default_user"}
VENV_PATH="/NS/ssdecl/work/${YOUR_NAME}/chartqa_env"

echo "[INFO] Creating virtual environment at: ${VENV_PATH}"
python3 -m venv "${VENV_PATH}"

echo "[INFO] Activating virtual environment..."
source "${VENV_PATH}/bin/activate"

echo "[INFO] Confirming Python path (should point to venv):"
which python

echo "[INFO] Installing dependencies..."
pip install transformers accelerate sentencepiece Pillow requests \
    pandas matplotlib seaborn tqdm torch torchvision datasets

echo ""
echo "[DONE] Setup complete."
echo "To activate in future sessions, run:"
echo "  source ${VENV_PATH}/bin/activate"
echo "To deactivate:"
echo "  deactivate"
