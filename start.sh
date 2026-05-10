#!/usr/bin/env bash
# =============================================================================
# start.sh — Activate venv and start ComfyUI
# =============================================================================
#
# Run from your ComfyUI installation directory, e.g.:
#   cd ~/comfy-local && bash ~/git/comfyui-cloud-parity/start.sh
#
# Tested with Windows Git Bash. Linux/macOS: if activation fails, the script
# tries both venv/Scripts/activate (Windows) and venv/bin/activate (Unix).
# =============================================================================

set -euo pipefail

if [[ -f "venv/Scripts/activate" ]]; then
    source venv/Scripts/activate
elif [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
else
    echo "ERROR: No venv found in $(pwd)."
    echo "  Run setup.sh first to create the ComfyUI environment."
    exit 1
fi

echo "Starting ComfyUI on http://localhost:8188 ... (Ctrl+C to stop)"
python main.py --listen
