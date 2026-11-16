#!/usr/bin/env bash
# Source this file: source scripts/activate.sh
_dad_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$_dad_root/.venv/Scripts/activate"
export PYTHONPATH="$_dad_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUTF8=1
export UV_CACHE_DIR="$_dad_root/.uv-cache"
export UV_PYTHON_INSTALL_DIR="$_dad_root/.python"
python -c 'import sys; print("Active Python:", sys.executable); assert sys.version_info[:2] == (3,12), "Python 3.12 required"'
unset _dad_root
