#!/bin/bash
# Usage: ./run_py_script.sh path/to/script.py [args...]

SCRIPT="$1"
shift

if [[ -z "$SCRIPT" ]]; then
    echo "Usage: $0 path/to/script.py [args...]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$SCRIPT")" && pwd)"
SCRIPT_BASENAME="$(basename "$SCRIPT")"
LOG_FILE="${SCRIPT_DIR}/${SCRIPT_BASENAME}.log"

PYTHON_BIN=""

# Prefer an active virtual environment if present.
if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    echo "Warning: no active virtual environment detected. Activate one before running this script." >&2
    exit 1
fi

nohup "$PYTHON_BIN" "$SCRIPT" "$@" > "$LOG_FILE" 2>&1 &
PID=$!

echo "Started $SCRIPT with PID $PID"
echo "Logging to $LOG_FILE"
