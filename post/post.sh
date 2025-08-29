#!/bin/bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
    echo "Error: Usage: ./post.sh <flow_case> <velocity_set> <id>"
    echo "Example: ./post.sh JET D3Q19 000"
    exit 1
fi

FLOW_CASE=$1
VELOCITY_SET=$2
SIM_ID=$3

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PYTHON_CMD="python3"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    PYTHON_CMD="python"
else
    echo "Operating system not recognized. Trying python3 by default."
    PYTHON_CMD="python3"
fi

$PYTHON_CMD processSteps.py "$FLOW_CASE" "$VELOCITY_SET" "$SIM_ID"
