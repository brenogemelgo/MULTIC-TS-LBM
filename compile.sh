#!/bin/bash
set -euo pipefail

CC=86
FLOW_CASE=${1:-}
VELOCITY_SET=${2:-}
ID=${3:-}

if [ -z "$FLOW_CASE" ] || [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
    echo "Usage: ./compile.sh <FLOW_CASE> <VELOCITY_SET> <ID>"
    exit 1
fi

if [ "$VELOCITY_SET" != "D3Q19" ] && [ "$VELOCITY_SET" != "D3Q27" ]; then
    echo "Invalid VELOCITY_SET. Use 'D3Q19' or 'D3Q27'."
    exit 1
fi

#if [ "$VELOCITY_SET" = "D3Q27" ]; then
#    MAXRREG=72
#else
    MAXRREG=68
#fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

if [ -d "${SCRIPT_DIR}/src" ]; then
    BASE_DIR="${SCRIPT_DIR}"
elif [ -d "${SCRIPT_DIR}/../src" ]; then
    BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
    echo "Error: could not locate project root (missing 'src/' next to or above compile.sh)."
    echo "Checked: ${SCRIPT_DIR}/src and ${SCRIPT_DIR}/../src"
    exit 1
fi

SRC_DIR="${BASE_DIR}/src"
OUTPUT_DIR="${BASE_DIR}/bin/${FLOW_CASE}/${VELOCITY_SET}"
EXECUTABLE="${OUTPUT_DIR}/${ID}sim_${FLOW_CASE}_${VELOCITY_SET}_sm${CC}"

mkdir -p "${OUTPUT_DIR}"

echo "Project root detected: ${BASE_DIR}"
echo "Compiling to ${EXECUTABLE}..."

nvcc -O3 --restrict \
     -gencode arch=compute_${CC},code=sm_${CC} -rdc=true --ptxas-options=-v \
     --use_fast_math --fmad=true \
     -I"${SRC_DIR}" \
     "${SRC_DIR}/main.cu" \
     -D${VELOCITY_SET} -D${FLOW_CASE} \
     -o "${EXECUTABLE}"

echo "Compilation completed successfully: ${EXECUTABLE}"
