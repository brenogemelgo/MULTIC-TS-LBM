#!/bin/bash

CC=86
VELOCITY_SET=$1
ID=$2

if [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
    echo "Usage: ./compile.sh <VELOCITY_SET> <ID>"
    exit 1
fi

if [ "$VELOCITY_SET" != "D3Q19" ] && [ "$VELOCITY_SET" != "D3Q27" ]; then
    echo "Invalid VELOCITY_SET. Use 'D3Q19' or 'D3Q27'."
    exit 1
fi

# least register value without spills
if [ "$VELOCITY_SET" = "D3Q27" ]; then
    MAXRREG=128
elif [ "$VELOCITY_SET" = "D3Q19" ]; then
    MAXRREG=68 
    # under this value there seems 
    # to be phantom spills that the 
    # compiler does not report
fi

BASE_DIR=$(dirname "$0")
SRC_DIR="${BASE_DIR}/src"
OUTPUT_DIR="${BASE_DIR}/bin/${VELOCITY_SET}"
EXECUTABLE="${OUTPUT_DIR}/${ID}sim_${VELOCITY_SET}_sm${CC}"

mkdir -p "${OUTPUT_DIR}"

echo "Compiling to ${EXECUTABLE}..."

nvcc -O3 --restrict \
     -gencode arch=compute_${CC},code=sm_${CC} \
     -rdc=true --ptxas-options=-v -use_fast_math \
     -I"${SRC_DIR}" \
     "${SRC_DIR}/device_setup.cu" \
     "${SRC_DIR}/main.cu" \
     -maxrregcount=${MAXRREG} -D${VELOCITY_SET} \
     -o "${EXECUTABLE}" \
     -std=c++17

if [ $? -eq 0 ]; then
    echo "Compilation completed successfully: ${OUTPUT_DIR}/${EXECUTABLE_NAME}"
else
    echo "Compilation error!"
    exit 1
fi
