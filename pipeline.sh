#!/bin/bash
set -euo pipefail

GPU_INDEX=0

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

OS_TYPE=$(uname -s)

detect_sm() {
    local cc=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU_INDEX}" 2>/dev/null | head -n1 | tr -d '.')
    fi
    echo "${cc}"
}
GPU_ARCH="$(detect_sm)"; GPU_ARCH="${GPU_ARCH:-86}"

runPipeline() {
    local FLOW_CASE=${1:-}
    local VELOCITY_SET=${2:-}
    local ID=${3:-}

    if [ -z "$FLOW_CASE" ] || [ -z "$VELOCITY_SET" ] || [ -z "$ID" ]; then
        echo -e "${RED}Error: Insufficient arguments.${RESET}"
        echo -e "${YELLOW}Usage: ./pipeline.sh <flow_case> <velocity_set> <id>${RESET}"
        echo -e "${YELLOW}Example: ./pipeline.sh JET D3Q19 000${RESET}"
        exit 1
    fi

    local BASE_DIR
    BASE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

    local MODEL_DIR="${BASE_DIR}/bin/${FLOW_CASE}/${VELOCITY_SET}"
    local SIMULATION_DIR="${MODEL_DIR}/${ID}"

    echo -e "${BLUE}GPU index: ${CYAN}${GPU_INDEX}${BLUE} → SM: ${CYAN}${GPU_ARCH}${RESET}"

    echo -e "${YELLOW}Preparing directory ${CYAN}${SIMULATION_DIR}${RESET}"
    mkdir -p "${SIMULATION_DIR}"

    echo -e "${YELLOW}Cleaning directory ${CYAN}${SIMULATION_DIR}${RESET}"
    find "${SIMULATION_DIR}" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

    local FILES
    FILES=$(ls -A "${SIMULATION_DIR}" | grep -v '^\.gitkeep$' || true)
    if [ -n "$FILES" ]; then
        echo -e "${RED}Error: The directory ${CYAN}${SIMULATION_DIR}${RED} still contains files!${RESET}"
        exit 1
    else
        echo -e "${GREEN}Directory cleaned successfully.${RESET}"
    fi

    echo -e "${YELLOW}Entering ${CYAN}${BASE_DIR}${RESET}"
    cd "${BASE_DIR}" || { echo -e "${RED}Error: Directory ${CYAN}${BASE_DIR}${RED} not found!${RESET}"; exit 1; }

    echo -e "${BLUE}Executing: ${CYAN}${BASE_DIR}/compile.sh ${FLOW_CASE} ${VELOCITY_SET} ${ID}${RESET}"

    GPU_INDEX="${GPU_INDEX}" GPU_ARCH="${GPU_ARCH}" bash "${BASE_DIR}/compile.sh" "${FLOW_CASE}" "${VELOCITY_SET}" "${ID}" \
        || { echo -e "${RED}Error executing compile.sh${RESET}"; exit 1; }

    local EXECUTABLE_BASENAME="${ID}sim_${FLOW_CASE}_${VELOCITY_SET}_sm${GPU_ARCH}"
    local EXECUTABLE="${MODEL_DIR}/${EXECUTABLE_BASENAME}"

    if [ ! -f "$EXECUTABLE" ] && [ -f "${EXECUTABLE}.exe" ]; then
        EXECUTABLE="${EXECUTABLE}.exe"
    fi
    if [ ! -f "$EXECUTABLE" ]; then
        echo -e "${RED}Error: Executable not found: ${CYAN}${MODEL_DIR}/${EXECUTABLE_BASENAME}[.exe]${RESET}"
        echo -e "${YELLOW}Tip:${RESET} confirm that compile.sh uses _sm${GPU_ARCH}."
        exit 1
    fi

    export GPU_INDEX

    echo -e "${BLUE}Running: ${CYAN}${EXECUTABLE} ${FLOW_CASE} ${VELOCITY_SET} ${ID}${RESET}"
    "${EXECUTABLE}" "${FLOW_CASE}" "${VELOCITY_SET}" "${ID}" 1 || {
        echo -e "${RED}Error running the simulator${RESET}"
        exit 1
    }

    echo -e "${YELLOW}Entering ${CYAN}${BASE_DIR}/post${RESET}"
    cd "${BASE_DIR}/post" || { echo -e "${RED}Error: Directory ${CYAN}${BASE_DIR}/post${RED} not found!${RESET}"; exit 1; }

    echo -e "${BLUE}Executing: ${CYAN}./post.sh ${FLOW_CASE} ${VELOCITY_SET} ${ID}${RESET}"
    ./post.sh "${FLOW_CASE}" "${VELOCITY_SET}" "${ID}" \
        || { echo -e "${RED}Error executing post.sh${RESET}"; exit 1; }

    echo -e "${GREEN}Process completed successfully!${RESET}"
}

runPipeline "$1" "$2" "$3"
