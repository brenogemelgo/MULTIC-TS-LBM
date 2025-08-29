#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

GPU_ARCH=86
OS_TYPE=$(uname -s)

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

    echo -e "${YELLOW}Preparing directory ${CYAN}${SIMULATION_DIR}${RESET}"
    mkdir -p "${SIMULATION_DIR}"

    echo -e "${YELLOW}Cleaning directory ${CYAN}${SIMULATION_DIR}${RESET}"
    find "${SIMULATION_DIR}" -mindepth 1 ! -name ".gitkeep" -exec rm -rf {} +

    # sanity check
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
    bash "${BASE_DIR}/compile.sh" "${FLOW_CASE}" "${VELOCITY_SET}" "${ID}" \
        || { echo -e "${RED}Error executing compile.sh${RESET}"; exit 1; }

    local EXECUTABLE="${MODEL_DIR}/${ID}sim_${FLOW_CASE}_${VELOCITY_SET}_sm${GPU_ARCH}"
    if [ ! -f "$EXECUTABLE" ]; then
        echo -e "${RED}Error: Executable not found in ${CYAN}${EXECUTABLE}${RESET}"
        exit 1
    fi

    echo -e "${BLUE}Running: ${CYAN}${EXECUTABLE} ${FLOW_CASE} ${VELOCITY_SET} ${ID}${RESET}"
    if [[ "$OS_TYPE" == "Linux" ]]; then
        "${EXECUTABLE}" "${FLOW_CASE}" "${VELOCITY_SET}" "${ID}" 1 || {
            echo -e "${RED}Error running the simulator${RESET}"
            exit 1
        }
    else
        "${EXECUTABLE}.exe" "${FLOW_CASE}" "${VELOCITY_SET}" "${ID}" 1 || {
            echo -e "${RED}Error running the simulator (Windows)${RESET}"
            exit 1
        }
    fi

    echo -e "${YELLOW}Entering ${CYAN}${BASE_DIR}/post${RESET}"
    cd "${BASE_DIR}/post" || { echo -e "${RED}Error: Directory ${CYAN}${BASE_DIR}/post${RED} not found!${RESET}"; exit 1; }

    echo -e "${BLUE}Executing: ${CYAN}./post.sh ${FLOW_CASE} ${VELOCITY_SET} ${ID}${RESET}"
    ./post.sh "${FLOW_CASE}" "${VELOCITY_SET}" "${ID}" \
        || { echo -e "${RED}Error executing post.sh${RESET}"; exit 1; }

    echo -e "${GREEN}Process completed successfully!${RESET}"
}

runPipeline "$1" "$2" "$3"
