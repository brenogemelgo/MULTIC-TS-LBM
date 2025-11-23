#!/bin/bash
set -euo pipefail

./pipeline.sh JET D3Q19 re5000we100 100

./pipeline.sh JET D3Q19 re5000we500 500

./pipeline.sh JET D3Q19 re5000we2000 2000