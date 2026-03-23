NVCC ?= nvcc
TARGET ?= phaseFieldLBM
BUILD_DIR ?= build
PREFIX ?= $(HOME)/.local

CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | head -n1 | tr -d '.')
CUDA_ARCH := $(if $(CUDA_ARCH),$(CUDA_ARCH),86)

SRC := src/main.cu
TARGET_PATH := $(BUILD_DIR)/$(TARGET)
INSTALL_BIN_DIR := $(DESTDIR)$(PREFIX)/bin

NVCC_FLAGS := -O3 --restrict \
	-gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) \
	-gencode arch=compute_$(CUDA_ARCH),code=lto_$(CUDA_ARCH) \
	-rdc=true \
	--ptxas-options=-v \
	--extra-device-vectorization \
	--fmad=true \
	--extended-lambda \
	-std=c++20 \
	-Isrc

CPP_DEFS := \
	-DENABLE_FP16=1 \
	-DBENCHMARK=1 \
	-DTIME_AVERAGE=0 \
	-DREYNOLDS_MOMENTS=0 \
	-DVORTICITY_FIELDS=0 \
	-DPASSIVE_SCALAR=0

all: $(TARGET_PATH)

$(TARGET_PATH): $(SRC) $(shell find src -type f)
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $(CPP_DEFS) $(SRC) -o $(TARGET_PATH)

install: all
	install -d $(INSTALL_BIN_DIR)
	install -m 0755 $(TARGET_PATH) $(INSTALL_BIN_DIR)/$(TARGET)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all install clean
