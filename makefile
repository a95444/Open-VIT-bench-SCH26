CC := g++
CFLAGS := -std=c++11 -O3
OMPFLAGS := -fopenmp

BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src

OMP_BIN_FOLDER := omp_bin
OMP_OBJ_FOLDER := omp_obj
OMP_SRC_FOLDER := omp_src

TEST_BIN_FOLDER := test_bin
TEST_OBJ_FOLDER := test_obj
TEST_SRC_FOLDER := test_src

# --- NVTX profiling build (Nsight Systems) ---------------------------------
# Same code + flags as the normal build, plus debug info and NVTX ranges
# (gated by -DUSE_NVTX in include/vit_nvtx.h). Objects go to separate folders
# so the benchmark artifacts in obj/ and omp_obj/ are never clobbered.
# CUDA_HOME must point at a CUDA toolkit that provides nvtx3/nvToolsExt.h.
CUDA_HOME ?= /usr/local/cuda
NVTX_INC := -I$(CUDA_HOME)/include
NVTX_LIB := -ldl
PROF_CFLAGS := $(CFLAGS) -g -fno-omit-frame-pointer -DUSE_NVTX $(NVTX_INC)

PROF_BIN_FOLDER := prof_bin
PROF_OBJ_FOLDER := prof_obj
OMP_PROF_OBJ_FOLDER := omp_prof_obj

# --- CUDA build (hand-written kernels; starts with Linear) ------------------
NVCC := nvcc
CUDA_ARCH ?= sm_80                 # Leonardo Booster = A100; override for other GPUs
NVCCFLAGS := -O3 -std=c++11 -arch=$(CUDA_ARCH)
CUDA_BIN_FOLDER := cuda_bin
CUDA_OBJ_FOLDER := cuda_obj
CUDA_SRC_FOLDER := cuda_src



all : vit

clean :
	rm -rf ./$(OBJ_FOLDER)/* ./$(BIN_FOLDER)/* ./$(OMP_OBJ_FOLDER)/* ./$(OMP_BIN_FOLDER)/* \
		   ./$(TEST_OBJ_FOLDER)/* ./$(TEST_BIN_FOLDER)/* \
		   ./$(PROF_OBJ_FOLDER)/* ./$(OMP_PROF_OBJ_FOLDER)/* ./$(PROF_BIN_FOLDER)/* \
		   ./$(CUDA_OBJ_FOLDER)/* ./$(CUDA_BIN_FOLDER)/* \
		   ./out_comparison/* ./logs/*

clean_everything :
	rm -rf ./$(OBJ_FOLDER)/* ./$(BIN_FOLDER)/* ./$(OMP_OBJ_FOLDER)/* ./$(OMP_BIN_FOLDER)/* \
		   ./$(TEST_OBJ_FOLDER)/* ./$(TEST_BIN_FOLDER)/* ./test_files/* \
		   ./$(PROF_OBJ_FOLDER)/* ./$(OMP_PROF_OBJ_FOLDER)/* ./$(PROF_BIN_FOLDER)/* \
		   ./$(CUDA_OBJ_FOLDER)/* ./$(CUDA_BIN_FOLDER)/* \
		   ./data/* ./models/* ./out/* ./measures/* \
		   ./out_comparison/* ./logs/*

vit : % : $(BIN_FOLDER)/%.exe



# OBJs
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(OBJ_FOLDER)/main.o \
\
: $(OBJ_FOLDER)/%.o : $(SRC_FOLDER)/%.cpp
	$(CC) -c $(CFLAGS) $^ -o $@

# Executables
$(BIN_FOLDER)/vit.exe : \
\
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(OBJ_FOLDER)/main.o
	$(CC) $(CFLAGS) $^ -o $@



# OMP OBJs
$(OMP_OBJ_FOLDER)/datatypes.o \
$(OMP_OBJ_FOLDER)/modules.o \
$(OMP_OBJ_FOLDER)/conv2d.o \
$(OMP_OBJ_FOLDER)/attention.o \
$(OMP_OBJ_FOLDER)/vision_transformer.o \
\
: $(OMP_OBJ_FOLDER)/%.o : $(OMP_SRC_FOLDER)/%.cpp
	$(CC) -c $(CFLAGS) $(OMPFLAGS) $^ -o $@

# OMP Executables
$(OMP_BIN_FOLDER)/vit.exe : \
\
$(OMP_OBJ_FOLDER)/datatypes.o \
$(OMP_OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OMP_OBJ_FOLDER)/conv2d.o \
$(OMP_OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OMP_OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(OBJ_FOLDER)/main.o
	$(CC) $(CFLAGS) $(OMPFLAGS) $^ -o $@



# Test OBJs
$(TEST_OBJ_FOLDER)/test_datatypes.o \
$(TEST_OBJ_FOLDER)/test_modules.o \
$(TEST_OBJ_FOLDER)/test_mlp.o \
$(TEST_OBJ_FOLDER)/test_conv2d.o \
$(TEST_OBJ_FOLDER)/test_attention.o \
$(TEST_OBJ_FOLDER)/test_block.o \
$(TEST_OBJ_FOLDER)/test_patch_embed.o \
$(TEST_OBJ_FOLDER)/test_vision_transformer.o \
$(TEST_OBJ_FOLDER)/test_utils.o \
\
: $(TEST_OBJ_FOLDER)/%.o : $(TEST_SRC_FOLDER)/%.cpp
	$(CC) -c $(CFLAGS) $^ -o $@

# Test Executables
$(TEST_BIN_FOLDER)/test_datatypes.exe \
$(TEST_BIN_FOLDER)/test_modules.exe \
$(TEST_BIN_FOLDER)/test_mlp.exe \
$(TEST_BIN_FOLDER)/test_conv2d.exe \
$(TEST_BIN_FOLDER)/test_attention.exe \
$(TEST_BIN_FOLDER)/test_block.exe \
$(TEST_BIN_FOLDER)/test_patch_embed.exe \
$(TEST_BIN_FOLDER)/test_vision_transformer.exe \
$(TEST_BIN_FOLDER)/test_utils.exe \
\
: $(TEST_BIN_FOLDER)/%.exe : \
\
$(OBJ_FOLDER)/datatypes.o \
$(OBJ_FOLDER)/modules.o \
$(OBJ_FOLDER)/mlp.o \
$(OBJ_FOLDER)/conv2d.o \
$(OBJ_FOLDER)/attention.o \
$(OBJ_FOLDER)/block.o \
$(OBJ_FOLDER)/patch_embed.o \
$(OBJ_FOLDER)/vision_transformer.o \
$(OBJ_FOLDER)/utils.o \
$(TEST_OBJ_FOLDER)/%.o
	$(CC) $(CFLAGS) $^ -o $@



# ===========================================================================
# NVTX profiling builds (for Nsight Systems)
# ===========================================================================
# Convenience targets:
#   make prof_cpp   -> prof_bin/vit_prof.exe       (serial, NVTX, debug info)
#   make prof_omp   -> prof_bin/vit_omp_prof.exe   (OpenMP, NVTX, debug info)
#   make prof       -> both
# Override the CUDA toolkit location if nvtx3 headers live elsewhere:
#   make prof CUDA_HOME=/opt/cuda
.PHONY : prof prof_cpp prof_omp
prof : prof_cpp prof_omp
prof_cpp : $(PROF_BIN_FOLDER)/vit_prof.exe
prof_omp : $(PROF_BIN_FOLDER)/vit_omp_prof.exe

# Serial profiling OBJs (every source, all NVTX-instrumented)
$(PROF_OBJ_FOLDER)/datatypes.o \
$(PROF_OBJ_FOLDER)/modules.o \
$(PROF_OBJ_FOLDER)/mlp.o \
$(PROF_OBJ_FOLDER)/conv2d.o \
$(PROF_OBJ_FOLDER)/attention.o \
$(PROF_OBJ_FOLDER)/block.o \
$(PROF_OBJ_FOLDER)/patch_embed.o \
$(PROF_OBJ_FOLDER)/vision_transformer.o \
$(PROF_OBJ_FOLDER)/utils.o \
$(PROF_OBJ_FOLDER)/main.o \
\
: $(PROF_OBJ_FOLDER)/%.o : $(SRC_FOLDER)/%.cpp
	@mkdir -p $(@D)
	$(CC) -c $(PROF_CFLAGS) $^ -o $@

# OpenMP profiling OBJs (the subset that has an omp_src override)
$(OMP_PROF_OBJ_FOLDER)/datatypes.o \
$(OMP_PROF_OBJ_FOLDER)/modules.o \
$(OMP_PROF_OBJ_FOLDER)/conv2d.o \
$(OMP_PROF_OBJ_FOLDER)/attention.o \
$(OMP_PROF_OBJ_FOLDER)/vision_transformer.o \
\
: $(OMP_PROF_OBJ_FOLDER)/%.o : $(OMP_SRC_FOLDER)/%.cpp
	@mkdir -p $(@D)
	$(CC) -c $(PROF_CFLAGS) $(OMPFLAGS) $^ -o $@

# Serial profiling executable
$(PROF_BIN_FOLDER)/vit_prof.exe : \
\
$(PROF_OBJ_FOLDER)/datatypes.o \
$(PROF_OBJ_FOLDER)/modules.o \
$(PROF_OBJ_FOLDER)/mlp.o \
$(PROF_OBJ_FOLDER)/conv2d.o \
$(PROF_OBJ_FOLDER)/attention.o \
$(PROF_OBJ_FOLDER)/block.o \
$(PROF_OBJ_FOLDER)/patch_embed.o \
$(PROF_OBJ_FOLDER)/vision_transformer.o \
$(PROF_OBJ_FOLDER)/utils.o \
$(PROF_OBJ_FOLDER)/main.o
	@mkdir -p $(@D)
	$(CC) $(PROF_CFLAGS) $^ -o $@ $(NVTX_LIB)

# OpenMP profiling executable (same object mix as omp_bin/vit.exe:
# omp overrides for datatypes/modules/conv2d/attention/vision_transformer,
# serial builds for the rest -- all NVTX-instrumented)
$(PROF_BIN_FOLDER)/vit_omp_prof.exe : \
\
$(OMP_PROF_OBJ_FOLDER)/datatypes.o \
$(OMP_PROF_OBJ_FOLDER)/modules.o \
$(PROF_OBJ_FOLDER)/mlp.o \
$(OMP_PROF_OBJ_FOLDER)/conv2d.o \
$(OMP_PROF_OBJ_FOLDER)/attention.o \
$(PROF_OBJ_FOLDER)/block.o \
$(PROF_OBJ_FOLDER)/patch_embed.o \
$(OMP_PROF_OBJ_FOLDER)/vision_transformer.o \
$(PROF_OBJ_FOLDER)/utils.o \
$(PROF_OBJ_FOLDER)/main.o
	@mkdir -p $(@D)
	$(CC) $(PROF_CFLAGS) $(OMPFLAGS) $^ -o $@ $(NVTX_LIB)



# ===========================================================================
# CUDA build (hand-written kernels; Linear ported so far)
# ===========================================================================
#   make cuda                     -> cuda_bin/vit.exe   (Linear runs on the GPU)
#   make cuda CUDA_ARCH=sm_70     -> for a non-A100 GPU
# Fully isolated in cuda_obj/ + cuda_bin/, so the CPU/OpenMP builds are untouched.
# modules is compiled by nvcc (it holds the kernel); the other 9 units stay
# plain C++ (g++) and are linked in by nvcc.
.PHONY : cuda
cuda : $(CUDA_BIN_FOLDER)/vit.exe

# the CUDA translation unit (holds linear_kernel)
$(CUDA_OBJ_FOLDER)/modules.o : $(CUDA_SRC_FOLDER)/modules.cu
	@mkdir -p $(@D)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

# the remaining units stay plain C++ but live in cuda_obj/ so nothing is shared
$(CUDA_OBJ_FOLDER)/datatypes.o \
$(CUDA_OBJ_FOLDER)/mlp.o \
$(CUDA_OBJ_FOLDER)/conv2d.o \
$(CUDA_OBJ_FOLDER)/attention.o \
$(CUDA_OBJ_FOLDER)/block.o \
$(CUDA_OBJ_FOLDER)/patch_embed.o \
$(CUDA_OBJ_FOLDER)/vision_transformer.o \
$(CUDA_OBJ_FOLDER)/utils.o \
$(CUDA_OBJ_FOLDER)/main.o \
\
: $(CUDA_OBJ_FOLDER)/%.o : $(SRC_FOLDER)/%.cpp
	@mkdir -p $(@D)
	$(CC) -c $(CFLAGS) $< -o $@

# link with nvcc so it pulls in the CUDA runtime
$(CUDA_BIN_FOLDER)/vit.exe : \
\
$(CUDA_OBJ_FOLDER)/modules.o \
$(CUDA_OBJ_FOLDER)/datatypes.o \
$(CUDA_OBJ_FOLDER)/mlp.o \
$(CUDA_OBJ_FOLDER)/conv2d.o \
$(CUDA_OBJ_FOLDER)/attention.o \
$(CUDA_OBJ_FOLDER)/block.o \
$(CUDA_OBJ_FOLDER)/patch_embed.o \
$(CUDA_OBJ_FOLDER)/vision_transformer.o \
$(CUDA_OBJ_FOLDER)/utils.o \
$(CUDA_OBJ_FOLDER)/main.o
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ -o $@
