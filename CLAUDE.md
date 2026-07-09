# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A bachelor thesis (C-ViT, maintainer Alex Pegoraro) that reimplements a Vision Transformer from scratch in C++ and benchmarks it against a reference Python/`timm` implementation. There are **three implementations of the same model** that must produce numerically identical predictions:

1. **Plain C++** (`src/`, `include/`) — the from-scratch implementation.
2. **OpenMP C++** (`omp_src/`) — a *partial* parallelization that overrides only some translation units of the plain C++ build (see "OMP build" below).
3. **Python** (`timm_train_vit/`) — the ground-truth reference using a vendored, FBK-modified copy of `timm`.

The C++ model is hand-tuned to mirror `timm.create_model('vit_base_patch16_224', num_classes=100)` exactly, so correctness is defined as: C++ output ≈ Python output within a float threshold.

## Environment

Scripts are **bash** and call `python3`, `make`, and `g++` — even though the repo lives on Windows. Run them from a Unix-like shell (MSYS2/Git Bash; the VS Code config points at `C:/msys64/ucrt64/bin/gcc.exe`). Executables carry a `.exe` suffix. `python3` must have `torch` and `timm` importable (the modified `timm` is vendored in `timm_train_vit/timm/`, so run Python scripts with `timm_train_vit/` as the working dir or on the path).

## Build

Compiler flags come from `makefile`: `g++ -std=c++11 -O3`, plus `-fopenmp` for the OMP build.

```bash
make bin/vit.exe                       # plain C++ executable
make omp_bin/vit.exe                   # OpenMP executable
bash compile.sh                        # builds both of the above (creates obj/ dirs first)
make test_bin/test_<component>.exe     # build one parity test (e.g. test_attention)
make clean                             # remove binaries/objs + out_comparison/ + logs/ (keeps data/ + models/)
make clean_everything                  # also removes data/, models/, out/, measures/, test_files/
```

**OMP build is a partial override.** `omp_bin/vit.exe` links OMP object files for only `datatypes, modules, conv2d, attention, vision_transformer` and reuses the *plain* `obj/` builds of `mlp, block, patch_embed, utils, main`. So `omp_src/` intentionally contains only those five files; the rest are shared. If you change a shared component's `.cpp`/`.h`, both builds are affected.

## Running the benchmark

The full pipeline, in order (all driven by `params.sh`, which exports dataset dims, comparison thresholds, and `THREAD_LIST`):

```bash
bash compile.sh          # 1. build bin/ and omp_bin/
bash create_dataset.sh   # 2. random .cpic picture batches -> data/
bash create_models.sh    # 3. two ViT models -> models/vit_{1,2}.{cvit,pt}
bash run_cpp.sh          # 4a. run plain C++, outputs -> out/cpp_{1,2}/, timings -> measures/
bash run_py.sh           # 4b. run Python reference
bash run_omp.sh          # 4c. run OMP for each thread count in THREAD_LIST
bash elaborate.sh        # 5. compare all outputs vs C++ + aggregate timings -> logs/
```

Final results land in `logs/`: `dataset_info.txt`, `model_info.txt`, `output_analysis.txt` (do all implementations agree?), `measures_analysis.txt` (which is faster?). Adjust dataset size, float comparison thresholds, and OMP thread counts in `params.sh` — not in the scripts.

The runnable unit is `vit.exe <cvit_path> <cpic_path> <cprd_path> <measure_file_path>` (same CLI for `bin/`, `omp_bin/`, and `timm_train_vit/vit.py`). It loads a model + one picture batch, runs a timed forward pass, writes a prediction file, and appends a `;`-separated timing row to the measure CSV.

## Parity tests

Tests prove the C++ and Python implementations of a single component compute the same thing. **Run them in two side-by-side terminals** — there is no automated assertion; you eyeball that the printed floats match to the least-significant digits:

```bash
# terminal A (from timm_train_vit/)
python3 test_<component>.py
# terminal B
make test_bin/test_<component>.exe && ./test_bin/test_<component>.exe
```

Each test hard-codes a component, its parameters, and a random input. To keep the two languages on identical data, `python3 scripts/create_tensor.py <shape>` prints matching Python and C++ snippets for constructing the same random tensor — regenerate both sides together when changing test inputs. Test binaries link against the plain `obj/` builds, so build the component first.

## Architecture

**Component mirror.** `include/*.h` + `src/*.cpp` map one-to-one onto `timm`'s ViT building blocks: `patch_embed` (Conv2d tokenizer) → `block` (`attention` + `mlp` with `LayerNorm`/`LayerScale`) → `vision_transformer` (wrapper). `modules.h` holds the leaf ops (`Linear`, `LayerNorm`, `LayerScale`, `Activation`, `ReLU`/`GELU`, `global_pool_nlc`). When editing a C++ component, cross-check the corresponding class/method in `timm_train_vit/timm/models/vision_transformer.py`.

**Data types** (`datatypes.h`): `RowVector`, `Matrix`, `Tensor` (always 3-D: B×N×C), `PictureBatch` (B×C×H×W), `PredictionBatch`. These are **move-only** (copy ctor/assignment are `= delete`d); pass by `&&`/`std::move`, not by value. Every type implements `to_ofstream`/`from_ifstream` for binary serialization.

**Custom binary formats** (serialized via `utils.h` `load_*`/`store_*`, which delegate to the types' stream methods):
- `.cvit` — a serialized `VisionTransformer` (C++ side). Python reads/writes it via `timm_train_vit/cvit_utils.py`.
- `.cpic` — a `PictureBatch` (the dataset unit).
- `.cprd` — a `PredictionBatch` (model output). `.pt` is the standard PyTorch state-dict used by the Python path only; `convert_pt_cvit.py` bridges the two.

**Forward pass** (`vision_transformer.cpp`): `forward_features` (patch_embed → `position_embed` prepends cls/reg tokens and adds pos_embed → blocks → norm) then `forward_head` (`pool` via `global_pool_nlc` → optional fc_norm → `head`). `timed_forward` is the benchmarked variant: it re-implements the same pipeline while timing each phase (patch-embed, embed, per-block attn/mlp, head) into a `RowVector`, which `main.cpp` flattens into the measure CSV.

**Modified `timm`.** `timm_train_vit/timm/` is a vendored copy with project-specific additions — notably a `timed_forward` method on the ViT model (`timm/models/vision_transformer.py`) that stock `timm` does not have, mirroring the C++ `timed_forward`. Treat this folder as source you may edit, not an untouched dependency.
