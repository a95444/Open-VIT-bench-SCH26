# Profiling guide — understanding `profile.slurm` end to end

This document explains, in depth: what the codebase is, what happens step by step
when you run `sbatch profile.slurm`, what every compiled file does, and exactly what
output to expect and how to read it. Read it top to bottom once and you'll have a
full mental model of the project and the profiling flow.

---

## 1. What this project is (the 30-second version)

It's a **Vision Transformer (ViT) inference engine written from scratch in C++**, plus
a Python reference built on `timm`. There are **three implementations of the exact same
model**, which must produce identical predictions:

| Implementation | Where | What it is |
|---|---|---|
| **Serial C++** | `src/` + `include/` | plain nested-loop implementation |
| **OpenMP C++** | `omp_src/` (overrides 5 files) | same code, `#pragma omp parallel for` on the hot loops |
| **Python** | `timm_train_vit/` | ground-truth reference using a modified `timm` |

The model is `vit_base_patch16_224` with `num_classes=100`. Concretely that means:

- Input image batch: `B × 3 × 224 × 224` (we use **B=16**).
- Patch size 16×16 → a 14×14 grid = **196 patches**, + 1 class token = **N = 197 tokens**.
- Embedding dimension **C = 768**, **12 transformer blocks (depth)**, **12 attention heads**
  (head_dim 64), MLP hidden size **3072** (= 4×768), output **100 classes**.

**Goal of the profiling:** measure where the forward pass spends its time so we know
which pieces to rewrite as **CUDA** kernels (and later distribute with **MPI** across
multiple GPUs / 2 nodes). Right now everything runs on CPU only.

---

## 2. The big picture of `profile.slurm`

`profile.slurm` is a Slurm batch script. `sbatch profile.slurm` **submits** it to the
queue and returns immediately; the actual work runs on a **compute node**, and all
console output is written to `<jobid>.out` (and errors to `<jobid>.err`) in this folder.

What the job does, in order:

```
sbatch profile.slurm
   │
   ├─ (Slurm) allocate 1 Booster node, 32 cores, 30 min
   ├─ module load cuda            → gives us `nsys`, `g++`, and the nvtx3 headers
   ├─ source venv/bin/activate    → Python with torch + timm (to generate model/input)
   │
   └─ bash profile_nsys.sh
         ├─ 1. make prof          → compile the two NVTX-instrumented executables
         ├─ 2. create_models.sh   → generate the model file  models/vit_1.cvit (+ .pt)
         ├─ 3. random_cpic.py     → generate the input file   data/prof/pic_b16.cpic
         ├─ 4. nsys profile (serial)  → reports/cpp_b16.nsys-rep
         ├─ 5. nsys profile (openmp)  → reports/omp_b16.nsys-rep
         └─ 6. nsys stats         → print NVTX region-time tables to stdout (the .out file)
```

The heavy lifting (`profile_nsys.sh`) is deliberately reused so you can also run it by
hand on an interactive node (see §9).

---

## 3. Step 1 — what `make prof` compiles

`make prof` builds **two executables**. Both are the normal build **plus** three extra
compiler flags: `-g` (debug symbols so Nsight can name functions), `-fno-omit-frame-pointer`,
and `-DUSE_NVTX` (turns on the NVTX ranges — see §6). It also adds
`-I$CUDA_HOME/include` (for `nvtx3/nvToolsExt.h`) and links `-ldl`.

Objects go into separate folders (`prof_obj/`, `omp_prof_obj/`) so the normal benchmark
build in `obj/`/`omp_obj/` is never touched.

### `prof_bin/vit_prof.exe` — the **serial** build
Compiles all 10 sources from `src/` and links them:

```
datatypes.o modules.o mlp.o conv2d.o attention.o
block.o patch_embed.o vision_transformer.o utils.o main.o
```

### `prof_bin/vit_omp_prof.exe` — the **OpenMP** build
This is the important subtlety. It links a **mix**: the 5 files that have an OpenMP
version come from `omp_src/` (compiled with `-fopenmp`), the other 5 come from `src/`:

```
from omp_src/ (parallel):  datatypes  modules  conv2d  attention  vision_transformer
from src/     (serial):    mlp  block  patch_embed  utils  main
```

Why this still parallelizes almost everything: the *leaf* compute ops live in
`modules.o` (Linear, LayerNorm, GELU), `attention.o`, `conv2d.o`, `datatypes.o` — all the
OpenMP versions. The serial `mlp.o`/`block.o`/`patch_embed.o` are just *orchestrators*
that call those parallel leaf ops. So `block.forward` is "serial" code, but every
expensive thing it calls runs multi-threaded.

> This is exactly the same object mix as the normal `omp_bin/vit.exe`; we just added NVTX.

---

## 4. What each compiled file does

Think of it as a call tree. `main` drives everything; each class maps 1:1 to a `timm`
ViT component.

| File (`src/`) | Class / role | Heavy compute? |
|---|---|---|
| `main.cpp` | entry point: load model, load input, run+time forward, save output | no |
| `utils.cpp` | `load_cvit` / `load_cpic` / `store_cprd` — read/write the binary formats | I/O |
| `datatypes.cpp` | `RowVector`, `Matrix`, `Tensor`, `PictureBatch`, `PredictionBatch` — storage + `at()/set()` accessors, `+`, `+=`, `copy_tensor` | some (element-wise, softmax in PredictionBatch) |
| `vision_transformer.cpp` | `VisionTransformer` — the top wrapper; `forward_features` → blocks → `forward_head`; `position_embed`; `timed_forward` | pos_embed |
| `patch_embed.cpp` | `PatchEmbed` — turn image into patch tokens (calls Conv2d + flatten) | via conv |
| `conv2d.cpp` | `Conv2d::forward` — the strided convolution that tokenizes the image | **yes** |
| `block.cpp` | `Block` — one transformer block: norm→attn→scale→resid, then norm→mlp→scale→resid | via attn/mlp |
| `attention.cpp` | `Attention` — q/k/v Linear, then `multi_head_attention` (QKᵀ, softmax, A·V), then proj | **yes** |
| `mlp.cpp` | `Mlp` — fc1 → GELU → fc2 | via Linear |
| `modules.cpp` | `Linear` (matmul), `LayerNorm`, `LayerScale`, `Activation` (GELU/ReLU), `global_pool_nlc` | **yes (Linear dominates)** |

The single most expensive thing is **`Linear::operator()`** — a naive
`B × N × out_features × in_features` loop — because it's called for q, k, v, proj (768→768),
the MLP `fc1` (768→3072) and `fc2` (3072→768), and the head (768→100), **in every one of
the 12 blocks**. Second is **`multi_head_attention`** (O(N²·d) per head × 12 heads × 12 blocks).

---

## 5. What the executable does at runtime (`main.cpp`)

The command line is:
```
vit_prof.exe  <model.cvit>  <input.cpic>  <output.cprd>  <measures.csv>
```
Inside `main` (see [src/main.cpp](src/main.cpp)):

1. `load_cvit(model)` — deserialize the whole `VisionTransformer` (weights) from the `.cvit` file.
2. `load_cpic(input)` — deserialize the `PictureBatch` (the 16 images) from the `.cpic` file.
3. `vit.timed_forward(pic, pred, times)` — **the forward pass we care about**. It runs the
   full network and fills a `RowVector times` with per-stage timings (chrono), size `4 + 2·depth = 28`.
4. `store_cprd(output)` — write the `PredictionBatch` (per-image class + probabilities).
5. Append one `;`-separated row to `measures.csv` (batch, depth, load times, the 28 stage
   times, store time). In profiling we point this at a scratch file and ignore it — NVTX is
   our real data source.

### The forward pass, stage by stage (`timed_forward`)
```
patch_embed.forward   → conv2d + flatten (+ optional norm)        [stage 0]
position_embed        → prepend class token, add positional embed [stage 1]
for each of 12 blocks:
    blk_attn: norm1 → attention(q,k,v, QKᵀ, softmax, A·V, proj) → layerscale → +residual
    blk_mlp : norm2 → fc1 → GELU → fc2 → layerscale → +residual
final norm → pool (take class token) → head Linear → softmax → PredictionBatch
```

---

## 6. The NVTX ranges — what you'll actually see in the profile

NVTX is "add a named marker around a region of code so Nsight draws a labelled bar on the
timeline." They compile to **nothing** unless `-DUSE_NVTX` is set (so the benchmark build
is unaffected). See [include/vit_nvtx.h](include/vit_nvtx.h). The ranges we added:

| NVTX label | Wraps | Meaning / future CUDA kernel |
|---|---|---|
| `patch_embed` | `PatchEmbed::forward` | tokenizer (coarse group) |
| `conv2d` | `Conv2d::forward` | the strided convolution |
| `pos_embed` | `VisionTransformer::position_embed` | class token + positional add |
| `blk_attn` | block's attention half | coarse group per block |
| `blk_mlp` | block's MLP half | coarse group per block |
| `linear` | every `Linear::operator()` | **the GEMMs** (q/k/v/proj, fc1, fc2, head) |
| `attention` | `multi_head_attention` | QKᵀ + softmax + A·V (scaled dot-product) |
| `layernorm` | `LayerNorm::operator()` | normalization |
| `gelu` | `Activation::operator()` | activation |
| `pool` | `global_pool_nlc` | class-token pooling (tiny) |

On the timeline they **nest**: e.g. inside one `blk_mlp` you'll see `layernorm`, `linear`,
`gelu`, `linear`. Because `Linear` is generic, the coarse `blk_attn`/`blk_mlp`/`patch_embed`
groups are what let you tell *which* `linear` is which (qkv vs fc1/fc2 vs head).

**Important note on placement:** the ranges sit on the **main thread**, outside the
`#pragma omp parallel for`. So in the OpenMP profile a `linear` bar shows the *wall-clock*
duration of that op (the parallel region), which is exactly what we want to compare against
the serial version. We are intentionally **not** profiling per-thread behavior.

---

## 7. Steps 2–3 — generating the model and the input

These need the Python env (torch + timm), which is why `profile.slurm` activates `venv`.

- **`create_models.sh`** runs `timm_train_vit/create_model.py`, which does
  `timm.create_model('vit_base_patch16_224', num_classes=100)` with **random** weights and
  saves it twice: `models/vit_1.cvit` (for C++) and `models/vit_1.pt` (for the Python
  reference). It makes a second model `vit_2` too; we only use `vit_1`. Random weights are
  fine — we're measuring *time*, not accuracy. (Runs once; if `models/vit_1.cvit` already
  exists the script skips this.)
- **`scripts/random_cpic.py`** creates one deterministic batch of 16 random images at
  `data/prof/pic_b16.cpic` (`torch.rand(16, 3, 224, 224)`), in the custom `.cpic` binary format.

### The three binary formats (all little custom formats, read/written by `utils.cpp` / `cvit_utils.py`)
- **`.cvit`** — a serialized `VisionTransformer` (all weights + config).
- **`.cpic`** — a `PictureBatch` (the input images). Header `CPIC` + dims + float data.
- **`.cprd`** — a `PredictionBatch` (the output: per-image predicted class + probabilities).

---

## 8. Steps 4–6 — the captures and what to expect in the output

The script runs `nsys profile` twice (serial, then OpenMP with 32 threads), each doing
**one full forward pass**, then prints summary tables.

```
nsys profile --trace=nvtx,osrt --sample=none --force-overwrite=true \
     -o reports/cpp_b16  ./prof_bin/vit_prof.exe  models/vit_1.cvit  data/prof/pic_b16.cpic ...
```
- `--trace=nvtx,osrt` = record our NVTX ranges + OS runtime calls. `--sample=none` = we
  don't need statistical CPU sampling; NVTX ranges are exact recorded events, so **one
  forward pass is enough**.

### What lands on disk
```
reports/cpp_b16.nsys-rep     ← open in the Nsight Systems GUI (serial timeline)
reports/omp_b16.nsys-rep     ← open in the Nsight Systems GUI (OpenMP timeline)
reports/*.cprd               ← the model outputs (can be diffed for correctness)
```

### What prints into `<jobid>.out`
The header echoes (node, cores, CUDA_HOME, threads) and then **two NVTX region tables**,
one per implementation, that look like this (numbers illustrative):

```
** NVTX Range Summary (nvtx_pushpop_sum):
 Time(%)  Total Time(ns)  Instances   Avg(ns)     Range
 -------  --------------  ---------  ----------   ---------
   61.2      7,350,000,000     84     87,500,000   linear
   28.9      3,470,000,000     12    289,000,000   attention
    4.1        490,000,000     25     19,600,000   layernorm
    2.8        330,000,000     12     27,700,000   gelu
    1.9        225,000,000      1    225,000,000   conv2d
    ...
```
Read it as: **`linear` and `attention` dominate** (expected — that's the GEMMs and the
attention core). "Instances" = how many times the range fired in one forward pass
(e.g. `linear` fires many times: qkv+proj+fc1+fc2 per block × 12 + head). Sort by
`Total Time` = your **CUDA porting priority list**.

> If `nsys stats --report nvtx_pushpop_sum` complains the report name is unknown (varies by
> nsys version), list the available ones with `nsys stats --help-reports` and try
> `nvtx_sum` instead.

### Reading it in the GUI (recommended)
Copy the `.nsys-rep` to your laptop and open it:
```
scp 'a08tra06@login07.leonardo.cineca.it:~/challenge/Open-VIT-bench-SCH26/reports/*.nsys-rep' .
```
You'll get a timeline; expand the **NVTX** row. Bar **width = time**. Compare the serial
timeline vs the OpenMP one at a glance: the same `linear`/`attention` bars should be
visibly narrower in the OpenMP run (that's your speedup), while tiny serial bits
(`pos_embed`, `pool`, the softmax in `PredictionBatch`) barely shrink.

### Two caveats when interpreting the absolute numbers
1. **Asserts are ON.** The profiling build is `-O3` but **not** `-DNDEBUG`, so the bounds
   checks inside every `at()`/`set()` run in the hot loops. Absolute times are inflated,
   but the *ranking* of regions is still representative. (We can add an `-DNDEBUG` variant
   later to measure this overhead.)
2. It's **one forward pass**. Good enough to rank regions; if you want steadier averages,
   run it a few times or raise the batch size (`B=64 sbatch profile.slurm`).

---

## 9. How to run everything yourself (cheat sheet)

### A) The whole profiling flow via Slurm (normal path)
```bash
# edit profile.slurm once: set --account=<your account>  (check with `saldo -b`)
sbatch profile.slurm
squeue --me                 # PD = pending, R = running
tail -f <jobid>.out         # watch progress / read the NVTX tables at the end
```

### B) Interactively on a compute node (great for debugging, when Slurm is up)
```bash
srun --account=<acct> --partition=boost_usr_prod --qos=boost_qos_dbg \
     --nodes=1 --ntasks=1 --cpus-per-task=32 --gres=gpu:1 --time=00:30:00 --pty bash
# then, on the compute node:
module load cuda
source venv/bin/activate
OMP_THREADS=32 bash profile_nsys.sh
```

### C) Individual pieces (understand/verify each stage)
```bash
module load cuda
source venv/bin/activate

# build just one:
make prof_cpp                 # -> prof_bin/vit_prof.exe   (serial)
make prof_omp                 # -> prof_bin/vit_omp_prof.exe (OpenMP)

# make model + input by hand:
bash create_models.sh
python3 scripts/random_cpic.py data/prof/pic_b16.cpic 16 16 3 224 224 0.0 1.0

# run WITHOUT profiling, just to see it work (writes a .cprd + a measures row):
./prof_bin/vit_prof.exe models/vit_1.cvit data/prof/pic_b16.cpic /tmp/out.cprd /tmp/m.csv

# profile one manually:
nsys profile --trace=nvtx,osrt --sample=none -o reports/cpp_b16 \
     ./prof_bin/vit_prof.exe models/vit_1.cvit data/prof/pic_b16.cpic /tmp/out.cprd /tmp/m.csv
nsys stats --report nvtx_pushpop_sum reports/cpp_b16.nsys-rep
```

### D) Sanity check: NVTX didn't change the math
The profiling build must produce the same predictions as the untouched benchmark build:
```bash
bash compile.sh                                  # builds bin/vit.exe (no NVTX)
./bin/vit.exe        models/vit_1.cvit data/prof/pic_b16.cpic /tmp/plain.cprd /tmp/m.csv
./prof_bin/vit_prof.exe models/vit_1.cvit data/prof/pic_b16.cpic /tmp/nvtx.cprd  /tmp/m.csv
python3 scripts/compare_cpred.py /tmp/plain.cprd /tmp/nvtx.cprd /tmp/cmp.txt 0.0001 0.000001
```

### E) The full 3-way benchmark (context — from the README, not needed for profiling)
`bash compile.sh` → `bash create_dataset.sh` → `bash create_models.sh` →
`bash run_cpp.sh` / `bash run_py.sh` / `bash run_omp.sh` → `bash elaborate.sh`.
Results land in `logs/` (`output_analysis.txt`, `measures_analysis.txt`). This is the
original timing+correctness harness; our NVTX profiling is a separate, finer-grained view.

---

## 10. Troubleshooting quick reference

| Symptom | Cause / fix |
|---|---|
| `sbatch` "nothing happens" | Normal — it's async. Check `squeue --me`; output is in `<jobid>.out`. |
| Job never starts (`PD`) | Slurm down or queue full; check `REASON` in `squeue`. |
| `make prof` → `Nothing to be done` | Already built and up to date. Force a rebuild with `make clean` then `make prof`. |
| `‘X’ is predetermined ‘shared’ for ‘shared’` | Old OpenMP bug — already fixed (removed `shared()` clauses). `grep -rn "shared(" omp_src/` should return nothing. |
| `No module named 'timm'` | `uv pip install timm` into the venv. |
| `nvtx3/nvToolsExt.h: No such file` | `module load cuda`, or pass `make prof CUDA_HOME=/path/to/cuda`. |
| `nsys: command not found` | `module load cuda` (or `module load nsight-systems`). |
| unknown report `nvtx_pushpop_sum` | `nsys stats --help-reports`; use `nvtx_sum` instead. |

---

## 11. Where this is heading (why we profile)

The NVTX ranking tells us the CUDA porting order. Expected conclusion: **`linear`
(the MLP + q/k/v/proj GEMMs) and `attention` dominate**, so those become the first GPU
kernels (cuBLAS GEMMs + a fused attention kernel), `conv2d` (patch embed) next, and the
element-wise ops (`layernorm`, `gelu`) fuse cheaply. The batch/image axis is the natural
split for **MPI across the 2 nodes / multiple GPUs**. After the GPU port we re-profile
with Nsight Systems (now with CUDA/NCCL trace) and Nsight Compute for kernel-level tuning.
