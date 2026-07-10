#!/bin/bash
# Run + profile the CUDA build (Linear on GPU) on one Leonardo Booster node,
# with Nsight Systems capturing the GPU (CUDA) timeline, and a correctness
# check of the GPU predictions against the plain CPU build.
#
# What it measures: ONE forward inference of ONE batch of B images (no training,
# no backward, no epoch). main.cpp runs timed_forward once, so nsys captures a
# single forward pass.
#
# Submit:  sbatch profile_cuda.slurm      Watch:  squeue --me   (output: <jobid>.out)

#SBATCH --job-name=vit-cuda
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --account=tra26_sumsch_0          # <-- set your account (check: saldo -b)
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1

### environment ###
module load cuda
# Python env (torch + timm) only needed the first time, to generate the model
# and the input .cpic. Comment out once models/ and data/prof/ exist.
source venv/bin/activate

B="${B:-16}"
MODEL="models/vit_1.cvit"
PIC="data/prof/pic_b${B}.cpic"
mkdir -p reports data/prof

echo "node = $(hostname) | gpu = $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) | B = $B | "

### 1. build: NVTX CUDA binary — Linear on GPU, every other kernel serial CPU ###
make cuda_serial_prof

### 2. model + input (generated once, then reused) ###
[ -f "$MODEL" ] || bash create_models.sh
[ -f "$PIC" ]   || python3 scripts/random_cpic.py "$PIC" "$B" "$B" 3 224 224 0.0 1.0

### 3. GPU profile with Nsight Systems (CUDA timeline) ###
echo; echo "=== nsys capture (GPU) -> reports/cuda_b${B}.nsys-rep ==="
nsys profile --trace=cuda,nvtx,osrt --force-overwrite=true \
     -o "reports/cuda_b${B}" \
     ./cuda_bin/vit_cuda_serial_prof.exe "$MODEL" "$PIC" "reports/gpu_b${B}.cprd" /tmp/m.csv

### 4. text summaries (GPU kernels + memory transfers) ###
echo; echo "=== NVTX ranges (CPU stages + linear; look at where time goes) ==="
nsys stats --report nvtx_pushpop_sum      --format table "reports/cuda_b${B}.nsys-rep" || true
echo; echo "=== GPU kernel time (look for linear_kernel) ==="
nsys stats --report cuda_gpu_kern_sum     --format table "reports/cuda_b${B}.nsys-rep" || true
echo; echo "=== GPU memcpy time (HtoD / DtoH — expected to dominate for now) ==="
nsys stats --report cuda_gpu_mem_time_sum --format table "reports/cuda_b${B}.nsys-rep" || true
echo; echo "=== CUDA API call time ==="
nsys stats --report cuda_api_sum          --format table "reports/cuda_b${B}.nsys-rep" || true

echo; echo "Done. Copy reports/cuda_b${B}.nsys-rep to your laptop and open it in the Nsight Systems GUI."
