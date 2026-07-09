#!/bin/bash
# Nsight Systems profiling of the serial (src/) and OpenMP (omp_src/) ViT
# forward pass, on one Leonardo compute node.
#
# Submit with:   sbatch profile.slurm
# Watch:         squeue --me      (output goes to <jobid>.out / <jobid>.err)
#
# NOTE: this profiling is CPU + OpenMP only (no GPU yet). The GPU request below
# is just because boost_usr_prod requires it; switch to a CPU partition
# (e.g. dcgp_usr_prod, no --gres) if you have allocation there.

#SBATCH --job-name=vit-nsys
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --account=tra26_sumsch_0         # <-- set your training/project account
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32                # Booster node = 32 physical cores
#SBATCH --gres=gpu:1
#SBATCH --exclusive

### environment ###
module load cuda
# python env only needed the FIRST time, to generate models/vit_1.cvit and the
# input .cpic (requires torch + timm). Comment out once those files exist.
source /leonardo/home/usertrain/a08tra06/challenge/Open-VIT-bench-SCH26/venv/bin/activate

# CUDA_HOME is set by `module load cuda` (provides nvtx3 headers + nsys).
export OMP_THREADS="${SLURM_CPUS_PER_TASK:-8}"   # threads for the OpenMP capture
export B="${B:-16}"                              # fixed batch size

echo "node       = $(hostname)"
echo "cores      = $SLURM_CPUS_PER_TASK"
echo "CUDA_HOME  = $CUDA_HOME"
echo "OMP_THREADS= $OMP_THREADS  B=$B"

### build + generate + capture (all handled by the helper script) ###
bash profile_nsys.sh
