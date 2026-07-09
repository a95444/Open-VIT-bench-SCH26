#!/usr/bin/env bash


#profile_nsys.sh
# Profile the serial (src/) and OpenMP (omp_src/) ViT forward pass with
# Nsight Systems, using the NVTX ranges compiled into the *_prof executables.
#
# What it does:
#   1. builds prof_bin/vit_prof.exe and prof_bin/vit_omp_prof.exe (make prof)
#   2. makes sure a model exists (models/vit_1.cvit via create_models.sh)
#   3. creates ONE deterministic fixed-size input  data/prof/pic_b<B>.cpic
#   4. runs `nsys profile` on each executable -> reports/<impl>_b<B>.nsys-rep
#   5. prints the NVTX region time summary (widest region = first CUDA target)
#
# Open the reports/*.nsys-rep files in the Nsight Systems GUI to read the
# timeline visually. Tune with env vars, e.g.:
#   B=64 OMP_THREADS=16 CUDA_HOME=/opt/cuda bash profile_nsys.sh

set -e
source params.sh

# --- knobs -----------------------------------------------------------------
B="${B:-16}"                         # fixed batch size for the profiling run
OMP_THREADS="${OMP_THREADS:-8}"      # threads for the OpenMP capture
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

PIC="data/prof/pic_b${B}.cpic"
MODEL="models/vit_1.cvit"
MEAS="reports/_measures_scratch.csv"  # required 4th arg of vit.exe; unused here

# --- prerequisites ---------------------------------------------------------
if ! command -v nsys >/dev/null 2>&1; then
    echo "Error: nsys (Nsight Systems) not found on PATH."
    echo "       On the cluster, load it first (e.g. 'module load cuda' or 'module load nsight-systems')."
    exit 1
fi

mkdir -p prof_obj omp_prof_obj prof_bin data/prof reports

# --- 1. build the NVTX-instrumented executables ----------------------------
make prof CUDA_HOME="$CUDA_HOME"
echo "prof executables built"

# --- 2. model --------------------------------------------------------------
if [ ! -f "$MODEL" ]; then
    echo "Model $MODEL missing -> running create_models.sh"
    bash create_models.sh
fi

# --- 3. fixed-size input (min_b == max_b == B for a deterministic batch) ----
if [ ! -f "$PIC" ]; then
    python3 scripts/random_cpic.py "$PIC" "$B" "$B" \
        "$DTASET_C" "$DTASET_H" "$DTASET_W" "$DTASET_MIN_VAL" "$DTASET_MAX_VAL"
    echo "created fixed input $PIC (batch size $B)"
fi

# --- 4. capture ------------------------------------------------------------
NSYS_ARGS="--trace=nvtx,osrt --sample=none --force-overwrite=true"

echo
echo "=== serial capture -> reports/cpp_b${B} ==="
nsys profile $NSYS_ARGS -o "reports/cpp_b${B}" \
    ./prof_bin/vit_prof.exe "$MODEL" "$PIC" "reports/cpp_prd_b${B}.cprd" "$MEAS"

echo
echo "=== OpenMP capture (${OMP_THREADS} threads) -> reports/omp_b${B} ==="
OMP_NUM_THREADS="$OMP_THREADS" OMP_PROC_BIND=close OMP_PLACES=cores \
nsys profile $NSYS_ARGS -o "reports/omp_b${B}" \
    ./prof_bin/vit_omp_prof.exe "$MODEL" "$PIC" "reports/omp_prd_b${B}.cprd" "$MEAS"

# --- 5. NVTX region time summary (sorted table per report) -----------------
echo
echo "=== NVTX region summary: SERIAL (reports/cpp_b${B}) ==="
nsys stats --report nvtx_pushpop_sum --format table "reports/cpp_b${B}.nsys-rep" || true
echo
echo "=== NVTX region summary: OPENMP (reports/omp_b${B}) ==="
nsys stats --report nvtx_pushpop_sum --format table "reports/omp_b${B}.nsys-rep" || true

echo
echo "Done. Open reports/cpp_b${B}.nsys-rep and reports/omp_b${B}.nsys-rep in the Nsight Systems GUI."
echo "The widest NVTX ranges (expect 'linear' and 'attention') are the first CUDA targets."
