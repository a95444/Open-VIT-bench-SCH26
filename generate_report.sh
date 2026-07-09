#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Usage helper function
print_usage() {
    echo "Usage: $0 [options] ./your_executable [args...]"
    echo ""
    echo "Options:"
    echo "  --nsys    Run only Nsight Systems profile"
    echo "  --ncu     Run only Nsight Compute profile"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "If no profiling options are chosen, BOTH will run sequentially."
}

# Initialize toggle variables
RUN_NSYS=true
RUN_NCU=true

# Parse optional profiling flags
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --nsys)
            RUN_NSYS=true
            RUN_NCU=false
            shift
            ;;
        --ncu)
            RUN_NSYS=false
            RUN_NCU=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            # If it's not an option flag, we've hit the executable/arguments
            break
            ;;
    esac
done

# Check if an executable argument remains
if [ "$#" -eq 0 ]; then
    echo "Error: No executable provided."
    print_usage
    exit 1
fi

# Define the directories to store reports
report_dir_nsys="reports/nsys"
report_dir_ncu="reports/ncu"

# Generate a unique report file name based on the current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
report_file="report_${timestamp}"

# ----------------------------------------------------
# 1. Nsight Systems Execution
# ----------------------------------------------------
if [ "$RUN_NSYS" = true ]; then
    mkdir -p "$report_dir_nsys"
    echo "===================================================="
    echo "Running Nsight Systems (nsys) timeline profile..."
    echo "===================================================="
    
    nsys profile \
        --trace=cuda,nvtx,openacc \
        --stats=true \
        -o "${report_dir_nsys}/${report_file}" \
        "$@"
        
    echo "Nsight Systems report generated: ${report_dir_nsys}/${report_file}.nsys-rep"
    echo ""
fi

# ----------------------------------------------------
# 2. Nsight Compute Execution
# ----------------------------------------------------
if [ "$RUN_NCU" = true ]; then
    mkdir -p "$report_dir_ncu"
    echo "===================================================="
    echo "Running Nsight Compute (ncu) kernel profile..."
    echo "===================================================="
    
    ncu \
        --set detailed \
        -o "${report_dir_ncu}/${report_file}" \
        "$@"
        
    echo "Nsight Compute report generated: ${report_dir_ncu}/${report_file}.ncu-rep"
    echo "===================================================="
fi