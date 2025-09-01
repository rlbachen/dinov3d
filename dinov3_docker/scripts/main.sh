#!/usr/bin/env bash

DOWNLOAD=true
CONFIG=""
GPUS=0
OUTPUT="/output"

while getopts "c:g:o:d" opt; do
  case "$opt" in
    c)
      CONFIG="$OPTARG"
      ;;
    g)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        GPUS="$OPTARG"
      else
        echo "Error: -g expects a non-negative integer" >&2
        exit 1
      fi
      ;;
    d)
      DOWNLOAD=false
      ;;
    o)
      OUTPUT="$OPTARG"
      ;;
    \?)
      echo "Usage: $0 [-c config/path.yaml] [-g gpu_count] [-o /output/dir (default /output)] [-d] (skip download) [-b] (do s3 backup)" >&2
      exit 1
      ;;
  esac
done

# Defaults for non-SLURM system
RDZV_HOST="localhost"
RDZV_PORT=29500
RDZV_ID="local_run"

# # Check if SLURM
# if [[ -n "${SLURM_JOB_ID:-}" ]]; then
#     echo "Running on slurm"

#     # RDZV from jobid
#     RDZV_PORT=$((10000 + SLURM_JOB_ID % 50000))

#     if command -v scontrol &> /dev/null && [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
#         RDZV_HOST=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
#     else
#         RDZV_HOST=$(hostname)
#     fi

#     echo "Using RDZV_HOST=${RDZV_HOST}, RDZV_PORT=${RDZV_PORT}, RDZV_ID=${SLURM_JOB_ID}"
# fi

if [[ "$DOWNLOAD" == "true" ]]; then
  bash /app/scripts/util/copy_dataset.sh --out_dir /dataset/
fi

echo "Starting training with ${GPUS} GPU(s)"
echo "RDZV_HOST=${RDZV_HOST}, RDZV_PORT=${RDZV_PORT}, RDZV_ID=${RDZV_ID}"
echo "Saving in ${OUTPUT}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

/opt/conda/bin/torchrun \
  --max-restarts 4 \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=${GPUS} \
  --rdzv_id=${RDZV_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${RDZV_HOST}:${RDZV_PORT} \
  /app/dinov3/dinov3/train/train_3d.py \
  --output-dir "${OUTPUT}" \
  --config-file "${CONFIG}" \
  > "${OUTPUT}/all_log.txt" 2>&1

# USAGE: main.sh -c /path/to/config.yaml -g (num gpus as single number: 8) -d (pass if you want to skip download) -o /path/to/output