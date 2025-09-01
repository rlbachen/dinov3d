#!/bin/bash

# mkdir -p /workspace/data
# mkdir -p /workspace/output
mkdir -p /workspace/data
mkdir -p /workspace/output


/opt/conda/bin/python /app/dinov3/scripts/download_dataset.py --out_dir /workspace/data --mini true
