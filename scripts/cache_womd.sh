#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=testing # training, validation, testing

python \
  -m src.data_preprocess \
  --mode preprocess \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --input_dir /workspace/scratch/data/womd/uncompressed/scenario \
  --output_dir /workspace/scratch/cache/SMART