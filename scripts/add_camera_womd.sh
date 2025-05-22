#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=testing # training, validation, testing

python \
  -m src.data_preprocess \
  --mode add_camera \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --pickle_dir /workspace/scratch/cache/SMART/$DATA_SPLIT \
  --camera_data_dir /workspace/scratch/data/womd/uncompressed/lidar_and_camera/$DATA_SPLIT \
  --output_dir /workspace/scratch/cache/SMART_with_camera