#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

DATA_SPLIT=testing # training, validation, testing

# Add debugging information using find instead of ls
echo "Checking pickle files in: /workspace/scratch/cache/SMART/$DATA_SPLIT"
find /workspace/scratch/cache/SMART/$DATA_SPLIT -name "*.pkl" | wc -l

echo "Checking camera files in: /workspace/scratch/data/womd/uncompressed/lidar_and_camera/$DATA_SPLIT"
find /workspace/scratch/data/womd/uncompressed/lidar_and_camera/$DATA_SPLIT -name "*.tfrecord" | wc -l

# Let's also check a few example files to verify the naming pattern
echo "Example pickle files:"
find /workspace/scratch/cache/SMART/$DATA_SPLIT -name "*.pkl" | head -n 3

echo "Example camera files:"
find /workspace/scratch/data/womd/uncompressed/lidar_and_camera/$DATA_SPLIT -name "*.tfrecord" | head -n 3

python \
  -m src.data_preprocess \
  --mode add_camera \
  --split $DATA_SPLIT \
  --num_workers 12 \
  --pickle_dir /workspace/scratch/cache/SMART/$DATA_SPLIT \
  --camera_data_dir /workspace/scratch/data/womd/uncompressed/lidar_and_camera/$DATA_SPLIT \
  --output_dir /workspace/scratch/cache/SMART_with_camera