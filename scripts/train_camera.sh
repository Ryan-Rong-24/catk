#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MY_EXPERIMENT="camera_aware_smart"
MY_TASK_NAME=$MY_EXPERIMENT"-finetune"

# Optionally activate your conda environment here
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate catk

torchrun \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  task_name=$MY_TASK_NAME \
  dataloader.data_dir=/scratch/cache/SMART_with_camera/training

echo "bash train_camera.sh done!"