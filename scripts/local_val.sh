#!/bin/sh
export LOGLEVEL=INFO
export HYDRA_FULL_ERROR=1
export TF_CPP_MIN_LOG_LEVEL=2

MY_EXPERIMENT="local_val"
VAL_K=48
MY_TASK_NAME=$MY_EXPERIMENT-K$VAL_K"-debug"

# local_val runs on single GPU
python \
  -m src.run \
  experiment=$MY_EXPERIMENT \
  trainer=default \
  model.model_config.validation_rollout_sampling.num_k=$VAL_K \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  task_name=$MY_TASK_NAME \
  ckpt_path=logs/clsft_E9.ckpt \
  ++data.val_raw_dir=/workspace/scratch/cache/SMART/validation \
  ++data.val_tfrecords_splitted=/workspace/scratch/cache/SMART/validation_tfrecords_splitted

echo "bash local_val.sh done!"