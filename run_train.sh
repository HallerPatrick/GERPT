#!/bin/bash

MODEL_PATH=/tmp/test-clm

rm -rf $MODEL_PATH

RUN_NAME=wikitext-2
export WANDB_PROJECT=gerpt-neox
export CUDA_VISIBLE_DEVICES=0

echo "Training model"
python train_hf.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir $MODEL_PATH \
    --overwrite_output_dir \
    --num_train_epochs 1 \
    --logging_steps 100 \
    --run_name $RUN_NAME

