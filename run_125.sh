#!/bin/bash

MODEL_PATH=/tmp/test-clm

rm -rf $MODEL_PATH

RUN_NAME=babylm_small_10Epochs_125M
export WANDB_PROJECT=gerpt-neox
export CUDA_VISIBLE_DEVICES=0

echo "Training model"
python train_hf.py \
    --dataset_name babylm10M \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir $MODEL_PATH \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --logging_steps 1000 \
    --run_name $RUN_NAME \
    --hidden_size 768 \
    --num_hidden_layers 12 \
    --num_attention_heads 12 \
    --intermediate_size 3072 \


