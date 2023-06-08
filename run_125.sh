#!/bin/bash

# MODEL_PATH=/tmp/test-clm
MODEL_PATH=/glusterfs/dfs-gfs-dist/hallepat/gerpt_neox

rm -rf $MODEL_PATH

RUN_NAME=babylm_small_10Epochs_125M
export WANDB_PROJECT=gerpt-neox
export CUDA_VISIBLE_DEVICES=0

echo "Training model"
python train_hf.py \
    --dataset_name babylm10M \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --do_train \
    --do_eval \
    --output_dir $MODEL_PATH \
    --overwrite_output_dir \
    --num_train_epochs 10 \
    --logging_steps 1000 \
    --run_name $RUN_NAME \
    --hidden_size 1024 \
    --num_hidden_layers 4 \
    --num_attention_heads 4 \
    --intermediate_size 1024 \
    --block_size 512
