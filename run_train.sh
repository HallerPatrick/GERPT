#!/bin/bash

# Run this script if argument is "train"
WANDB_DISABLED=true

rm -rf /tmp/test-clm

if [ "$1" = "train" ]; then
    echo "Training model"
    python train_hf.py \
        --dataset_name wikitext \
        --dataset_config_name wikitext-2-raw-v1 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --do_train \
        --do_eval \
        --output_dir /tmp/test-clm \
        --overwrite_output_dir \
        --num_train_epochs 100 \
        --logging_steps 4000 \


# Else run generation script
else
    echo "Generating text"
    python generate_hf.py /tmp/test-clm
fi
