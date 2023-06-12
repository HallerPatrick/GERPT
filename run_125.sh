#!/bin/bash

MODEL_PATH=/bigone/models/gerpt_neox_50M
# MODEL_PATH=/glusterfs/dfs-gfs-dist/hallepat/gerpt_neox

rm -rf $MODEL_PATH

export WANDB_PROJECT=gerpt-neox
export CUDA_VISIBLE_DEVICES=0


MODEL_SIZE="small"
EPOCHS=10
EXTRA=""

if [[ "$EXTRA" = "" ]]; then
    RUN_NAME="gerpt_${MODEL_SIZE}_${EPOCHS}_epochs"
else
    RUN_NAME="gerpt_${MODEL_SIZE}_${EPOCHS}_epochs_${EXTRA}"
fi

echo $RUN_NAME


if [[ "$MODEL_SIZE" = "large" ]]; then
    # 175M with 3-gram vocab
    HIDDEN_SIZE=1024
    NUM_HIDDEN_LAYERS=16
    NUM_ATTENTION_HEADS=16
    INTERMEDIATE_SIZE=
elif [[ "$MODEL_SIZE" = "medium" ]]; then
    # 90M Params with 3-gram vocab
    HIDDEN_SIZE=768
    NUM_HIDDEN_LAYERS=12
    NUM_ATTENTION_HEADS=12
    INTERMEDIATE_SIZE=3072
else
    # 57M Params with 3-gram vocab
    HIDDEN_SIZE=1024
    NUM_HIDDEN_LAYERS=8
    NUM_ATTENTION_HEADS=8
    INTERMEDIATE_SIZE=3072
fi

echo "Training model"

python train_hf.py \
    --dataset_name babylm10M \
    --per_device_train_batch_size 64 \
    --per_device_eval_batch_size 64 \
    --do_train \
    --do_eval \
    --output_dir $MODEL_PATH \
    --overwrite_output_dir \
    --num_train_epochs $EPOCHS \
    --logging_steps 1000 \
    --run_name $RUN_NAME \
    --hidden_size $HIDDEN_SIZE \
    --num_hidden_layers $NUM_HIDDEN_LAYERS \
    --num_attention_heads $NUM_ATTENTION_HEADS \
    --intermediate_size $INTERMEDIATE_SIZE \
    --block_size 512

