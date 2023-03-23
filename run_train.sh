python -c "from src.models import GPTNeoXModel, GPTNeoXConfig; config = GPTNeoXConfig(); config.save_pretrained('gpt_neox');model = GPTNeoXModel(config); model.save_pretrained('gpt_neox');"

python test.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --output_dir /tmp/test-clm \
    # --config_overrides="n_embd=512,n_head=1,n_layer=1"

    # --do_eval \
    # --tokenizer_name gpt2 \
