# Activate the relevant virtual environment:
source ./venv/bin/activate

# Generate:

python generate.py \
    --model-path checkpoints/model.pt.last.ckpt \
    --mode gen \
    --num-iters 1 \
    --num-chars 1000 \

