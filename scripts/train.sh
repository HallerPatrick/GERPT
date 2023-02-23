# Activate the relevant virtual environment:
source ./venv/bin/activate

# Optional if not not prepare data:
python preprocess.py --config configs/base.yaml

# Train the model:
python train.py --config configs/base.yaml

