# GERPT - Training a German Generative Transformer Model using N-Gram Multihot Encodings

Experiments for my thesis 🤗


## Setup


Install necessary dependencies:

```
pip install -r requirements.txt
```

To run the training on GPUs please install `pytorch` for CUDA support.


## Pre-Training

### Pre-Process




### Training

The training script will either train a standard implementation of a LSTM or Transformer model,
with the N-Gram Multihot approach.

All parameters can be defined in a yaml configuration file. See `configs/base.yaml` for possible
options or run `python train.py --help`.

```
python train.py --config configs/base.yaml
```


## Downstream Evaluation

