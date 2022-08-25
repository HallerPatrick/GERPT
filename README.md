# GERPT - Training a German Generative Transformer Model using N-Gram Multihot Encodings

Experiments for my thesis 🤗


## Setup


Install necessary dependencies:

```
pip install -r requirements.txt
```

To run the training on GPUs please install `pytorch` with CUDA support.


## Pre-Training

### Pre-Process

The preprocess script sets the vocabulary and the tokenized dataset up.
The easiest way is to use the training config, with the configs `data` for the dataset, `saved_dict` and 
`saved_data` for the outfile of the dictionary and tokenized dataset respectively.

*NOTE:* The `data` setting can be a huggingface dataset set or a local one that is prefixed with `"text/"`

```
python preprocess.py --config configs/base.yaml
```


### Training

The training script will either train a standard implementation of a LSTM or Transformer model,
with the N-Gram Multihot approach.

All parameters can be defined in a yaml configuration file. See `configs/base.yaml` for possible
options or run `python train.py --help`.

```
python train.py --config configs/base.yaml
```

Parameters can also be set through the command line and will overwrite the yaml configs.


## Downstream Evaluation

For downstream evaluation we use the `flair` library. In another yaml configuration file (see `configs/flair_base.yaml`) different downstream tasks can be declared. If the setting `use` is set to `True` training for the task is started. Multiple training tasks can be declared.

```
python train_ds.py --config configs/flair_base.yaml
```


## Troubleshooting

* Deepspeed tries to access some tmp folders for cuda extensions, that the user may not have permissions for. Export `TORCH_EXTENSIONS_DIR` to a new location.
