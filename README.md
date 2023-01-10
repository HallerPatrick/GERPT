# GERPT - Training a German Generative Transformer Model using N-Gram Multihot Encodings

Experiments for my thesis 🤗


## Setup


Install necessary dependencies:

```
pip install -r requirements.txt
```

To run the training on GPUs please install `pytorch` with CUDA support.


The following tasks can all be run with: `tools/run_all.sh`

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




## Encoding benchmarks

### Pack/Unpack Function (1 000 000 Iterations)

+---------------+----------------+------------+
|    Function   | Implementation | Time (sec) |
+---------------+----------------+------------+
|      pack     |     Python     |    0.58    |
|     unpack    |     Python     |    0.52    |
|      pack     |      C++       |    0.37    |
|     unpack    |      C++       |    0.35    |
|  pack (fast)  |      C++       |    0.36    |
| unpack (fast) |      C++       |    0.22    |
+---------------+----------------+------------+

### Pack Tensor Function (1 000 000 Iterations)
+-------------+----------------+------------+
|   Function  | Implementation | Time (sec) |
+-------------+----------------+------------+
| pack_tensor |     Python     |    27.4    |
| pack_tensor |      C++       |    2.86    |
+-------------+----------------+------------+

### Unpack Tensor Function (1 0000 000 Iterations)

Note: Higher numbers more diverges

+---------------+----------------+------------+
|    Function   | Implementation | Time (sec) |
+---------------+----------------+------------+
| unpack_tensor |     Python     |    8.21    |
| unpack_tensor |      C++       |    2.05    |
+---------------+----------------+------------+

### n_hot unpacked Function (1 000 000 Iterations)

+----------+----------------+------------+
| Function | Implementation | Time (sec) |
+----------+----------------+------------+
|  n_hot   |     Python     |    2.65    |
|  n_hot   |      C++       |    1.8     |
+----------+----------------+------------+

### n_hot packed function (1 000 000 Iterations)

+----------+----------------+------------+
| Function | Implementation | Time (sec) |
+----------+----------------+------------+
|  n_hot   |     Python     |   22.15    |
|  n_hot   |      C++       |    5.92    |
+----------+----------------+------------+

### Embedding Layer function (1 000 000 Iterations)

+--------------------+----------------+------------+
|      Function      | Implementation | Time (sec) |
+--------------------+----------------+------------+
| Embedding Unpacked |     Python     |   44.66    |
|  Embedding Packed  |     Python     |   520.91   |
| Embedding Unpacked |      C++       |   43.62    |
|  Embedding Packed  |      C++       |   42.72    |
+--------------------+----------------+------------+



### Embedding Layer (1 Iteration)

| Type | implementation | Device  | Time  |
|---|---|---|---|---|
| Packed  | Python | cpu | 1.51ms  |
| Packed  | Python | cuda | 3.59ms  | 
| Unpacked  | Python | cpu | 127.35 us  |
| Unpacked  | Python | cuda | 167.61 us  |




