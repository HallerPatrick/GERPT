

if [ "$VIRTUAL_ENV" == "" ]; then
  echo "WARNING: No virtualenv activated"
fi


echo "RUNNING PREPROCESSING"
python preprocess.py --config configs/base.yaml

echo "RUNNING TRAINING"
# Will also run downstream training if successful
python train.py --config configs/base.yaml

# Otherwise
# python train_ds.py --config configs/flair_base.yaml