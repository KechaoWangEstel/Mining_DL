# Mining_DL

## data prepare

data should be placed in folder `data/` as `.csv`format

use `data/preprocessing.py` to normalize the data and get the target data

## train the dataset

all the models are written in the `model.py`, the model we used is `ANN_relu_2`

run the command ``python train.py`` to train a model

all the models`*.pth`saved in fold`model/`

## validate and test the dataset

run the command ``python validation.py`` to do the validations

run the command ``python test.py`` to test a certain dataset
