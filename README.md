# Breast cancer segmentation using U-net

## This repository contains the followings: implementation of U-net architecture, training of the model and prediction on test images. Please note that the dataset is not provided because it is private.

The repository contains the following files:

- train.py: main function used to train the U-net model
- dataset.py: preprocessing (+augmentatioon) of original images and creation of DataLoaders (the path for training and validation sets can be specified for custom use)
- model.py: implementation of U-net using Tensorflow framework
- predict.py: prediction for test images (path for test images can be specified for custom use)

### Please note that the code is not fully functional yet. Each individual .py is correct, but I want to connect them in a more efficient way. If you have any suggestions please do not hesitate to contact me or clone the repo and try to find ways to upgrade the code.
