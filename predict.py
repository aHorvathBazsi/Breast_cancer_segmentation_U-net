"""
The following script is used for segmentation. Actually we load the trained parameters of the model and use them to make a segemtation.
We use the same create_U_net and create_dataset function (similar to train.py)
"""

import os
import random
import numpy as np
import tensorflow as tf
import cv2 as cv
 
from tqdm import tqdm 

from skimage.io import imread, imshow
from skimage.transform import resize
from keras.preprocessing import image
import matplotlib.pyplot as plt

from model import create_U_net
from dataset import create_dataset

RAND_SEED= 42
TRAIN_IMAGES = '/tmp/Train_original/'
TRAIN_MASK = '/tmp/Train_mask/'
TEST_IMAGES = '/tmp/Test_original/'
TEST_MASK = '/tmp/Test_mask/'
LOAD_MODEL_FILE = "/tmp/Saved_models/model_U_net.h5"
INPUT_WIDTH = 256
INPUT_HEIGHT = 256
IMG_CHANNEL = 3
SHEAR = 0.1
ROTATION = 0.15
ZOOM = 0.1
WIDTH_SHIFT = 0.1
HEIGHT_SHIFT = 0.1
SPLIT_RATIO = 0.8
BATCH_SIZE = 16
THRESHOLD = 0.5
LEARNING_RATE = 5e-3

def main():

	model = create_U_net(INPUT_WIDTH,INPUT_HEIGHT,IMG_CHANNEL,LEARNING_RATE)

	data_info = create_dataset(
		train_image_path = TRAIN_IMAGES,
		train_mask_path = TRAIN_MASK,
		test_image_path = TEST_IMAGES,
		test_mask_path = TEST_MASK,
		img_height = INPUT_HEIGHT,
		img_width = INPUT_WIDTH,
		img_channels = IMG_CHANNEL,
		shear = SHEAR,
		rotation = ROTATION,
		zoom = ZOOM,
		width_shift = WIDTH_SHIFT,
		height_shift = HEIGHT_SHIFT,
		split_ratio = SPLIT_RATIO,
		batch_size = BATCH_SIZE,
		rand_seed = RAND_SEED)

	test_images = data_info["test_images"]
	test_masks = data_indo["test_labels"]

	model = tf.keras.models.load_model("LOAD_MODEL_FILE")

	# Use the model to make prediction (please note that we use a threshold of 0.5, which means that if the model predicts that one pixel
	# is part of the class of interest with a probability higher than 50% that that pixel will be set to 1)
	predictions = model.predict(test_images,verbose=1)
	predictions = (predictions > THRESHOLD).astype(np.uint8)

	#plot a random prediction
	idx = random.randint(0, len(predictions)-1)
	imshow(test_images[idx])
	plt.show()
	imshow(np.squeeze(test_masks[idx]))
	plt.show()
	imshow(np.squeeze(predictions[idx]))
	plt.show()

