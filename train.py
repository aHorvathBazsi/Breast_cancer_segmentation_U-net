"""
Train.py is used to train the U-net model. Before the traing we have to specify a number of hiperparameters.
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
SAVING_NAME = "model_U_net.h5"
SAVING_PATH = '/tmp/Saved_models/'
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
LEARNING_RATE = 5e-3;

def plot_metrics(results):
"""
Using the metrics logged in results variable, the function plots accuracy and loss against number of epochs.
"""
	acc = results.history['accuracy']
	val_acc = results.history['val_accuracy']
	loss = results.history['loss']
	val_loss = results.history['val_loss']

	epochs = range(len(acc))

	plt.figure(figsize=(10,6))
	plt.plot(epochs, acc, 'r', label = 'Training accuracy')
	plt.plot(epochs, val_acc, 'b', label ='Validation accuracy')
	plt.title('Training and validation accuracy')
	plt.legend()

	plt.figure(figsize=(10,6))
	plt.plot(epochs,loss,'r',label = 'Training loss')
	plt.plot(epochs,val_loss,'b',label='Validation_loss')
	plt.axis([0,100,0,1])
	plt.title('Training and validation loss')
	plt.legend()

	plt.show()

def save_model(model,saving_name, saving_path):

	path = os.path.join(saving_path,saving_name)
	model.save(path)

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

	train_generator = data_info["train_generator"] 
	validation_generator = data_info["validation_generator"]
	test_images = data_info["test_images"]
	test_masks = data_indo["test_labels"]

	results = model.fit_generator(train_generator,steps_per_epoch=7,  validation_data=val_generator, validation_steps=2, epochs=100)

	save_model(model,SAVING_NAME,SAVING_PATH)

	plot_metrics(results)

if __name__ == "__main__":
    main()