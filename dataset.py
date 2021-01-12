import os
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow
from skimage.transform import resize
from keras.preprocessing import image

def create_dataset(
	train_image_path,
	train_mask_path,
	test_image_path,
	test_mask_path,
	img_height,
	img_width,
	img_channels,
	shear = 0.1,
	rotation = 0.15,
	zoom = 0.1,
	width_shift = 0.1,
	height_shift = 0.1,
	split_ratio = 0.8,
	batch_size = 16,
	rand_seed = 42):

	"""
	This function creates and returns data generators based on specified directories (folders of images and masks for train and test): the 
	specific path is given for each folder as parameter.

	To ensure that the data generator will return batches of images having the same dimension as the input of the model, it can be specified
	using img_width, img_height and img_channels parameters.

	The training images are split in 2 sets (training and validation sets). The split ratio can be specified using split_ratio parameter 
	(it is set by default at 0.8, meaning that training set contains 80% of the images while validation 20%)

	Other parameters refer to preprocessing and augmentation, such as: img_height, img_width, img_channels, shear_range, rotation_range,
	zoom_range, width_shift_range, height_shift_range.
	"""

	train_image_ids = np.sort(os.listdir(train_image_path))
	train_mask_ids = np.sort(os.listdir(train_mask_path))
	test_image_ids = np.sort(os.listdir(test_image_path))
	test_mask_ids = np.sort(os.listdir(test_mask_path))

	X_train = np.zeros((len(train_image_ids), img_height, img_width, img_channels), dtype=np.uint8)
	Y_train = np.zeros((len(train_mask_ids), img_height, img_width, 1), dtype=np.bool)
	X_test = np.zeros((len(test_image_ids),img_height,img_width,img_channels), dtype=np.uint8)
	Y_test = np.zeros((len(test_mask_ids), img_height, img_width, 1), dtype=np.bool)

	print('Resizing training images and masks')
	for i, id_ in tqdm(enumerate(train_image_ids), total=len(train_image_ids)):   
	    path_img = train_image_path + id_
	    img = imread(path_img)[:,:,:img_channels]  
	    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
	    X_train[i] = img  #Fill empty X_train with values from img

	for i, id_ in tqdm(enumerate(train_mask_ids), total = len(train_mask_ids)):
	    path_mask = train_mask_path + id_
	    mask = imread(path_mask)
	    if (len(mask.shape) ==3):
	      mask = mask[:,:,0]
	    mask = resize(mask, (img_height, img_width), mode='constant',preserve_range=True)
	    Y_train[i,:,:,0] = np.maximum(mask,Y_train[i,:,:,0])

	for i, id_ in tqdm(enumerate(test_image_ids), total=len(test_image_ids)):   
	    path_img = test_image_path + id_
	    img = imread(path_img)[:,:,:img_channels]  
	    img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
	    X_test[i] = img  #Fill empty X_train with values from img

	for i, id_ in tqdm(enumerate(test_mask_ids), total = len(test_mask_ids)):
	    path_mask = test_mask_path + id_
	    mask = imread(path_mask)
	    if (len(mask.shape) ==3):
	      mask = mask[:,:,0]
	    mask = resize(mask, (img_height, img_width), mode='constant',preserve_range=True)
	    Y_test[i,:,:,0] = np.maximum(mask,Y_test[i,:,:,0])

	print('Done!')



	# Creating the training Image and Mask generator
	image_datagen = image.ImageDataGenerator(shear_range=shear, rotation_range=rotation, zoom_range=zoom, width_shift_range=width_shift, height_shift_range=height_shift, fill_mode='nearest')
	mask_datagen = image.ImageDataGenerator(shear_range=shear, rotation_range=rotation, zoom_range=zoom, width_shift_range=width_shift, height_shift_range=height_shift, fill_mode='nearest')

	# Keep the same seed for image and mask generators so they fit together

	image_datagen.fit(X_train[:int(X_train.shape[0]*split_ratio)], augment=True, seed=rand_seed)
	mask_datagen.fit(Y_train[:int(Y_train.shape[0]*split_ratio)], augment=True, seed=rand_seed)

	x=image_datagen.flow(X_train[:int(X_train.shape[0]*split_ratio)],batch_size=batch_size,shuffle=True, seed=rand_seed)
	y=mask_datagen.flow(Y_train[:int(Y_train.shape[0]*split_ratio)],batch_size=batch_size,shuffle=True, seed=rand_seed)


	# Creating the validation Image and Mask generator
	image_datagen_val = image.ImageDataGenerator()
	mask_datagen_val = image.ImageDataGenerator()

	image_datagen_val.fit(X_train[int(X_train.shape[0]*split_ratio):], augment=True, seed=rand_seed)
	mask_datagen_val.fit(Y_train[int(Y_train.shape[0]*split_ratio):], augment=True, seed=rand_seed)

	x_val=image_datagen_val.flow(X_train[int(X_train.shape[0]*split_ratio):],batch_size=batch_size,shuffle=True, seed=rand_seed)
	y_val=mask_datagen_val.flow(Y_train[int(Y_train.shape[0]*split_ratio):],batch_size=batch_size,shuffle=True, seed=rand_seed)

	train_generator = zip(x, y)
	val_generator = zip(x_val, y_val)

	data_info = {}
	data_info["train_generator"] = train_generator
	data_info["validation_generator"] = val_generator
	data_info["test_images"] = X_test
	data_indo["test_labels"] = Y_test

	return data_info