import tensorflow as tf

def create_U-net(input_width, input_height, img_channels):

	"""
	Create a U-net architecture having an input size of input_width x input_height x img_channels

	The U-net architecture has three different parts: 
		- contraction path:  consists of Conv2D layers followed by MaxPooling2D; 
		on this path the width and height of image is gradually reduced after each level (we have 4 downsampling levels)
		- bottleneck: the smallest dimensions are obtained here (we have a reduced width and height and a high number of filters);
		actually the width and height is reduced by a factor of 16 (4 MaxPooling layers) Please not that convolutional layers do not
		contribute to dimension reduction (because of the same padding)
		- expansive path: consists of ConvTranspose layers (which is used to upsample the dimensions of the data) and Conv2D layers. 
		Please note that the skip-connections are created using layers.concatenate; The last layer has just 1 filter and a 'sigmoid'
		activatation function (remember, we want to make segmentation);

	The model is compiled using Adam optimizer with learning rate of 5e-3 and binary_crossentropy loss

	"""

	inputs = tf.keras.layers.Input((img_width, img_height, img_channels))

	s = tf.keras.layers.Lambda(lambda x: x/255)(inputs)

	#Contraction path 
	conv1 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(s)
	conv1 = tf.keras.layers.Dropout(0.1) (conv1)
	conv1 = tf.keras.layers.Conv2D(16, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(conv1)
	pool1 = tf.keras.layers.MaxPooling2D((2,2)) (conv1)

	conv2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(pool1)
	conv2 = tf.keras.layers.Dropout(0.1) (conv2)
	conv2 = tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(conv2)
	pool2 = tf.keras.layers.MaxPooling2D((2,2)) (conv2)

	conv3 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(pool2)
	conv3 = tf.keras.layers.Dropout(0.2) (conv3)
	conv3 = tf.keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(conv3)
	pool3 = tf.keras.layers.MaxPooling2D((2,2)) (conv3)

	conv4 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(pool3)
	conv4 = tf.keras.layers.Dropout(0.2) (conv4)
	conv4 = tf.keras.layers.Conv2D(128, (3,3), activation = 'relu', kernel_initializer = 'he_normal', padding = 'same')(conv4)
	pool4 = tf.keras.layers.MaxPooling2D((2,2)) (conv4)

	#bottleneck 
	conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool4)
	conv5 = tf.keras.layers.Dropout(0.3)(conv5)
	conv5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv5)

	#Expansive path

	upsample6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv5)
	upsample6 = tf.keras.layers.concatenate([upsample6, conv4])
	conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(upsample6)
	conv6 = tf.keras.layers.Dropout(0.2)(conv6)
	conv6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv6)

	upsample7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv6)
	upsample7 = tf.keras.layers.concatenate([upsample7, conv3])
	conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(upsample7)
	conv7 = tf.keras.layers.Dropout(0.1)(conv7)
	conv7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv7)

	upsample8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv7)
	upsample8 = tf.keras.layers.concatenate([upsample8, conv2])
	conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(upsample8)
	conv8 = tf.keras.layers.Dropout(0.1)(conv8)
	conv8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv8)

	upsample9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv8)
	upsample9 = tf.keras.layers.concatenate([upsample9, conv1])
	conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(upsample9)
	conv9 = tf.keras.layers.Dropout(0.1)(conv9)
	conv9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv9)

	outputs = tf.keras.layers.Conv2D(1, (1,1), activation = 'sigmoid')(conv9)

	model = tf.keras.Model(inputs = [inputs], outputs = [outputs])
	optimizer = tf.keras.optimizers.Adam(learning_rate=5e-3)
	model.compile(optimizer = optimizer, loss='binary_crossentropy', metrics = ['accuracy'])

	return model