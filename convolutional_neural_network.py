import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time
import pickle
import numpy as np

Name = "street_view_{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(Name))

gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

X = np.load('features.npy')
y = np.load('label.npy')

model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
'''activation layer'''
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))

'''output layer'''
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = "binary_crossentropy",
                optimizer = "adam",
                metrics = ['accuracy'])
model.fit(X,y, batch_size = 32, epochs=10, validation_split = 0.3,callbacks = [tensorboard])