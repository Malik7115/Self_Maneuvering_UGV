import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Sequential
import numpy as np

IMG_HEIGHT = 720
IMG_WIDTH = 1280
CHANNELS = 1

# nvidia_model = Sequential(
#     [
#     Conv2D(3, (5,5), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,CHANNELS)),
#     Conv2D(24, (5,5), activation='relu'),
#     Conv2D(36, (5,5), activation='relu'),
#     Conv2D(48, (3,3), activation='relu'),
#     Conv2D(64, (5,5), activation='relu'),
#     layers.Flatten(),
#     layers.Dense(100, activation='relu'),
#     layers.Dense(10, activation='relu'),
#     layers.Dense(1)
#     ]
# )

# custom_model_1 = Sequential(
#     [
#         Conv2D(16,(3,3), activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT,CHANNELS)),
#         MaxPool2D(pool_size=(2,2)),
#         Conv2D(32,(3,3), activation='relu'),
#         MaxPool2D(pool_size=(2,2)),
#         layers.Flatten(),
#         layers.Dropout(0.2),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.2),
#         layers.Dense(1)

#     ]
# ) 
custom_model_2 = models.Sequential()
custom_model_2.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT+2, IMG_WIDTH+2, 1)))
custom_model_2.add(layers.MaxPooling2D((2, 2)))
custom_model_2.add(layers.Conv2D(64, (3, 3), activation='relu'))
custom_model_2.add(layers.MaxPooling2D((2, 2)))
custom_model_2.add(layers.Conv2D(64, (3, 3), activation='relu'))
