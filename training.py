import os
import methods
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from DeepLearnig import train_data


cwd = os.getcwd()

csv = methods.open_csv(cwd, 'data.csv')

data= train_data.Data(csv)

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(150, activation='relu', input_shape=(data.x_train.shape[1],)),
  tf.keras.layers.Dense(150, activation='tanh'),
  tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data.x_train, data.y_train, epochs=100, validation_data=(data.x_val, data.y_val))





