from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf


class Data:

    def __init__(self, data):
        self.data = data
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.split_data()

    def split_data(self):
        x = self.data.iloc[:, :-3]
        y = self.data.iloc[:, -2:]

        x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train_and_val, y_train_and_val, test_size=0.2,
                                                          random_state=42)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(x)

        x_train = normalizer(x_train)
        x_val = normalizer(x_val)
        x_test = normalizer(x_test)

        return x_train, x_val, x_test, y_train, y_val, y_test
