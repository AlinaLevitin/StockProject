from __future__ import absolute_import, division, print_function

import keras
import tensorflow as tf


class DLModel:

    def __init__(self, cwd):
        self.cwd = cwd
        self.checkpoint_path = self.cwd + "\\training\\cp.ckpt"
        self.data = None
        self.neurons = None
        self.epochs = None
        self.learning_rate = None
        self.batch_size = None
        self.model = None

    def __repr__(self):
        return f"neurons: {self.neurons}, epochs: {self.epochs}, learning rate: {self.learning_rate}, batch size: {self.batch_size}"

    def train_and_test(self, data, neurons, epochs, learning_rate, batch_size, save=False):
        self.data = data
        self.neurons = neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.neurons, activation='tanh', input_shape=(self.data.x_train.shape[1],)),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dense(self.neurons, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        callbacks = None

        if save:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path, save_weights_only=True, verbose=0, period=5)
            callbacks = [cp_callback]

        self.model.fit(self.data.x_train, self.data.y_train,
                       epochs=self.epochs,
                       validation_data=(self.data.x_val, self.data.y_val),
                       batch_size=self.batch_size, callbacks=callbacks,
                       verbose=0
                       )

        result = self.model.evaluate(self.data.x_test, self.data.y_test)
        accuracy = result[1]

        return accuracy

    def load_callback(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_path)
        self.model.load_weights(latest)

    def save_model(self, name) -> str:
        self.model.save(name + '.h5')

    def load_model(self, name):
        self.model = keras.models.load_model(name + '.h5')




