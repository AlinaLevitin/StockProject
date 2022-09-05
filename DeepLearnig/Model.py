"""
Model class to generate a neural network
This class allows training, saving, loading, saving and using the model
"""
# TODO: finish the documentation of Model class

import json
import keras
import tensorflow as tf
import pandas as pd
import os

import utils


class Model:
    """
        A class to generate a model for deep learning

    ...

    Attributes
    ----------
    cwd : str
        a string of the working directory
    neurons : int
        the number of neurons in the model
    epochs : int
        the number of epochs in the model
    learning_rate : float
        the learning rate for stochastic gradient descent
    batch_size : int
        the batch size for stochastic gradient descent
    data : pandas DataFrame
            should be TrainingData instance after split_data function
    model: TensorFlow/Keras sequential model

    Methods
    -------
    train_and_test(self, data, neurons, epochs, learning_rate, batch_size, save=False)
        class method to test and train the model according to the chosen hyper parameters
        returns accuracy of test
        will save the weights every 5 epochs using the callbacks function
    load_callback(self)
        will load the latest callback
    save_model(self, name: str)
        will save a trained model
    load_model(self, name: str)
        will load a trained model
    repeat_train(self, repeats, neurons, epochs, learning_rate, batch_size)
        a class method for optimization of hyper parameters, this will repeat the class method train_and_test
        and result a summery file
    summary(self, accuracy)
        a class method to generate a summary file

    """
    def __init__(self, cwd: str):
        """

        :param cwd: a string of the working directory
        """

        self.cwd = cwd
        self.checkpoint_path = self.cwd + "\\training\\cp.ckpt"
        self.data = None
        self.neurons = None
        self.epochs = None
        self.learning_rate = None
        self.batch_size = None
        self.model = None

    def __repr__(self):
        """

        :return: number of neurons, number of epochs learning rate and batch size
        """
        return f"neurons: {self.neurons}, epochs: {self.epochs}, learning rate: {self.learning_rate}, batch size: {self.batch_size}"

    def set_model(self, data, neurons: int, epochs: int, learning_rate: float, batch_size: int):
        self. data = data
        self.neurons = neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.neurons, activation='tanh', input_shape=(self.data.x_train.shape[1],)),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dense(self.neurons, activation='relu'),
            tf.keras.layers.Dense(self.neurons, activation='relu'),
            tf.keras.layers.Dense(self.data.y_train.shape[1], activation='softmax')])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()
        return self.model

    def train_and_test(self, data,  save: bool = False):
        """

        :param data: make sure to split the data using TrainingData class method split_data
        :type data: TrainingData
        :param save:
        :return:
        """
        """
        class method to test and train the model according to the chosen hyper parameters
        returns accuracy of test
        will save the weights every 5 epochs using the callbacks function

        :param data: TrainingData
            make sure to split the data using TrainingData class method split_data
        :param neurons: int
            the number of neurons in the model
        :param epochs: int
            the number of epochs in the model
        :param learning_rate: float
             the learning rate for stochastic gradient descent
        :param batch_size: int
            the batch size for stochastic gradient descent
        :param save: boolean optional
            set as True in to save every 5 epochs
        :return:
            accuracy of test

        """
        self.data = data

        callbacks = None

        if save:
            self.checkpoint_path = self.cwd + "\\training\\cp.ckpt"
            cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                             save_weights_only=True, verbose=1, save_freq='epoch')
            callbacks = [cp_callback]

        self.model.fit(self.data.x_train, self.data.y_train,
                       epochs=self.epochs,
                       validation_data=(self.data.x_val, self.data.y_val),
                       batch_size=self.batch_size, callbacks=callbacks,
                       verbose=1
                       )

        result = self.model.evaluate(self.data.x_test, self.data.y_test)
        accuracy = result[1]
        self.summary(accuracy)
        return accuracy

    def load_callback(self):
        self.model.load_weights(self.checkpoint_path)

    def save_model(self, name: str):
        os.chdir(self.cwd)
        self.model.save(name + '.h5')
        params = {'neurons': self.neurons, 'epochs': self.epochs, 'learning_rate': self.learning_rate,
                  'batch_size': self.batch_size}
        json_save = json.dumps(params)
        with open(f'params_{name}.json', 'w') as json_file:
            json_file.write(json_save)

    def load_model(self, name: str):
        os.chdir(self.cwd)
        self.model = keras.models.load_model(name + '.h5')
        with open(f'params_{name}.json') as json_file:
            params = json.load(json_file)
        self.neurons = params['neurons']
        self.epochs = params['epochs']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        print(self)

    def repeat_train(self, data, repeats, neurons, epochs, learning_rate, batch_size):
        self.set_model(data, neurons, epochs, learning_rate, batch_size)
        acc = []
        for i in range(repeats):
            result = self.train_and_test(self.data)
            acc.append(result)

        self.summary(accuracy=acc)

    def summary(self, accuracy):

        accuracy_average = utils.average(accuracy)
        stddev = utils.stddev(accuracy)

        summary_dict = {'average accuracy': accuracy_average,
                        'STDEV': stddev,
                        'repeats': len(accuracy),
                        'training data': self.data.train_num,
                        'validation data': self.data.val_num,
                        'test data': self.data.test_num,
                        'steps back': self.data.steps_back,
                        'steps forward': self.data.steps_forward,
                        'batch_size': self.model.batch_size,
                        'epochs': self.model.epochs,
                        'neurons': self.model.neurons,
                        'learning_rate': self.model.learning_rate,
                        }

        summary = pd.DataFrame([summary_dict])
        utils.save_to_csv(f'summary_for_{len(accuracy)}_repeats.csv', summary, self.cwd)

    def predict_values(self, x):
        x_numpy = x.to_numpy(copy=True)
        result = self.model.predict(x_numpy)
        return result




