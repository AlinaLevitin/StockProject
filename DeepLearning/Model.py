"""
Model class to generate a neural network
This class allows training, saving, loading, saving and using the model
"""
# TODO: finish the documentation of Model class

import json
import shutil

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
    set_model(self, data, neurons: int, epochs: int, learning_rate: float, batch_size: int)
        initializing the model with the chosen hyperparameters
        returns TensorFlow sequential model
    train_and_test(self, data,  save: bool = False)
        class method to test and train the model according to the chosen hyperparameters in set_model
        returns accuracy of test
        will save the weights every 5 epochs using the callbacks function
    load_callback(self)
        will load the latest callback
    save_model(self, name: str)
        will save a trained model
    load_model(self, name: str)
        will load a trained model
    repeat_train(self, repeats, neurons, epochs, learning_rate, batch_size)
        for optimization of hyperparameters, this will repeat the class method train_and_test
        and result a summery file
    summary(self, accuracy)
        generates a summary file
    predict_values(self, data)
        predicting outcomes according to the trained neural network
        returns a numpy array


    """
    def __init__(self, cwd: str):
        """

        :param cwd: a string of the working directory
        """

        self.cwd = cwd
        self.checkpoint_path = self.cwd + "\\callback\\cp.ckpt"
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

        input_shape = self.data.input_shape

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.neurons, activation='tanh', input_shape=(input_shape,)),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dense(self.neurons, activation='relu'),
            tf.keras.layers.Dense(self.neurons, activation='relu'),
            tf.keras.layers.Dense(4, activation='sigmoid')])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=['accuracy'])
        self.model.summary()
        print(self.model)
        return self.model

    def train_and_test(self, data,  save: bool = True):
        """

        :param data: make sure to split the data using TrainingData class method split_data
        :type data: TrainingData
        :param save: optional to save callbacks of the neural network
        :type save: bool optional
        :return: accuracy of testing data, and a summary file (.csv)
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
            self.checkpoint_path = self.cwd + "\\callback\\cp.ckpt"
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
        print('Loading latest callback')

    def delete_callbacks(self):
        folder = self.cwd + "\\callback"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    def save_model(self, name: str):
        os.chdir(self.cwd)
        self.model.save(name + '.h5')
        params = {'neurons': self.neurons, 'epochs': self.epochs, 'learning_rate': self.learning_rate,
                  'batch_size': self.batch_size}
        json_save = json.dumps(params)
        with open(f'params_{name}.json', 'w') as json_file:
            json_file.write(json_save)
        print(f'model was saved to file: {name}.h5')

    def load_model(self, name: str):
        print(f'Loading model from file: {name}.h5')
        os.chdir(self.cwd)
        self.model = keras.models.load_model(name + '.h5')
        with open(f'params_{name}.json') as json_file:
            params = json.load(json_file)
        if not self.neurons:
            self.neurons = params['neurons']
        if not self.epochs:
            self.epochs = params['epochs']
        if not self.learning_rate:
            self.learning_rate = params['learning_rate']
        if not self.batch_size:
            self.batch_size = params['batch_size']

    def repeat_train(self, data, repeats, neurons, epochs, learning_rate, batch_size):
        self.set_model(data, neurons, epochs, learning_rate, batch_size)
        acc = []
        for i in range(repeats):
            result = self.train_and_test(self.data)
            acc.append(result)

        self.summary(accuracy=acc)

    def summary(self, accuracy):

        if isinstance(accuracy, list):
            repeats = len(accuracy)
        else:
            repeats = 1

        accuracy_average = utils.average(accuracy)
        stddev = utils.stddev(accuracy)

        summary_dict = {'average accuracy': accuracy_average,
                        'STDEV': stddev,
                        'repeats': repeats,
                        'training data': self.data.train_num,
                        'validation data': self.data.val_num,
                        'test data': self.data.test_num,
                        'steps back': self.data.steps_back,
                        'steps forward': self.data.steps_forward,
                        'batch_size': self.batch_size,
                        'epochs': self.epochs,
                        'neurons': self.neurons,
                        'learning_rate': self.learning_rate,
                        }

        summary = pd.DataFrame([summary_dict])
        utils.save_to_csv(f'summary_for_{repeats}_repeats.csv', summary, self.cwd)
        print(summary)
        print(f'Saved summary to file: summary_for_{repeats}_repeats.csv')

    def predict_values(self, data):
        data = data.pandas_to_numpy()
        return self.model.predict(data)




