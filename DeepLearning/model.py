"""
Model class to generate a neural network
This class allows training, saving, loading and using the model to predict results
"""

import time
import keras
import tensorflow as tf
import pandas as pd
import os
import utils
from DeepLearning import dl_utils


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
        the number of epochs for training
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
        return f"neurons: {self.neurons}, epochs: {self.epochs}," \
               f"learning rate: {self.learning_rate}, batch size: {self.batch_size} "

    def set_model(self, data, neurons: int, epochs: int, learning_rate: float, batch_size: int):
        """
        initializing the model

        :param data: desired training data, must be of TrainingData class
        :param neurons: the number of neurons in the model
        :param epochs: the number of epochs for training
        :param learning_rate: the learning rate for stochastic gradient descent
        :param batch_size: the batch size for stochastic gradient descent
        :return: TensorFlow sequential model
        """
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
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
        return self.model

    def train_and_test(self, data,  save: bool = True):
        """
        train and test the model

        :param data: make sure to split the data using TrainingData class method split_data
        :type data: TrainingData
        :param save: optional to save callbacks of the neural network, will create new folder in case its missing
        :type save: bool optional
        :return: accuracy of testing data, and a summary file (.csv)
        """

        self.data = data

        callbacks = None

        if save:
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
        """
        load the latest callback
        """
        self.model.summary()
        self.model.load_weights(self.checkpoint_path)
        print('Loading latest callback')

    def save_model(self, name: str):
        """
        saves the trained neural network as a .h5 file

        :param name: name of the file
        """
        os.chdir(self.cwd)
        self.model.save(name + '.h5')
        print(f'model was saved to file: {name}.h5')

    def load_model(self, name: str):
        """
        loads the latest saved model

        :param name: name of the file
        """
        os.chdir(self.cwd)
        self.model = keras.models.load_model(name + '.h5')
        self.model.summary()
        loaded_model_neurons = self.model.get_layer('dense').output_shape[1]
        if not self.neurons or self.neurons == loaded_model_neurons:
            self.neurons = loaded_model_neurons
        else:
            raise ValueError("number of neurons in the load_model and set_model don't match,"
                  "if you want to change the number of neurons please delete old trained_neural_network.h5 file")
        print(f'Loading model from file: {name}.h5')

    def repeat_train(self, data, repeats, neurons, epochs, learning_rate, batch_size):
        """
        trains the model chosen number of times for hyperparameters optimization

        :param data: TrainingData, make sure to split the data
        :param repeats: number of training repeats
        :param neurons: the number of neurons in the model
        :param epochs: the number of epochs for training
        :param learning_rate: the learning rate for stochastic gradient descent
        :param batch_size: the batch size for stochastic gradient descent
        :return: summary csv file
        """
        self.set_model(data, neurons, epochs, learning_rate, batch_size)
        acc = []
        for i in range(repeats):
            result = self.train_and_test(self.data)
            acc.append(result)

        self.summary(accuracy=acc)

    def summary(self, accuracy):
        """
        generated a summary csv file

        :param accuracy: float or list of accuracy after training
        """

        if isinstance(accuracy, list):
            repeats = len(accuracy)
        else:
            repeats = 1

        accuracy_average = dl_utils.average(accuracy)
        stddev = dl_utils.stddev(accuracy)

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
        gmt = time.gmtime()
        summary = pd.DataFrame([summary_dict])
        utils.save_to_csv(f'summary_{gmt[0]}_{gmt[1]}_{gmt[2]}_{gmt[3]}_{gmt[4]}', summary,
                          self.cwd)
        print(summary)
        print(f'Saved summary to file: summary_{gmt[0]}_{gmt[1]}_{gmt[2]}_{gmt[3]}_{gmt[4]}.csv')

    def predict_values(self, data):
        """
        predicting results

        :param data: PredictData
        :return: numpy array of the predicted results
        """
        data = data.pandas_to_numpy()
        return self.model.predict(data)
