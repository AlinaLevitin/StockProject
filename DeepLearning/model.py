"""
Model class to generate a neural network
This class allows training, saving, loading and using the model to predict results
"""

import keras
import tensorflow as tf
import pandas as pd
import os

import DeepLearning


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
        will save the weights every epoch using the callbacks function
    load_callback(self)
        will load the latest callback
    save_model(self, name: str)
        will save a trained model
    load_model(self, name: str)
        will load a trained model
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

    def get_neurons(self):
        return self.neurons

    def get_epochs(self):
        return self.epochs

    def get_learning_rate(self):
        return self.learning_rate

    def get_batch_size(self):
        return self.batch_size

    def get_summary(self):
        self.model.summary()

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
            tf.keras.layers.Dense(self.neurons, activation='elu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dense(self.neurons, activation='elu', kernel_regularizer='l2'),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dense(self.neurons, activation='elu', kernel_regularizer='l2'),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dropout(.2),
            tf.keras.layers.Dense(self.neurons, activation='elu'),
            tf.keras.layers.Dense(self.neurons, activation='elu', kernel_regularizer='l2'),
            tf.keras.layers.Dense(4, activation='sigmoid')]
            )

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
        return self.model

    def train_and_test(self, data,  save: bool = True):
        """
        train and test the model

        :param data: make sure to split the data using TrainingData class method split_data
        :type data: TrainingData
        :param save: optional to save callbacks of the neural network every epoch, will create new folder in case its missing

        :return: accuracy of testing data, and history pandas dataframe
        """

        self.data = data

        callbacks = None

        if save:
            cp_callback = tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                                             save_weights_only=True, verbose=1, save_freq='epoch')
            callbacks = [cp_callback]

        history = self.model.fit(self.data.x_train, self.data.y_train,
                       epochs=self.epochs,
                       validation_data=(self.data.x_val, self.data.y_val),
                       batch_size=self.batch_size, callbacks=callbacks,
                       verbose=1
                       )
        epochs = [i for i in range(1, self.epochs+1)]
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        acc_and_loss = {'epoch': epochs, 'accuracy': acc, 'loss': loss, 'validation_accuracy' : val_acc, 'validation_loss': val_loss}
        acc_and_loss_df = pd.DataFrame.from_dict(acc_and_loss)

        result = self.model.evaluate(self.data.x_test, self.data.y_test)
        accuracy = result[1]
        return accuracy, acc_and_loss_df

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
        os.makedirs(self.cwd + '\\trained_model', exist_ok=True)
        os.chdir(self.cwd + '\\trained_model')
        self.model.save(name + '.h5')
        print(f'model was saved to file: {name}.h5')

    def load_model(self, name: str):
        """
        loads the latest saved model

        :param name: name of the file
        """
        os.makedirs(self.cwd + '\\trained_model', exist_ok=True)
        os.chdir(self.cwd + '\\trained_model')
        print('-' * 60)
        try:
            self.model = keras.models.load_model(name + '.h5')
        except (Exception,):
            raise FileNotFoundError('Unable to load trained neural network,'
                                    f'please make sure that {name}.h5 is in the folder "trained_model"')
        self.model.summary()
        loaded_model_neurons = self.model.get_layer('dense').output_shape[1]
        if not self.neurons or self.neurons == loaded_model_neurons:
            self.neurons = loaded_model_neurons
        else:
            raise ValueError("number of neurons in the load_model and set_model don't match,"
                  "if you want to change the number of neurons please delete old trained_model.h5 file")
        print(f'Loading model from file: {name}.h5')
        print('-' * 60)

    def repeat_train(self, data, repeats, neurons, epochs, learning_rate, batch_size):
        """
        trains the model chosen number of times for hyperparameters optimization

        :param data: TrainingData, make sure to split the data and reduce data volume or else it will take a long time
        :param repeats: number of training repeats
        :param neurons: the number of neurons in the model
        :param epochs: the number of epochs for training
        :param learning_rate: the learning rate for stochastic gradient descent
        :param batch_size: the batch size for stochastic gradient descent
        :return: summary.csv and acc_and_loss.csv files in "model_opt" folder
        """
        model = self.set_model(data, neurons, epochs, learning_rate, batch_size)
        self.model.summary()
        os.makedirs(self.cwd + "\\model_opt", exist_ok=True)
        os.chdir(self.cwd + "\\model_opt")
        self.model.save_weights('model.h5')
        acc = []
        for i in range(repeats):
            print('-'*60)
            print(f'Repeat #{i+1} out of {repeats}')
            print(self)
            print('-' * 60)
            os.chdir(self.cwd + "\\model_opt")
            self.model.load_weights('model.h5')
            result = self.train_and_test(self.data, save=False)
            accuracy = result[0]
            acc_and_loss = result[1]
            acc.append(accuracy)
            report = DeepLearning.Reports(self.cwd, self.data, self.model)
            report.repeat_train_report(i, accuracy, acc_and_loss)
        report = DeepLearning.Reports(self.cwd, self.data, self.model)
        path = f"\\model_opt\\reports\\neurons_{self.neurons}"
        summary = report.repeat_train_summary(accuracy=acc, path=path)
        return summary

    def predict_values(self, data):
        """
        predicting results

        :param data: PredictData
        :return: numpy array of the predicted results
        """
        data = data.pandas_to_numpy()
        return self.model.predict(data)
