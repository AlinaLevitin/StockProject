import tensorflow as tf
import math
import pandas as pd


class DLModel:

    def __init__(self, data, neurons, epochs, learning_rate, batch_size):
        self.data = data
        self.neurons = neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.neurons, activation='elu', input_shape=(self.data.x_train.shape[1],)),
            tf.keras.layers.Dense(self.neurons, activation='tanh'),
            tf.keras.layers.Dense(self.neurons, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')])
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train_and_test(self):

        self.model.fit(self.data.x_train, self.data.y_train,
                  epochs=self.epochs,
                  validation_data=(self.data.x_val, self.data.y_val),
                  batch_size=self.batch_size,
                  verbose=0
                  )
        result = self.model.evaluate(self.data.x_test, self.data.y_test)
        accuracy = result[1]

        return accuracy


class Training:

    def __init__(self, data, model):
        self.data = data
        self.model = model

    def repeat_train(self, repeats):
        acc = []
        for i in range(repeats):
            result = self.model.train_and_test()
            acc.append(result)

        summary = self.summary(accuracy=acc)
        print(summary)

        return summary


    def summary(self, accuracy):

        accuracy_average = self.average(accuracy)
        stddev = self.stddev(accuracy)

        summary_dict = {'average accuracy': accuracy_average,
                        'STDEV': stddev,
                        'repeats': len(accuracy),
                        'training data': self.data.train_num,
                        'validation data': self.data.val_num,
                        'test data': self.data.test_num,
                        'steps back': self.data.steps_back,
                        'steps forward': self.data.steps_forward,
                        'batch_size': self.batch_size,
                        'epochs': self.model.epochs,
                        'neurons': self.model.neurons,
                        'learning_rate': self.model.learning_rate,
                        }

        summary = pd.DataFrame([summary_dict])
        return summary


    @staticmethod
    def average(num):
        avg = sum(num) / len(num)
        return avg

    def variance(self, num):
        n = len(num)
        var = sum((x - self.average(num)) ** 2 for x in num) / n
        return var

    def stddev(self, num):
        var = self.variance(num)
        std_dev = math.sqrt(var)
        return std_dev