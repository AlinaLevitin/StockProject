import math
import pandas as pd


class Optimization:

    def __init__(self, model, data):
        self.model = model
        self.data = data

    def repeat_train(self, repeats, neurons, epochs, learning_rate, batch_size):
        acc = []
        for i in range(repeats):
            result = self.model.train_and_test(self.data, neurons, epochs, learning_rate, batch_size)
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
                        'batch_size': self.model.batch_size,
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