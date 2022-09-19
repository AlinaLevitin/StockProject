"""
Reports class to generate reports after raining a model.
will generate csv or Excel files with accuracy, the model parameters and history of accuracy and loss over time.
"""

import os
import time

import pandas as pd

import DeepLearning


class Reports:
    """
        A class to generate reports after training a model

    ...

    Attributes
    ----------
    cwd : str
        a string of the working directory
    data : pandas DataFrame
            should be TrainingData instance after split_data function
    model: TensorFlow/Keras sequential model

    Methods
    -------
    single_train_summary(self, accuracy)
        generates a summary as pandas DataFrame
    repeat_train_summary(self, accuracy, path: str=None)
        generates a summary as pandas DataFrame
    train_report(self, accuracy, acc_and_loss)
        Generates an Excel file after training a model
    repeat_train_report(self, i, accuracy, acc_and_loss)
        Generates an Excel file after training a model in a repeat train
    """

    def __init__(self, cwd: str, data, model
                 ):
        self.cwd = cwd
        self.data = data
        self.model \
            = model

    def single_train_summary(self, accuracy):
        """
        generates a summary as pandas DataFrame

        :param accuracy: float or list of accuracy after training
        """

        neurons = self.model.neurons
        epochs = self.model.epochs
        learning_rate = self.model.learning_rate
        batch_size = self.model.batch_size

        summary_dict = {'test accuracy': accuracy,
                        'training data': self.data.train_num,
                        'validation data': self.data.val_num,
                        'test data': self.data.test_num,
                        'steps back': self.data.steps_back,
                        'steps forward': self.data.steps_forward,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'neurons': neurons,
                        'learning_rate': learning_rate,
                        }
        summary = pd.DataFrame([summary_dict])
        return summary

    def repeat_train_summary(self, accuracy, path: str = None):
        """
        generates a summary as pandas DataFrame

        :param accuracy: float or list of accuracy after training
        :param path: str optional if saving is needed
        """

        if isinstance(accuracy, list):
            repeats = len(accuracy)
        else:
            repeats = 1

        accuracy_average = DeepLearning.dl_utils.average(accuracy)
        stddev = DeepLearning.dl_utils.stddev(accuracy)

        neurons = self.model.neurons
        epochs = self.model.epochs
        learning_rate = self.model.learning_rate
        batch_size = self.model.batch_size

        summary_dict = {'average test accuracy': accuracy_average,
                        'STDEV': stddev,
                        'repeats': repeats,
                        'training data': self.data.train_num,
                        'validation data': self.data.val_num,
                        'test data': self.data.test_num,
                        'steps back': self.data.steps_back,
                        'steps forward': self.data.steps_forward,
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'neurons': neurons,
                        'learning_rate': learning_rate,
                        }
        summary = pd.DataFrame([summary_dict])
        if path:
            gmt = time.gmtime()
            time_stamp = f'{gmt[0]}_{gmt[1]}_{gmt[2]}_{gmt[3]}_{gmt[4]}'
            summary_file = f'summary_{time_stamp}'
            print('+' * 60)
            DeepLearning.utils.save_to_csv(f'{summary_file}', summary, self.cwd + path)
            print('+' * 60)
        return summary

    def train_report(self, accuracy, acc_and_loss):
        """
        Generates an Excel file after training a model

        :param accuracy: float returned from train_and_test, first item in the tuple
        :param acc_and_loss: pandas DataFrame returned from train_and_test, second item in the tuple
        """
        gmt = time.gmtime()
        time_stamp = f'{gmt[0]}_{gmt[1]}_{gmt[2]}_{gmt[3]}_{gmt[4]}'
        path = self.cwd + f"\\reports"
        os.chdir(self.cwd)
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        summary = self.single_train_summary(accuracy)
        file = f'summary_{time_stamp}'
        with pd.ExcelWriter(f'{file}.xlsx') as writer:
            summary.to_excel(writer, sheet_name=f'summary')
            acc_and_loss.to_excel(writer, sheet_name=f'acc_and_loss')
        print(f'report saved to {file}.xlsx in {path} folder')

    def repeat_train_report(self, i, accuracy, acc_and_loss):
        """
        Generates an Excel file after training a model in a repeat train

        :param i: repeat number
        :param accuracy: float returned from train_and_test, first item in the tuple
        :param acc_and_loss: pandas DataFrame returned from train_and_test, second item in the tuple
        """
        neurons = self.model.neurons
        gmt = time.gmtime()
        time_stamp = f'{gmt[0]}_{gmt[1]}_{gmt[2]}_{gmt[3]}_{gmt[4]}'
        path = self.cwd + f"\\model_opt\\reports\\{neurons}_neurons"
        os.chdir(self.cwd)
        os.makedirs(path, exist_ok=True)
        os.chdir(path)
        summary = self.single_train_summary(accuracy)
        file = f'summary_{i + 1}_repeats_{neurons}_neurons_{time_stamp}'
        with pd.ExcelWriter(f'{file}.xlsx', engine='xlsxwriter') as writer:
            summary.to_excel(writer, sheet_name=f'summary')
            acc_and_loss.to_excel(writer, sheet_name=f'acc_and_loss')
        print(f'report saved to {file}.xlsx in {path} folder')
