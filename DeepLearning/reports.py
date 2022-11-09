"""
Reports class to generate reports after raining a model.
will generate csv or Excel files with accuracy, the model parameters and history of accuracy and loss over time.
"""
import os
import time

import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

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
    single_train_report(self, accuracy, acc_and_loss)
        Generates an Excel file after training a model
    repeat_train_report(self, i, accuracy, acc_and_loss)
        Generates an Excel file after training a model in a repeat train
    """

    def __init__(self, cwd: str, data, model):
        self.cwd = cwd
        self.data = data
        self.model = model

    def single_train_summary(self, accuracy):
        """
        generates a summary as pandas DataFrame

        :param accuracy: float or list of accuracy after training
        """

        summary_dict = {'test accuracy': accuracy,
                        'data': self.data.data_num,
                        'start date': self.data.start_date,
                        'end date': self.data.end_date,
                        'training data': self.data.train_num,
                        'validation data': self.data.val_num,
                        'test data': self.data.test_num,
                        'percent_long': self.data.percent_long,
                        'percent_short': self.data.percent_short,
                        'steps back': self.data.steps_back,
                        'steps forward': self.data.steps_forward,
                        'batch_size': self.model.batch_size,
                        'epochs': self.model.epochs,
                        'neurons': self.model.neurons,
                        'learning_rate': self.model.learning_rate,
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

        summary_dict = {'average test accuracy': accuracy_average,
                        'STDEV': stddev,
                        'repeats': repeats,
                        'data': self.data.data_num,
                        'start date': self.data.start_date,
                        'end date': self.data.end_date,
                        'training data': self.data.train_num,
                        'validation data': self.data.val_num,
                        'test data': self.data.test_num,
                        'percent_long': self.data.percent_long,
                        'percent_short': self.data.percent_short,
                        'steps back': self.data.steps_back,
                        'steps forward': self.data.steps_forward,
                        'batch_size': self.model.batch_size,
                        'epochs': self.model.epochs,
                        'neurons': self.model.neurons,
                        'learning_rate': self.model.learning_rate,
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

    def single_train_report(self, accuracy, acc_and_loss):
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

    def confusion_matrix(self):
        """
        confusion matrix function

        :return: creates a confusion matrix fig
        """

        # if not self.data.x_test or self.data.y_test:
        #     raise ValueError('Please use split training data')

        # make predictions on the input
        print('Preparing confusion matrix')
        print('=' * 60)
        y_pred = self.model.model.predict(self.data.x_test)

        # make confusion matrix
        cm = multilabel_confusion_matrix(self.data.y_test, y_pred.round(0))

        # set fig size
        figsize = (10, 10)

        # Create the confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        n_classes = cm[0].shape[0]

        text_labels = ['A_long', 'B_long', 'A_short', 'B_short']

        for p in range(0, len(cm)):
            # Make axes and colored boxes
            fig = plt.figure(figsize=figsize)
            ax = plt.subplot()
            cax = ax.matshow(cm[0], cmap=plt.cm.Blues)
            fig.colorbar(cax)

            # Create classes
            classes = False

            if classes:
                labels = classes
            else:
                labels = np.arange(n_classes)

            ax.set(title=f"Confusion Matrix {text_labels[p]}",
                   xlabel='Predicted Label',
                   ylabel='True Label',
                   xticks=np.arange(n_classes),
                   yticks=np.arange(n_classes),
                   xticklabels=labels,
                   yticklabels=labels)

            # Set x-axis labels to bottom
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.tick_bottom()

            # Adjust label size
            ax.yaxis.label.set_size(20)
            ax.xaxis.label.set_size(20)
            ax.title.set_size(20)

            # Set threshold for different colors
            threshold = (cm.max() + cm.min()) / 2.

            # Plot the text on each cell
            for i, j in itertools.product(range(cm[p].shape[0]), range(cm[p].shape[1])):
                plt.text(j, i, f"{cm[p][i, j]} ({cm_norm[p][i, j] * 100:.1f}%)",
                         horizontalalignment="center",
                         color="white" if cm[p][i, j] > threshold else "black",
                         size=15)
        plt.show()

    @staticmethod
    def plot_loss(history):
        epochs_plt = [i for i in range(1, len(history.history['loss']) + 1)]
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(10, 7))
        plt.plot(epochs_plt, [loss, val_loss])
        plt.ylabel('loss')
        plt.xlabel('epochs')
