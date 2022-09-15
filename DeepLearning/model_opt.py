"""
ModelOpt class for model architecture optimization
"""
import os

import DeepLearning


class ModelOpt:
    """
            A class to generate a model for deep learning

    ...

    Attributes
    ----------
    cwd : str
        a string of the working directory

    Methods
    -------
    repeat_train(self, repeats, neurons, epochs, learning_rate, batch_size)
        for optimization of hyperparameters, this will repeat the class method train_and_test
        and result a summery file

    """

    def __init__(self, cwd: str):
        """
        :param cwd: a string of the working directory
        """

        self.cwd = cwd

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
        model = DeepLearning.Model(self.cwd)
        model.set_model(data, neurons, epochs, learning_rate, batch_size)
        model.get_summary()
        os.makedirs(self.cwd + "\\model_opt", exist_ok=True)
        os.chdir(self.cwd + "\\model_opt")
        model.model.save_weights('model.h5')
        acc = []
        for i in range(repeats):
            print('-' * 60)
            print(f'Repeat #{i + 1} out of {repeats}')
            print(model)
            print('-' * 60)
            os.chdir(self.cwd + "\\model_opt")
            model.model.load_weights('model.h5')
            result = model.train_and_test(data, save=False)
            accuracy = result[0]
            acc_and_loss = result[1]
            acc.append(accuracy)
            report = DeepLearning.Reports(self.cwd, data, model)
            report.repeat_train_report(i, accuracy, acc_and_loss)
        report = DeepLearning.Reports(self.cwd, data, model)
        path = f"\\model_opt\\reports\\{model.neurons}_neurons"
        summary = report.repeat_train_summary(accuracy=acc, path=path)
        return summary
