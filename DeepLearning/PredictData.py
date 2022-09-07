"""
PredictData class inherits from Data class to predict results according to a trained neural network
"""
from .Data import Data
import pandas as pd
import os


class PredictData(Data):
    """

    A class to collect data for prediction \n

    ...

    Attributes
    ----------
    cwd : str
        a string of the working directory
    data : pandas DataFrame
        all the data collected according to the above parameters
        use class method read_all_data with the chosen parameters
        or open existing data with class method open_all_data

    Methods
    -------
    FROM Data CLASS:
    save_all_data(self, name='data')
        saves the data as csv file and the parameters as json file
        default name is 'data'
    open_all_data(self, name='data')
        opens the data from csv file and the parameters from json file
        default name is 'data'
    read_all_data(self, steps_back, steps_forward, percent, interval)
        reads the data according to the chosen parameters
    read_and_get_values(self, file)
        reads the data according to the chosen parameters from a chosen file

    read_all_predict_data(self, steps_back: int)
        reads the data according to the chosen parameters
    read_and_get_values(self, file)
        reads the data according to the chosen parameters from a chosen file
    pandas_to_numpy(self)
        converts pandas to numpy array
    """

    def __init__(self, cwd: str):
        """
        :param cwd: a string of the working directory
        """
        super().__init__(cwd)
        self.data = None
        self.input_shape = None
        self.steps_back = None

    def __repr__(self):
        """
        :return: parameters of object number of data points and steps back
        """
        return f"number of data points : {self.data.shape[0]}, " \
               f"steps_back: {self.steps_back}"

    def read_all_predict_data(self, steps_back: int):
        """
        reads the data according to the chosen parameters

        :param steps_back: the number of time points in the past

        :return: pandas DataFrame with the input to predict
        """

        self.steps_back = steps_back
        print('=' * 60)
        all_data = pd.DataFrame()
        print('reading all files in predict_data folder')
        print('=' * 60)
        predict_data = self.cwd + "\\predict_data"
        files = os.listdir(predict_data)
        for file in files:
            os.chdir(predict_data)
            data = self.read_and_get_values(file)
            all_data = pd.concat([all_data, data])
            print(f"analyzed {file}")
        self.data = self.scale_data(all_data)
        self.input_shape = self.data.shape[1]
        print('=' * 60)
        return self.data

    def read_and_get_values(self, file):
        """
        reads the data according to the chosen parameters from a chosen file

        :param file: csv file containing time series of symbol
        :type file: csv file

        :return: pandas DataFrame
            input and targets for one instance
        """
        first_symbol = file.split('_')[0]
        second_symbol = file.split('_')[1]

        data_raw = pd.read_csv(file, header=None, names=['time', 'A', 'B'])

        A_past = data_raw.iloc[-self.steps_back:, 1].to_frame()
        B_past = data_raw.iloc[-self.steps_back:, 2].to_frame()
        df_A = A_past.T
        df_B = B_past.T
        A_len = df_A.shape[1]
        B_len = df_B.shape[1]
        columns_A = [f"A_x({x})" for x in range(A_len)]
        columns_B = [f"B_x({x})" for x in range(B_len)]
        df_A.columns = columns_A
        df_B.columns = columns_B
        df_B.reset_index(inplace=True)
        df_A.reset_index(inplace=True)
        time_point = pd.concat([df_A, df_B], axis=1, join='outer')
        time_point.drop(labels='index', axis=1, inplace=True)
        index = pd.Index([first_symbol + ' ' + second_symbol])
        time_point.set_index(index, inplace=True)

        return time_point

    def pandas_to_numpy(self):
        """
        converts pandas to numpy array

        :return: numpy array
        """
        return self.data.to_numpy(copy=True)
