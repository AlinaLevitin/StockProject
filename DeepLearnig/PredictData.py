# TODO: document the file
from .Data import Data
import pandas as pd
import os


class PredictData(Data):

    def __init__(self, cwd: str):
        super().__init__(cwd)
        self.x = None

    def read_all_predict_data(self, steps_back: int):
        """
        reads the data according to the chosen parameters

        :param steps_back: the number of time points in the past

        :return: pandas DataFrame with the input to predict

        """

        self.steps_back = steps_back

        all_data = pd.DataFrame()

        predict_data = self.cwd + "\\predict_data"
        files = os.listdir(predict_data)
        for file in files:
            os.chdir(predict_data)
            data = self.read_and_get_values(file)
            all_data = pd.concat([all_data, data])
            print(f"analyzed {file}")
        self.x = all_data
        print(all_data)

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

        data_raw = pd.read_csv(file, header=None, names=['time', first_symbol, second_symbol])

        A_past = data_raw.iloc[-self.steps_back:, 1].to_frame()
        B_past = data_raw.iloc[-self.steps_back:, 2].to_frame()
        df_A = A_past.T
        df_B = B_past.T
        A_len = df_A.shape[1]
        columns = [f"x({x})" for x in range(A_len)]
        df_A.columns = columns
        df_B.columns = columns
        time_point = pd.concat([df_A, df_B], axis=0, join='outer')

        return time_point
