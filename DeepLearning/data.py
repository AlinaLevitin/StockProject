"""
Data class to handle csv time-point data sets and convert them to pandas Dataframes
"""
import pandas as pd
import os
import json

import utils


class Data:
    """
        A class to collect data for deep-learning

    ...

    Attributes
    ----------
    cwd : str
        a string of the working directory
    steps_back : int
        the number of time points in the past
    steps_forward : int
        the number of time points in the future
    percent : int
        percent as the threshold for win or loss outcome
    interval : int
        interval between time points collected
    data : pandas DataFrame
        all the data collected according to the above parameters
        use class method read_all_data with the chosen parameters
        or open existing data with class method open_all_data

    Methods
    -------
    read_all_data(self, steps_back, steps_forward, percent, interval)
        reads the data according to the chosen parameters
    read_and_get_values(self, file)
        reads the data according to the chosen parameters from a chosen file
    save_all_data(self, name='data')
        saves the data as csv file and the parameters as json file
        default name is 'data'
    open_all_data(self, name='data')
        opens the data from csv file and the parameters from json file
        default name is 'data'

    """

    def __init__(self, cwd: str):
        """
        :param cwd: a string of the working directory
        """

        self.cwd = cwd
        self.steps_back = None
        self.steps_forward = None
        self.percent = None
        self.interval = None
        self.data = None

    def __repr__(self):
        """
        :return: parameters of object number of data points, steps back, steps forward, percent and interval
        """
        return f"number of data points : {self.data.shape[0]}, " \
               f"steps_back: {self.steps_back}, steps_forward: {self.steps_forward}, " \
               f"percent: {self.percent}, interval: {self.interval}"

    def read_all_data(self, steps_back: int, steps_forward: int, percent: float, interval: int):
        """
        reads the data according to the chosen parameters

        :param steps_back: the number of time points in the past
        :param steps_forward: the number of time points in the future
        :param percent: percent as the threshold for win or loss outcome
        :param interval: interval between time points collected

        :return: pandas DataFrame with the input and targets for deep learning
        """

        self.steps_back = steps_back
        self.steps_forward = steps_forward
        self.percent = percent
        self.interval = interval

        all_data = pd.DataFrame()
        print('-' * 60)
        print('reading all files in all_hist folder')
        analysis_hist = self.cwd + "\\all_hist"
        files = os.listdir(analysis_hist)
        for file in files:
            os.chdir(analysis_hist)
            data = self.read_and_get_values(file)
            all_data = pd.concat([all_data, data])
            print(f"analyzed {file}")
        all_data.reset_index(drop=True, inplace=True)
        print('-' * 60)
        self.data = all_data

    def read_and_get_values(self, file):
        """
        reads the data according to the chosen parameters from a chosen file

        :param file: csv file containing time series of symbol
        :type file: csv file

        :return: pandas DataFrame
            input and targets for one instance
        """

        one_percent = float(file.split('_')[4].replace('.csv', ''))

        data_raw = pd.read_csv(file, header=None, names=['time', 'A', 'B'])
        shrink_dataframe = int(data_raw.shape[0] - self.steps_forward)

        data = pd.DataFrame()

        # Finding the result for each time point
        for time in range(self.steps_back, shrink_dataframe, self.interval):
            future = data_raw.iloc[time:time + self.steps_forward, :].copy()

            A_future_long = self.long(future, 'A', one_percent)
            B_future_long = self.long(future, 'B', one_percent)
            A_future_short = self.short(future, 'A', one_percent)
            B_future_short = self.short(future, 'B', one_percent)

            A_past = data_raw.iloc[time - self.steps_back:time, 1].to_frame()
            B_past = data_raw.iloc[time - self.steps_back:time, 2].to_frame()
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
            time_point['A_long_(y)'] = A_future_long
            time_point['B_long_(y)'] = B_future_long
            time_point['A_short_(y)'] = A_future_short
            time_point['B_short_(y)'] = B_future_short
            data.index.name = 'index'
            data = pd.concat([data, time_point])
        return data

    def save_all_data(self, name: str = 'data'):
        """
        saves the data to a csv file and the parameters to a json file

        :param name: name of file, default is data
        :type name: str optional

        :return: csv file
        saves a csv file with the data and a json file with the parameters
        """

        utils.save_to_csv(name, self.data, self.cwd)
        params = {'steps_back': self.steps_back, 'steps_forward': self.steps_forward, 'percent': self.percent,
                  'interval': self.interval}
        json_save = json.dumps(params)
        with open(f'params_{name}.json', 'w') as json_file:
            json_file.write(json_save)
        print(self)
        print(f'Saved parameters as params_{name}.json')

    def open_all_data(self, name: str = 'data'):
        """
        opens the data from csv file and the parameters from json file

        :param name: name of file, default is data
        :type name: str optional

        :return: Data instance
        opens a csv file with the data and a json file with the parameters
        """
        print("-" * 60)
        print("Loading data")
        os.chdir(self.cwd)
        data = pd.read_csv(f'{name}.csv')
        with open(f'params_{name}.json') as json_file:
            params = json.load(json_file)
        self.steps_back = params['steps_back']
        self.steps_forward = params['steps_forward']
        self.percent = params['percent']
        self.interval = params['interval']
        self.data = data
        print(f"number of data points : {self.data.shape[0]}, "
              f"steps_back: {self.steps_back}, steps_forward: {self.steps_forward}, "
              f"percent: {self.percent}, interval: {self.interval}")
        print(f'opened the data from {name}.csv and the parameters from params_{name}.json')
        print("-" * 60)

    def long(self, future, symbol: str, one_percent: float) -> int:
        """
        method to check for win or loss in long position

        :param future: pandas DataFrame of the future according to the steps forward
        :type future: pandas DataFrame
        :param symbol: 'A' or 'B'
        :param one_percent: float

        :return: int
        """

        if symbol == 'A':
            col = 1
        else:
            col = 2

        time_0 = future.iloc[0, col]
        mask1 = future.iloc[:, col] > time_0 + self.percent*one_percent
        mask2 = future.iloc[:, col] < time_0 - self.percent*one_percent

        plus = future.loc[mask1, symbol]
        minus = future.loc[mask2, symbol]

        if not plus.empty and minus.empty:
            return 1
        else:
            return 0

    def short(self, future, symbol: str, one_percent: float) -> int:
        """
        method to check for win or loss in short position

        :param future: pandas DataFrame of the future according to the steps forward
        :type future: pandas DataFrame
        :param symbol: 'A' or 'B'
        :param one_percent: float

        :return: int
        """

        if symbol == 'A':
            col = 1
        else:
            col = 2

        time_0 = future.iloc[0, col]
        mask1 = future.iloc[:, col] > time_0 + self.percent*one_percent
        mask2 = future.iloc[:, col] < time_0 - self.percent*one_percent

        plus = future.loc[mask1, symbol]
        minus = future.loc[mask2, symbol]

        if not minus.empty and plus.empty:
            return 1
        else:
            return 0

    @staticmethod
    def scale_data(df):
        """
        to scale the data to range between -1 and 1

        :param df: pandas DataFrame of inputs
        :return: pandas DataFrame with data ranging between -1 and 1
        """

        return (df - df.min()) / (df.max() - df.min())
