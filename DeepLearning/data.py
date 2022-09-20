"""
Data class to handle csv time-point data sets and convert them to pandas Dataframes
"""
# TODO rewrite methods to be more clear
import os
import json

import pandas as pd

import DeepLearning


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
    percent_long : float
        percent as the threshold for win or loss outcome
    percent_short : float
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
        self.start_date = None
        self.end_date = None
        self.steps_back = None
        self.steps_forward = None
        self.percent_long = None
        self.percent_short = None
        self.interval = None
        self.data = None

    def __repr__(self):
        """
        :return: parameters of object number of data points, steps back, steps forward, percent, interval,
                start date and end date.
        """
        return f"number of data points : {self.data.shape[0]}, " \
               f"steps_back: {self.steps_back}, steps_forward: {self.steps_forward}, " \
               f"percent_long: {self.percent_long}, percent_short: {self.percent_short}, interval: {self.interval}"\
               f"start_date: {self.start_date}, end_date: {self.end_date}"

    def read_all_data(self, steps_back: int, steps_forward: int,
                      percent_long: float, percent_short: float, interval: int,
                      start_date: int, end_date: int
                      ):
        """
        reads the data according to the chosen parameters

        :param steps_back: the number of time points in the past
        :param steps_forward: the number of time points in the future
        :param percent_short: percent as the threshold for win or loss outcome
        :param percent_long: percent as the threshold for win or loss outcome
        :param interval: interval between time points collected
        :param end_date: date in which to start reading files
        :param start_date: date in which to end reading files

        :return: pandas DataFrame with the input and targets for deep learning
        """

        self.steps_back = steps_back
        self.steps_forward = steps_forward
        self.percent_long = percent_long
        self.percent_short = percent_short
        self.interval = interval
        self.start_date = start_date
        self.end_date = end_date

        all_data = pd.DataFrame()
        print('-' * 60)
        print(f'Reading all files in all_hist folder between {self.start_date} and {self.end_date}')
        print('-' * 60)
        analysis_hist = self.cwd + "\\all_hist"
        os.makedirs(analysis_hist, exist_ok=True)
        files = os.listdir(analysis_hist)
        if not files:
            raise FileNotFoundError('all_hist folder is empty,'
                                    'please copy all files to all_hist folder using copy_to_one_dir method')
        else:
            for file in files:
                os.chdir(analysis_hist)
                data = self.read_and_get_values(file)
                all_data = pd.concat([all_data, data])

        all_data.reset_index(drop=True, inplace=True)
        print('-' * 60)
        print('Finished reading files')
        print('-' * 60)
        print(self)
        print('-' * 60)
        self.data = all_data

    def read_and_get_values(self, file: str):
        """
        reads the data according to the chosen parameters from a chosen file

        :param file: csv file containing time series of symbol
        :type file: csv file

        :return: pandas DataFrame
            input and targets for one instance
        """
        data = pd.DataFrame()

        one_percent = float(file.split('_')[-1].replace('.csv', ''))
        date = int(file.split('_')[2])
        if int(self.start_date) < date < int(self.end_date):
            data_raw = pd.read_csv(file, header=None, names=['time', 'A', 'B'])
            shrink_dataframe = int(data_raw.shape[0] - self.steps_forward)

            # Finding the result for each time point
            for index in range(self.steps_back, shrink_dataframe, self.interval):
                time_stamp = data_raw.loc[index, 'time']
                h_m_s = time_stamp.split(' ')[1]
                if '10:00:00' < h_m_s < '14:00:00':
                    future = data_raw.iloc[index:index + self.steps_forward, :].copy()

                    A_future_long_profit = self.long_profit(future, 'A', one_percent)
                    B_future_long_profit = self.long_profit(future, 'B', one_percent)
                    A_future_short_profit = self.short_profit(future, 'A', one_percent)
                    B_future_short_profit = self.short_profit(future, 'B', one_percent)

                    A_past = data_raw.iloc[index - self.steps_back:index, 1].to_frame()
                    B_past = data_raw.iloc[index - self.steps_back:index, 2].to_frame()
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
                    time_point['A_long_(y)'] = A_future_long_profit
                    time_point['B_long_(y)'] = B_future_long_profit
                    time_point['A_short_(y)'] = A_future_short_profit
                    time_point['B_short_(y)'] = B_future_short_profit
                    data.index.name = 'index'
                    data = pd.concat([data, time_point])
            print(f"analyzed {file}")
            return data

    def save_all_data(self, name: str = 'data'):
        """
        saves the data to a csv file and the parameters to a json file

        :param name: name of file, default is data
        :type name: str optional

        :return: csv file
        saves a csv file with the data and a json file with the parameters
        """

        DeepLearning.utils.save_to_csv(name, self.data, self.cwd)
        params = {'steps_back': self.steps_back, 'steps_forward': self.steps_forward, 'percent_long': self.percent_long,
                  'percent_short': self.percent_short, 'interval': self.interval,
                  'start_date': self.start_date, 'end_date': self.end_date}
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
        self.percent_long = params['percent_long']
        self.percent_short = params['percent_short']
        self.interval = params['interval']
        # self.start_date = params['start_date']
        # self.end_date = params['end_date']
        self.data = data
        print(self)
        print(f'opened the data from {name}.csv and the parameters from params_{name}.json')
        print("-" * 60)

    def long_profit(self, future, symbol: str, one_percent: float) -> int:
        """
        method to check for win or loss in long position

        :param future: pandas DataFrame of the future according to the steps forward
        :type future: pandas DataFrame
        :param symbol: 'A' or 'B'
        :param one_percent: float

        :return: int
        """

        if symbol == 'A':
            column = 1
        else:
            column = 2

        stock_time_0 = future.iloc[0, column]
        profit = future.iloc[:, column] > stock_time_0 + self.percent_long*one_percent
        loss = future.iloc[:, column] < stock_time_0 - self.percent_long*one_percent

        take_profit = future.loc[profit, symbol]
        stop_loss = future.loc[loss, symbol]

        if take_profit.empty and stop_loss.empty:
            return 0
        elif not take_profit.empty and stop_loss.empty:
            return 1
        elif take_profit.empty and not stop_loss.empty:
            return 0
        elif not take_profit.empty and not stop_loss.empty:
            if profit.index[0] < loss.index[0]:
                return 1
            else:
                return 0

    def short_profit(self, future, symbol: str, one_percent: float) -> int:
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

        stock_time_0 = future.iloc[0, col]
        profit = future.iloc[:, col] < stock_time_0 + self.percent_short*one_percent
        loss = future.iloc[:, col] > stock_time_0 - self.percent_short*one_percent

        take_profit = future.loc[profit, symbol]
        stop_loss = future.loc[loss, symbol]

        if take_profit.empty and stop_loss.empty:
            return 0
        elif not take_profit.empty and stop_loss.empty:
            return 1
        elif take_profit.empty and not stop_loss.empty:
            return 0
        elif not take_profit.empty and not stop_loss.empty:
            if profit.index[0] < loss.index[0]:
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
