import pandas as pd
import os
import glob
import json


class Data:
    # A class to collect data for deeplearning
    def __init__(self, cwd):
        self.cwd = cwd
        self.steps_back = None
        self.steps_forward = None
        self.percent = None
        self.interval = None
        self.data = None

    def save_all_data(self, name='data'):
        os.chdir(self.cwd)
        self.data.to_csv(f'{name}.csv', index=False)
        params = {'steps_back': self.steps_back, 'steps_forward': self.steps_forward, 'percent': self.percent, 'interval': self.interval}
        json_save = json.dumps(params)
        with open(f'params_{name}.json', 'w') as json_file:
            json_file.json.write(json_save)

    def open_all_data(self, name='data'):
        os.chdir(self.cwd)
        data = pd.read_csv(f'{name}.csv')
        with open(f'params_{name}.json') as json_file:
            params = json.load(json_file)
        self.steps_back = params['steps_back']
        self.steps_forward = params['steps_forward']
        self.percent = params['percent']
        self.interval = params['interval']
        print(f"steps_back: {self.steps_back}, steps_forward: {self.steps_forward}, percent: {self.percent} interval: {self.interval}")
        self.data = data

    def read_all_data(self, steps_back, steps_forward, percent, interval):
        self.steps_back = steps_back
        self.steps_forward = steps_forward
        self.percent = percent
        self.interval = interval

        analysis_hist = glob.glob(self.cwd + "\\analysis_hist")
        date_folders = glob.glob(analysis_hist[0] + "\\*")
        for date in date_folders:
            symbol_folders = glob.glob(date + "\\*")
            for symbol in symbol_folders:
                files = os.listdir(symbol)
                for file in files:
                    os.chdir(symbol)
                    data = self.read_and_get_values(file)
                    try:
                        all_data = pd.concat([all_data, data])
                    except:
                        all_data = data
        all_data.reset_index(drop=True, inplace=True)
        self.data = all_data

    def read_and_get_values(self, file):

        one_percent = float(file.split('_')[4].replace('.csv', ''))

        data_raw = pd.read_csv(file, header=None, names=['time', 'A', 'B'])
        shrink_dataframe = int(data_raw.shape[0] - self.steps_forward)

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
            data = pd.concat([df_A, df_B], axis=1, join='outer')
            data.drop(labels='index', axis=1, inplace=True)
            data['A_long_(y)'] = A_future_long
            data['B_long_(y)'] = B_future_long
            data['A_short_(y)'] = A_future_short
            data['B_short_(y)'] = B_future_short

        return data

    @staticmethod
    def long(future, symbol, one_percent):

        if symbol == 'A':
            col = 1
        else:
            col = 2

        time_0 = future.iloc[0, col]
        mask1 = future.iloc[:, col] > time_0 + one_percent
        mask2 = future.iloc[:, col] < time_0 - one_percent

        plus = future.loc[mask1, symbol]
        minus = future.loc[mask2, symbol]

        if not plus.empty and minus.empty:
            return 1
        else:
            return 0

    @staticmethod
    def short(future, symbol, one_percent):

        if symbol == 'A':
            col = 1
        else:
            col = 2

        time_0 = future.iloc[0, col]
        mask1 = future.iloc[:, col] > time_0 + one_percent
        mask2 = future.iloc[:, col] < time_0 - one_percent

        plus = future.loc[mask1, symbol]
        minus = future.loc[mask2, symbol]

        if not minus.empty and plus.empty:
            return 1
        else:
            return 0
