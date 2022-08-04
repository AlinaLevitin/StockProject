import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split


class Data:

    def __init__(self, cwd, steps_back, steps_forward, percent, file=None):
        self.cwd = cwd
        self.steps_back = steps_back
        self.steps_forward = steps_forward
        self.percent = percent
        self.file = file
        self.data = self.read_or_open()

    def read_or_open(self):
        try:
            data = self.open_all_data()
        except:
            data = self.read_all_data()

        return data


    def open_all_data(self):
        os.chdir(self.cwd)
        df = pd.read_csv(self.file)
        return df


    def read_all_data(self):
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
        return all_data

    def read_and_get_values(self, file):
        one_percent = float(file.split('_')[4].replace('.csv', ''))

        data_raw = pd.read_csv(file, header=None, names=['time', 'A', 'B'])

        # New column with difference
        data_raw['A-B'] = data_raw['A'] - data_raw['B']
        data_raw['A-B'] = data_raw['A-B'].abs()

        # Finding the maximum value within the middle half
        frac1 = int(data_raw.shape[0] * 0.25)
        frac2 = int(data_raw.shape[0] * 0.75)
        max_value = data_raw.iloc[frac1:frac2, 3].abs().max()
        max_index = data_raw.index[data_raw['A-B'] == max_value].tolist()[0]

        # Finding the result after the maximum difference
        future = data_raw.iloc[max_index:max_index + self.steps_forward, :3].copy()
        A_0 = future.iloc[0, 1]
        A_mask1 = future.A > A_0 + self.percent * one_percent
        A_mask2 = future.A < A_0 - self.percent * one_percent
        A_plus = future.loc[A_mask1, 'A']
        A_minus = future.loc[A_mask2, 'A']
        A_future_result = self.win_or_lose(A_plus, A_minus)

        B_0 = future.iloc[0, 2]
        B_mask1 = future.B > B_0 + self.percent * one_percent
        B_mask2 = future.B < B_0 - self.percent * one_percent
        B_plus = future.loc[B_mask1, 'B']
        B_minus = future.loc[B_mask2, 'B']
        B_future_result = self.win_or_lose(B_plus, B_minus)

        # finding the previous values and converting to dataframe.T
        A_past = data_raw.iloc[max_index - self.steps_back:max_index + 1, 1].to_frame()
        B_past = data_raw.iloc[max_index - self.steps_back:max_index + 1, 2].to_frame()
        df_A = A_past.T
        df_B = B_past.T
        A_len = df_A.shape[1]
        B_len = df_B.shape[1]
        columns_A = [f"A_x({x})" for x in range(A_len)]
        columns_B = [f"B_x({x})" for x in range(B_len)]
        df_A.columns = columns_A
        df_B.columns = columns_B

        # Preparing the final DataFrame
        df_B.reset_index(inplace=True)
        df_A.reset_index(inplace=True)

        data = pd.concat([df_A, df_B], axis=1, join='outer')
        data.drop(labels='index', axis=1, inplace=True)
        data['A_(y)'] = A_future_result
        data['B_(y)'] = B_future_result

        # Preparing chart DataFrame
        A_chart = data_raw.iloc[max_index:max_index + self.steps_forward + 1, 1].to_frame()
        B_chart = data_raw.iloc[max_index:max_index + self.steps_forward + 1, 2].to_frame()
        df_A_chart = A_chart.T
        df_B_chart = B_chart.T
        A_len_chart = df_A_chart.shape[1]
        B_len_chart = df_B_chart.shape[1]
        columns_A_chart = [f"A_x({x})" for x in range(A_len_chart)]
        columns_B_chart = [f"B_x({x})" for x in range(B_len_chart)]

        df_A_chart.columns = columns_A_chart
        df_B_chart.columns = columns_B_chart

        df_A_chart.reset_index(inplace=True)
        df_B_chart.reset_index(inplace=True)

        data_chart = pd.concat([df_A_chart, df_B_chart], axis=1, join='outer')
        data_chart.drop(labels='index', axis=1, inplace=True)
        data_chart['A_(y)'] = A_future_result
        data_chart['B_(y)'] = B_future_result
        return data


    @staticmethod
    def win_or_lose(plus, minus):
        if not plus.empty and minus.empty:
            return 1
        else:
            return 0

    @staticmethod
    def scale_data(df):
        return (df - df.min()) / (df.max() - df.min())


class TrainingData(Data):

    def __init__(self,cwd, steps_back, steps_forward, percent, file=None):
        super().__init__(cwd, steps_back, steps_forward, percent, file)
        self.x = self.data.iloc[:, :-3]
        self.y = self.data.iloc[:, -2:]
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.split_data()
        self.train_num = self.x_train.shape[0]
        self.val_num = self.x_val.shape[0]
        self.test_num = self.x_test.shape[0]
        self.data = self.read_or_open()

    def __repr__(self):
        return f"training data: {self.train_num}, validation data: {self.val_num}, test data: {self.test_num}"

    def split_data(self):
        x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                            random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train_and_val, y_train_and_val, test_size=0.2,
                                                          random_state=42)
        x_train = self.scale_data(x_train)
        x_val = self.scale_data(x_val)
        x_test = self.scale_data(x_test)

        return x_train, x_val, x_test, y_train, y_val, y_test
