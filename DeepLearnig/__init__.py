import pandas as pd
import os
import glob
from sklearn.model_selection import train_test_split


def read_all_data(cwd, steps_back, steps_forward):
    analysis_hist = glob.glob(cwd + "\\analysis_hist")
    date_folders = glob.glob(analysis_hist[0] + "\\*")
    for date in date_folders:
        symbol_folders = glob.glob(date + "\\*")
        for symbol in symbol_folders:
            os.chdir(symbol)
            file = os.listdir(symbol)
            try:
                data, data_chart = read_and_get_values(file[0], steps_back, steps_forward)
            except:
                print(f'Something is wrong with {file[0]}')
            try:
                all_data = pd.concat([all_data, data])
                all_data_chart = pd.concat([all_data_chart, data_chart])
            except:
                all_data = data
                all_data_chart = data_chart
    all_data.reset_index(drop=True, inplace=True)
    all_data_chart.reset_index(drop=True, inplace=True)
    save_to_csv('data.csv', all_data, cwd)
    save_to_csv('data_chart.csv', all_data_chart, cwd)
    print(all_data)


def up_or_down(future, past):
    if future > past:
        return 1
    else:
        return 0


def read_and_get_values(file, steps_back, steps_forward):
    data_raw = pd.read_csv(file, header=None, names=['time', 'A', 'B'])

    # New column with difference
    data_raw['A-B'] = data_raw['A'] - data_raw['B']
    data_raw['A-B'] = data_raw['A-B'].abs()

    # Finding the maximum value within the middle half
    frac1 = int(data_raw.shape[0] * 0.25)
    frac2 = int(data_raw.shape[0] * 0.75)
    equals_value = data_raw.iloc[frac1:frac2, 3].abs().max()
    equals = data_raw.index[data_raw['A-B'] == equals_value].tolist()[0]

    # Finding the result after the maximum difference
    A_future_result = up_or_down(data_raw.iloc[equals + steps_forward, 1], data_raw.iloc[equals, 1])
    B_future_result = up_or_down(data_raw.iloc[equals + steps_forward, 2], data_raw.iloc[equals, 2])

    # finding the previous values and converting to dataframe.T
    A_past = data_raw.iloc[equals - steps_back:equals + 1, 1].to_frame()
    B_past = data_raw.iloc[equals - steps_back:equals + 1, 2].to_frame()
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
    A_chart = data_raw.iloc[equals:equals + steps_forward+1, 1].to_frame()
    B_chart = data_raw.iloc[equals:equals + steps_forward+1, 2].to_frame()
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

    return data, data_chart


def save_to_csv(name, data, cwd):
    os.chdir(cwd)
    data.to_csv(name, index=False)


def scale_data(df):
    return (df - df.min()) / (df.max() - df.min())


class Data:

    def __init__(self, data):
        self.data = data
        self.x = self.data.iloc[:, :-3]
        self.y = self.data.iloc[:, -2:]


class TrainingData(Data):

    def __init__(self, data):
        super().__init__(data)
        self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = self.split_data()

    def __repr__(self):
        train_num = self.x_train.shape[0]
        val_num = self.x_val.shape[0]
        test_num = self.x_test.shape[0]
        return f"training data: {train_num}, validation data: {val_num}, test data: {test_num}"

    def split_data(self):
        x_train_and_val, x_test, y_train_and_val, y_test = train_test_split(self.x, self.y, test_size=0.2,
                                                                            random_state=42)
        x_train, x_val, y_train, y_val = train_test_split(x_train_and_val, y_train_and_val, test_size=0.3,
                                                          random_state=42)
        x_train = scale_data(x_train)
        x_val = scale_data(x_val)
        x_test = scale_data(x_test)

        return x_train, x_val, x_test, y_train, y_val, y_test