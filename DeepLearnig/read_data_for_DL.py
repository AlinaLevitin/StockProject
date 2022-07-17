import numpy as np
import pandas as pd
import os
import glob
import pickle


def read_all_data(cwd, steps_back, steps_forward):
    analysis_hist = glob.glob(cwd + "\\analysis_hist")
    date_folders = glob.glob(analysis_hist[0] + "\\*")
    for date in date_folders:
        symbol_folders = glob.glob(date + "\\*")
        for symbol in symbol_folders:
            os.chdir(symbol)
            file = os.listdir(symbol)
            data = read_and_get_values(file[0], steps_back, steps_forward)
            try:
                all_data = pd.concat([all_data, data])
            except:
                all_data = data
    all_data.reset_index(drop=True, inplace=True)
    save_to_csv(all_data, cwd)
    print(all_data)


def up_or_down(future):
    if future > 0:
        return 1
    else:
        return 0


def read_and_get_values(file, steps_back, steps_forward):
    data_raw = pd.read_csv(file, header=None, names=['time', 'A', 'B'])

    # Standardizing the data
    data_raw['A'] = data_raw['A'] - data_raw['A'].iloc[0]
    data_raw['B'] = data_raw['B'] - data_raw['B'].iloc[0]

    # New column with difference
    data_raw['A-B'] = data_raw['A'] - data_raw['B']

    # Finding the maximum value within the middle third
    third = int(data_raw.shape[0] * 0.33)
    two_thirds = int(data_raw.shape[0] * 0.66)
    equals_value = data_raw.iloc[third:two_thirds, 3].max()
    equals = data_raw.index[data_raw['A-B'] == equals_value].tolist()[0]

    # Finding the result after the maximum difference
    A_future = data_raw.iloc[equals + steps_forward, 1] - data_raw.iloc[equals, 1]
    B_future = data_raw.iloc[equals + steps_forward, 2] - data_raw.iloc[equals, 2]

    A_future_result = pd.Series([up_or_down(A_future)])
    B_future_result = pd.Series([up_or_down(B_future)])

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
    return data


def save_to_csv(data, cwd):
    os.chdir(cwd)
    data.to_csv('data.csv', index=False)


def save_pickle(data, cwd):
    os.chdir(cwd)
    pickle.dump(data, open("data.p", "wb"))
