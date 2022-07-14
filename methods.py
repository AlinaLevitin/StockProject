import numpy as np
import pandas as pd
import os
import glob
import pickle


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

    # Finding the maximum value
    equals_value = data_raw['A-B'].max()
    equals = 30
    # equals = data_raw.index[data_raw['A-B'] == equals_value].tolist()[0]

    # Finding the result after the maximum difference
    A_future = data_raw.iloc[equals + steps_forward, 1] - data_raw.iloc[equals, 1]
    B_future = data_raw.iloc[equals + steps_forward, 2] - data_raw.iloc[equals, 2]

    A_future_result = pd.Series([up_or_down(A_future)])
    B_future_result = pd.Series([up_or_down(B_future)])

    # finding the previus values
    A_past = data_raw.iloc[equals - steps_back:equals + 1, 1]
    B_past = data_raw.iloc[equals - steps_back:equals + 1, 2]

    # Preparing the final np array
    A = A_past.append(A_future_result)
    B = B_past.append(B_future_result)

    A_array = A.to_numpy()
    B_array = B.to_numpy()

    return np.stack((A_array, B_array))


def read_all_data(path):
    date_folders = glob.glob(path + "\\*")
    data = []
    for date in date_folders:
        symbol_folders = glob.glob(date + "\\*")
        for symbol in symbol_folders:
            files = os.listdir(symbol)
            for file in files:
                os.chdir(symbol)
                data.append(read_and_get_values(file, 10, 10))
    return data


def save_pickle(data):
    pickle.dump(data, open("data.p", "wb"))


