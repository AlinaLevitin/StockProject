import pickle
import pandas as pd
import os


def open_csv(cwd, file_name):
    os.chdir(cwd)
    df = pd.read_csv(file_name)
    return df


def save_to_csv(name, data, cwd):
    os.chdir(cwd)
    data.to_csv(name, index=False)


def save_pickle(data):
    pickle.dump(data, open("data.p", "wb"))


def average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg
