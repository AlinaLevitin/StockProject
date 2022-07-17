import pickle
import pandas as pd
import os


def open_csv(cwd, file_name):
    os.chdir(cwd)
    df = pd.read_csv(file_name)
    return df


def save_pickle(data):
    pickle.dump(data, open("data.p", "wb"))


