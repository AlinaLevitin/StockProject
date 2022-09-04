import math
import pandas as pd
import os
import glob
import shutil


def open_csv(cwd, file_name):
    os.chdir(cwd)
    df = pd.read_csv(file_name)
    return df


def save_to_csv(name, data, cwd):
    os.chdir(cwd)
    data.to_csv(name, index=False)


def average(num):
    if isinstance(num, list):
        avg = sum(num) / len(num)
        return avg
    else:
        return num


def variance(num):
    if isinstance(num, list):
        n = len(num)
        var = sum((x - average(num)) ** 2 for x in num) / n
        return var
    else:
        return num


def stddev(num):
    if isinstance(num, list):
        var = variance(num)
        std_dev = math.sqrt(var)
        return std_dev
    else:
        return num


def copy_to_one_dir(cwd):
    analysis_hist = glob.glob(cwd + "\\analysis_hist")
    date_folders = glob.glob(analysis_hist[0] + "\\*")
    for date in date_folders:
        symbol_folders = glob.glob(date + "\\*")
        for symbol in symbol_folders:
            files = os.listdir(symbol)
            for file in files:
                shutil.copy(symbol + "\\" + file, cwd + "\\all_hist\\" + file)
                print("copied " + file)