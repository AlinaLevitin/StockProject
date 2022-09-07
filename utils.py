"""
Supporting file containing utility functions
"""
# TODO: document position method

import math
import pandas as pd
import os
import glob
import shutil


def open_csv(cwd: str, file_name: str):
    """
    open a csv using pandas

    :param cwd: current working directory using cwd = os.getcwd()
    :param file_name: the name of the file
    :return: DataFrame
    """
    os.chdir(cwd)
    df = pd.read_csv(file_name)
    return df


def save_to_csv(name: str, data, cwd: str):
    """
    save to csv file using pandas

    :param name: name for the file
    :param data: the data to save in a form of pandas DataFrame
    :type data: pandas DataFrame
    :param cwd: current working directory using cwd = os.getcwd()
    :return: saves the data to csv file
    """
    os.chdir(cwd)
    data.to_csv(name + '.csv', index=False)


def average(num: list) -> float:
    """
    calculates the average of numbers in a list
    :param num: list of numbers
    :return: average of numbers
    """
    if isinstance(num, list):
        avg = sum(num) / len(num)
        return avg
    else:
        return num


def variance(num: list) -> float:
    """
    calculates variance of numbers in a list
    :param num: list of numbers
    :return: variance of numbers
    """
    if isinstance(num, list):
        n = len(num)
        var = sum((x - average(num)) ** 2 for x in num) / n
        return var
    else:
        return 0


def stddev(num: list) -> float:
    """
    calculates the standard deviation of numbers in a list
    :param num: list of numbers
    :return: standard deviation of numbers
    """
    if isinstance(num, list):
        var = variance(num)
        std_dev = math.sqrt(var)
        return std_dev
    else:
        return 0


def copy_to_one_dir(cwd: str):
    """
    copies all csv files in one directory, this saves time when reading the data
    :param cwd: current working directory using cwd = os.getcwd()
    :return: copies all csv files in one directory
    """
    analysis_hist = glob.glob(cwd + "\\analysis_hist")
    date_folders = glob.glob(analysis_hist[0] + "\\*")
    for date in date_folders:
        symbol_folders = glob.glob(date + "\\*")
        for symbol in symbol_folders:
            files = os.listdir(symbol)
            for file in files:
                shutil.copy(symbol + "\\" + file, cwd + "\\all_hist\\" + file)
                print("copied " + file)


def position(long, short):
    if long == 1:
        return 1
    elif short == 1:
        return -1
    elif long == 0 and short == 0:
        return 0
