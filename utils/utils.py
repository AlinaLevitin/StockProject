"""
Supporting file containing utility functions
"""

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


def save_to_csv(name: str, data, path: str):
    """
    save to csv file using pandas
    :param name: name for the file
    :param data: the data to save in a form of pandas DataFrame
    :type data: pandas DataFrame
    :param path: path for file saving
    :return: saves the data to csv file
    """
    os.makedirs(path, exist_ok=True)
    os.chdir(path)
    data.to_csv(name + '.csv', index=False)
    print(f'Saved file: {name}.csv in {path} folder')


def copy_to_one_dir(cwd: str):
    """
    copies all csv files in one directory, this saves time when reading the data
    :param cwd: current working directory using cwd = os.getcwd()
    :return: copies all csv files in one directory
    """
    analysis_hist = glob.glob(cwd + "\\analysis_hist")
    date_folders = glob.glob(analysis_hist[0] + "\\*")
    os.makedirs(cwd + "\\all_hist", exist_ok=True)
    for date in date_folders:
        symbol_folders = glob.glob(date + "\\*")
        for symbol in symbol_folders:
            files = os.listdir(symbol)
            for file in files:
                shutil.copy(symbol + "\\" + file, cwd + "\\all_hist\\" + file)
                print("copied " + file)
