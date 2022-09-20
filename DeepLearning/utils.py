"""
Supporting file containing utility functions
"""
import os
from _csv import reader

import pandas as pd
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
    new_folder = cwd + "\\all_hist"
    os.makedirs(new_folder, exist_ok=True)
    existing_files = os.listdir(new_folder)
    for date in date_folders:
        symbol_folders = glob.glob(date + "\\*")
        for symbol in symbol_folders:
            files = os.listdir(symbol)
            for file in files:
                if file in existing_files:
                    print(f'{file} already exists in "all_hist" folder')
                else:
                    shutil.copy(symbol + "\\" + file, cwd + "\\all_hist\\" + file)
                    print(f'copied {file} to "all_hist" folder')


def read_config(cwd: str) -> dict:
    """
    reading the config file from config.csv

    :param cwd: current working directory the config file should be present here
    :return: dictionary with all parameters from the config
    """
    os.chdir(cwd)
    config_list = []
    with open('config.csv') as file:
        csv_reader = reader(file)
        next(csv_reader)
        for row in csv_reader:
            config_list.append(row)
    config = {item[0]: item[1] for item in config_list}
    return config

