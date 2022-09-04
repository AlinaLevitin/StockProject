"""

This file will copy all csv files to one directory in order to reduce time when reading data
Run once.

"""
import utils
import os

cwd = os.getcwd()

utils.copy_to_one_dir(cwd)
