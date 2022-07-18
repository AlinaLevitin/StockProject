import os
from DeepLearnig import read_data_for_DL
import pandas as pd
import config
import glob

cwd = os.getcwd()


read_data_for_DL.read_all_data(cwd, config.steps_back, config.steps_forward)


