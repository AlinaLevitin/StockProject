import os
from DeepLearnig import data
import config

cwd = os.getcwd()


data = data.Data(cwd, config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT)

data.save_all_data('data.csv')
