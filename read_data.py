import os
from DeepLearnig import Data
import config
import methods

cwd = os.getcwd()


data = Data.Data(cwd, config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT)

methods.save_to_csv('data.csv', data.data, cwd)
