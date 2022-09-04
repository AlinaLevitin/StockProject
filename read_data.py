"""

This file will read the data and save it as csv according to the parameters set in the config.py file
Run every time you want to change the parameters of the data

"""

import os
import DeepLearnig
import config

cwd = os.getcwd()

data = DeepLearnig.TrainingData(cwd)
data.read_all_data(config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT, config.INTERVAL)
data.save_all_data()
