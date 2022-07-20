import os
import DeepLearnig
import config
import methods

cwd = os.getcwd()


data = DeepLearnig.Data(cwd, config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT)

methods.save_to_csv('data.csv', data.data, cwd)
