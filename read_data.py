import os
import DeepLearnig
import config
import methods

cwd = os.getcwd()


data = DeepLearnig.read_all_data(cwd, config.STEPS_BACK, config.STEPS_FORWARD)

methods.save_to_csv('data.csv', data, cwd)
