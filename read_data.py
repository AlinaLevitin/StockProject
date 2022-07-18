import os
import DeepLearnig
import config

cwd = os.getcwd()


DeepLearnig.read_all_data(cwd, config.steps_back, config.steps_forward)


