import os
import DeepLearnig
import config

cwd = os.getcwd()


data = DeepLearnig.TrainingData(cwd)
data.read_all_data(config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT, config.INTERVAL)
data.save_all_data()
print(data)
