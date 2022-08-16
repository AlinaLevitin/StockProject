import os
import methods
from DeepLearnig import data
import DeepLearnig
import config
import pandas as pd

REPEATS = 100

cwd = os.getcwd()

csv = 'data.csv'

data = data.TrainingData(cwd, config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT)
data.save_all_data('data.csv')
print(data)

model = DeepLearnig.DLModel(data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)

opt = DeepLearnig.Optimization(data, model)
summary = opt.repeat_train(REPEATS)

methods.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', summary, cwd)


