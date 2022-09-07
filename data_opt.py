import os
import utils
from DeepLearning import Data
import DeepLearning
import config
import pandas as pd

REPEATS = 10

cwd = os.getcwd()

for i in range(1, 10):

    data = data.TrainingData(cwd, config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT)
    print(data)

    model = DeepLearning.DLModel(data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)

    opt = DeepLearning.Optimization(data, model)
    summary = opt.repeat_train(REPEATS)

    try:
        all_summary = pd.concat([all_summary, summary])
    finally:
        all_summary = summary

    all_summary.reset_index(drop=True, inplace=True)
    utils.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', all_summary, cwd)
