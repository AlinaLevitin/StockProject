import os
import methods
from DeepLearnig import data
import DeepLearnig
import config
import pandas as pd

REPEATS = 10

cwd = os.getcwd()

csv = 'data.csv'

data = data.TrainingData(cwd, config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT)
data.save_all_data('data.csv')
print(data)
# methods.save_to_csv(csv, data, cwd)

for i in range(1, 10):
    neurons = 50 * i
    model = DeepLearnig.DLModel(data, neurons, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)

    opt = DeepLearnig.Optimization(data, model)
    summary = opt.repeat_train(REPEATS)

    try:
        all_summary = pd.concat([all_summary, summary])
    except:
        all_summary = summary

    all_summary.reset_index(drop=True, inplace=True)
    methods.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', all_summary, cwd)


