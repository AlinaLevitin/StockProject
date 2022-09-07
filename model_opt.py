"""

"""
import os
import utils
import DeepLearning
import config
import pandas as pd

REPEATS = 100

cwd = os.getcwd()

training_data = DeepLearning.TrainingData(cwd)
training_data.open_all_data()
training_data.split_data()
print(training_data)

all_summary = pd.DataFrame()

for i in range(1, 5):
    neurons = 50 * i
    model = DeepLearning.DLModel(cwd)
    opt = DeepLearning.Optimization(model, training_data)
    summary = opt.repeat_train(REPEATS, neurons, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
    print(model)
    all_summary = pd.concat([all_summary, summary])

    all_summary.reset_index(drop=True, inplace=True)
    utils.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', all_summary, cwd)


