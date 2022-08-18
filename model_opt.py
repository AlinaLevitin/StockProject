import os
import methods
import DeepLearnig
import config
import pandas as pd

REPEATS = 100

cwd = os.getcwd()

training_data = DeepLearnig.TrainingData(cwd)
training_data.open_all_data()
training_data.split_data()
print(training_data)


for i in range(1, 5):
    neurons = 50 * i
    model = DeepLearnig.DLModel(training_data, neurons, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
    print(model)
    opt = DeepLearnig.Optimization(training_data, model)
    summary = opt.repeat_train(REPEATS)

    try:
        all_summary = pd.concat([all_summary, summary])
    except:
        all_summary = summary

    all_summary.reset_index(drop=True, inplace=True)
    methods.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', all_summary, cwd)


