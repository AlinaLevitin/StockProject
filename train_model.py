import os
import methods
from DeepLearnig import Data
import DeepLearnig
import config
import pandas as pd

REPEATS = 10

cwd = os.getcwd()

csv = 'data.csv'

for steps_forward in range(5, 15):
    data = Data.TrainingData(cwd, config.STEPS_BACK, steps_forward, config.PERCENT)
    model = DeepLearnig.DLModel(data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
    acc = []
    for i in range(REPEATS):
        result = model.train_and_test()
        acc.append(result[1])

    accuracy_average = methods.average(acc)
    stdev = methods.stdev(acc)

    summary_dict = {'average accuracy': accuracy_average,
                    'STDEV': stdev,
                    'reapets': REPEATS,
                    'training data': data.train_num,
                    'validation data': data.val_num,
                    'test data': data.test_num,
                    'steps back': config.STEPS_BACK,
                    'steps forward': steps_forward,
                    'epochs': config.EPOCHS,
                    'neurons': config.NEURONS,
                    'learning_rate': config.LEARNING_RATE,
                    }

    summary = pd.DataFrame([summary_dict])
    print(summary)

    try:
        all_summary = pd.concat([all_summary, summary])
    except:
        all_summary = summary

    all_summary.reset_index(drop=True, inplace=True)
    methods.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', all_summary, cwd)


