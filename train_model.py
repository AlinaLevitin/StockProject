import os
import methods
from DeepLearnig import Data
import DeepLearnig
import config
import pandas as pd

REPEATS = 10

cwd = os.getcwd()

csv = 'data.csv'

for steps_back in range(30):
    data = Data.TrainingData(cwd, steps_back, config.STEPS_FORWARD, config.PERCENT)
    model = DeepLearnig.DLModel(data.x_train,
                                data.y_train,
                                data.x_val,
                                data.y_val,
                                data.x_test,
                                data.y_test,
                                config.NEURONS,
                                config.EPOCHS,
                                config.LEARNING_RATE,
                                config.BATCH_SIZE
                                )
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
                    'steps back': steps_back,
                    'steps forward': config.STEPS_FORWARD,
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


