"""
API to read the data, train the neural network and predict results
"""
import os
import time

import pandas as pd

import DeepLearning


def read_data_for_training(cwd, copy=False):
    """
    Use this method when reading the analysis_hist, copy=True will copy all csv files to one folder
    for more efficient data reading (do this once then change to copy=False)
    this will create a data.csv file with all the data for training.

    :param cwd: current working directory where main.py is
    :param copy: bool optional default is False, copy=True will copy all csv files from analysis_hist
    to all_hist folder
    """
    os.chdir(cwd)
    config_dict = DeepLearning.utils.read_config(cwd)
    steps_back = int(config_dict['STEPS_BACK'])
    steps_forward = int(config_dict['STEPS_FORWARD'])
    percent_long = float(config_dict['PERCENT_LONG'])
    percent_short = float(config_dict['PERCENT_SHORT'])
    interval = int(config_dict['INTERVAL'])
    start_date = int(config_dict['START_DATE'])
    end_date = int(config_dict['END_DATE'])

    if copy:
        DeepLearning.utils.copy_to_one_dir(cwd)
    data = DeepLearning.TrainingData(cwd)
    data.read_all_data(steps_back, steps_forward,
                       percent_long, percent_short, interval, start_date, end_date)
    data.save_all_data('train_data')


def model_opt(cwd):
    """
    Use this to optimize the neural network with reduced data volume to 20%.
    Optimize layers, activation functions and number or neuron

    :param cwd: current working directory where main.py is

    :return: csv files with reports for the optimization in model_opt folder.
    * summary for all the runs (repeat_summary)
    * summary for number of repeats (summary_#_repeats)
    * accuracy and loss at every run (acc_and_loss)
    """
    os.chdir(cwd)
    config_dict = DeepLearning.utils.read_config(cwd)
    test_data_portion = float(config_dict['TEST_DATA'])
    val_data_portion = float(config_dict['VALIDATION_DATA'])

    training_data = DeepLearning.TrainingData(cwd)
    training_data.open_all_data('train_data')
    training_data.reduce_data(0.4)
    training_data.split_data(test_data_portion, val_data_portion)
    print("=" * 60)
    #
    REPEATS = 1
    #
    all_summary = pd.DataFrame()
    print(f'Commencing model optimization for {REPEATS} repeats')
    print("=" * 60)
    for i in range(1, 5):
        epochs = int(config_dict['EPOCHS'])
        learning_rate = float(config_dict['LEARNING_RATE'])
        batch_size = int(config_dict['BATCH_SIZE'])
        neurons = 2000 * i
        repeat_train = DeepLearning.ModelOpt(cwd)
        result = repeat_train.repeat_train(training_data, REPEATS, neurons, epochs, learning_rate, batch_size)
        all_summary = pd.concat([all_summary, result])

    gmt = time.gmtime()
    all_summary_file = f'repeat_summary_{gmt[0]}_{gmt[1]}_{gmt[2]}_{gmt[3]}_{gmt[4]}'
    DeepLearning.utils.save_to_csv(f'{all_summary_file}', all_summary, cwd + "\\model_opt")
    print("=" * 60)
    print(all_summary)


def open_data_and_train(cwd, from_save=True):
    """
    Use this to train the neural network either from a callback or saved trained neural network.
    Training data should be saved to data.csv file
    * choose from_save=True to continue training from trained_neural_network.h5 file.
    * choose from_save=False to train from a callback or create a new neural network (make sure to delete previous
    callbacks and trained_neural_network.h5).

    :param cwd: current working directory where main.py is
    :param from_save: bool optional from_save=True to continue training from trained_neural_network.h5 file
                    from_save=False to train from a callback or create a new neural network
                    (make sure to delete previous callbacks and trained_neural_network.h5)
    """
    os.chdir(cwd)
    config_dict = DeepLearning.utils.read_config(cwd)
    test_data_portion = float(config_dict['TEST_DATA'])
    val_data_portion = float(config_dict['VALIDATION_DATA'])

    training_data = DeepLearning.TrainingData(cwd)
    training_data.open_all_data('train_data')
    training_data.split_data(test_data_portion, val_data_portion)

    model = DeepLearning.Model(cwd)
    neurons = int(config_dict['NEURONS'])
    epochs = int(config_dict['EPOCHS'])
    learning_rate = float(config_dict['LEARNING_RATE'])
    batch_size = int(config_dict['BATCH_SIZE'])

    model.set_model(training_data, neurons, epochs, learning_rate, batch_size)

    if from_save:
        try:
            print('=' * 60)
            model.load_model('trained_neural_network')
            print('=' * 60)
        except (Exception,):
            raise FileNotFoundError('Unable to load trained neural network,'
                                    'please please delete previous callback or trained_neural_network.h5'
                                    'and choose from_save=False')
    else:
        try:
            print('=' * 60)
            model.load_callback()
            print('=' * 60)
        except (Exception,):
            print('=' * 60)
            print('Unable to load callback, training is re-initialized')
            print('=' * 60)

    print(model)
    print('=' * 60)
    print('Training is commencing!')
    print('=' * 60)
    try:
        result = model.train_and_test(training_data)
        accuracy = result[0]
        acc_and_loss = result[1]
        report = DeepLearning.Reports(cwd, training_data, model)
        report.single_train_report(accuracy, acc_and_loss)
        model.save_model('trained_neural_network')
    except (Exception,):
        raise FileExistsError('Unable to start training,'
                              'please delete previous callback or trained_neural_network.h5 and retry')


def predict_results(cwd):
    """
    Use this to predict results (make sure to copy all desired csv files to predict_data folder)
    This will generate a predicted_results.csv file with symbol and suggested position 1/0/-1 for LONG/NO-POSITION/SHORT

    :param cwd: current working directory where main.py is
    """
    os.chdir(cwd)
    config_dict = DeepLearning.utils.read_config(cwd)
    steps_back = int(config_dict['STEPS_BACK'])

    predict_data = DeepLearning.PredictData(cwd)
    predict_x = predict_data.read_all_predict_data(steps_back)

    model = DeepLearning.Model(cwd)
    print('=' * 60)
    model.load_model('trained_neural_network')
    print('=' * 60)
    result = model.predict_values(predict_data).round(0)
    predict_y = pd.DataFrame(result, columns=['A_long_(y)', 'B_long_(y)', 'A_short_(y)', 'B_short_(y)'])
    index = predict_x.index
    predict_y.set_index(index, inplace=True)
    predict = pd.DataFrame()
    for index, row in predict_y.iterrows():
        A = str(index).split(" ")[0]
        B = str(index).split(" ")[1]
        new_row = pd.DataFrame({'symbol': A, 'position': DeepLearning.dl_utils.position(row['A_long_(y)'],
                                                                                        row['A_short_(y)'])}, index=[0])
        predict = pd.concat([predict, new_row])
        new_row = pd.DataFrame({'symbol': B, 'position': DeepLearning.dl_utils.position(row['B_long_(y)'],
                                                                                        row['B_short_(y)'])}, index=[0])
        predict = pd.concat([predict, new_row])
    predict.reset_index(drop=True, inplace=True)
    DeepLearning.utils.save_to_csv('predicted_results', predict, cwd)
