# TODO: document the methods
"""
API to read the data, train the neural network and predict results
"""
import DeepLearning
import config
import pandas as pd
import utils


def read_data_for_training(copy=False):
    """
    Use this method when reading the analysis_hist, copy=True will copy all csv files to one folder
    for more efficient data reading (do this once then change to copy=False)
    this will create a data.csv file with all the data for training.

    :param copy: bool optional default is False, copy=True will copy all csv files from analysis_hist
    to all_hist folder
    """
    if copy:
        utils.copy_to_one_dir(config.CWD)
    data = DeepLearning.TrainingData(config.CWD)
    data.read_all_data(config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT, config.INTERVAL)
    data.save_all_data('train_data')


def open_data_and_train(from_save=True):
    """
    Use this to train the neural network either from a callback or saved trained neural network.
    * choose from_save=True to continue training from trained_neural_network.h5 file.
    * choose from_save=False to train from a callback or create a new neural network (make sure to delete previous
    callbacks and trained_neural_network.h5).

    :param from_save: bool optional from_save=True to continue training from trained_neural_network.h5 file
                    from_save=False to train from a callback or create a new neural network
                    (make sure to delete previous callbacks and trained_neural_network.h5)
    """
    training_data = DeepLearning.TrainingData(config.CWD)
    training_data.open_all_data()
    training_data.split_data(config.TEST_DATA, config.VALIDATION_DATA)

    model = DeepLearning.Model(config.CWD)

    if from_save:
        try:
            print('=' * 60)
            model.load_model('trained_neural_network')
            print('=' * 60)
        except (Exception,):
            raise FileNotFoundError('Unable to load trained neural network, please choose from_save=False')
    else:
        try:
            print('=' * 60)
            model.set_model(training_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
            model.load_callback()
            print('=' * 60)
        except (Exception,):
            print('=' * 60)
            print('Unable to load callback, training is re-initialized')
            print('=' * 60)

    print(model)
    try:
        model.train_and_test(training_data)
        model.save_model('trained_neural_network')
    except (Exception,):
        raise FileExistsError('Unable to start training'
                              ', please delete previous callback or trained_neural_network.h5 and retry')


def predict_results():
    """
    Use this to predict results (make sure to copy all desired csv files in predict_data folder)
    This will generate a predicted_results.csv file
    """
    predict_data = DeepLearning.PredictData(config.CWD)
    predict_x = predict_data.read_all_predict_data(config.STEPS_BACK)

    model = DeepLearning.Model(config.CWD)
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
        new_row = pd.DataFrame({'symbol': A, 'position': utils.position(row['A_long_(y)'],
                                                                        row['A_short_(y)'])}, index=[0])
        predict = pd.concat([predict, new_row])
        new_row = pd.DataFrame({'symbol': B, 'position': utils.position(row['B_long_(y)'],
                                                                        row['B_short_(y)'])}, index=[0])
        predict = pd.concat([predict, new_row])
    predict.reset_index(drop=True, inplace=True)
    utils.save_to_csv('predicted_results', predict, config.CWD)
    print('predicted results saved to predicted_results.csv file')