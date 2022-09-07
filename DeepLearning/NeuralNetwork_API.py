# TODO: document the methods

import DeepLearning
import config
import pandas as pd

import utils


def read_data_for_training(copy=False):
    if copy:
        utils.copy_to_one_dir(config.CWD)
    data = DeepLearning.TrainingData(config.CWD)
    data.read_all_data(config.STEPS_BACK, config.STEPS_FORWARD, config.PERCENT, config.INTERVAL)
    data.save_all_data('train_data')


def open_data_and_train(from_save=True):
    training_data = DeepLearning.TrainingData(config.CWD)
    training_data.open_all_data()
    training_data.split_data(config.TEST_DATA, config.VALIDATION_DATA)

    model = DeepLearning.Model(config.CWD)

    if from_save:
        try:
            model.load_model('trained_neural_network')
        except (Exception,):
            print('Unable to load trained neural network, training is re-initialized')
            model.train_and_test(training_data)
    else:
        try:
            model.set_model(training_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
            model.load_callback()
        except (Exception,):
            print('Unable to load callback, training is re-initialized')

    print(model)
    try:
        model.train_and_test(training_data)
        model.save_model('trained_neural_network')
    except (Exception,):
        raise ValueError('Unable to start training'
                         ', please delete previous callback or trained_neural_network.h5 and retry')


def predict_results():
    predict_data = DeepLearning.PredictData(config.CWD)
    predict_x = predict_data.read_all_predict_data(config.STEPS_BACK)

    model = DeepLearning.Model(config.CWD)
    model.set_model(predict_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
    model.load_model('trained_neural_network')
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
    print(predict)

