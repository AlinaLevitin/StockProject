# TODO: write the methods open_data_and_train and predict_results

import DeepLearnig
import config


def open_data_and_train(from_save=True):
    training_data = DeepLearnig.TrainingData(config.CWD)
    training_data.open_all_data()
    training_data.split_data(0.2, 0.1)

    model = DeepLearnig.Model(config.CWD)
    model.set_model(training_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)

    if from_save:
        model.load_model('trained_neural_network')
    else:
        model.load_callback()

    model.train_and_test(training_data, save=True)

    model.save_model('trained_neural_network')


def predict_results():
    predict_data = DeepLearnig.PredictData(config.CWD)
    predict_data.read_all_predict_data(config.STEPS_BACK)

    model = DeepLearnig.Model(config.CWD)
    model.set_model(predict_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
    model.load_model('trained_neural_network')
    model.predict_values(predict_data)





