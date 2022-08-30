import os
import utils
import DeepLearnig
import config

REPEATS = 30

cwd = os.getcwd()

training_data = DeepLearnig.TrainingData(cwd)
training_data.open_all_data()
training_data.split_data()
print(training_data)

model = DeepLearnig.DLModel(cwd)
print(model)
summary = model.train_and_test(training_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE, save=True)
model.save_model('testing_model')

utils.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', summary, cwd)


