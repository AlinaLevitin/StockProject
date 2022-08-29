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
opt = DeepLearnig.Optimization(model, training_data)
summary = opt.repeat_train(REPEATS, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
model.save_model('testing_model')

utils.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', summary, cwd)


