"""

"""
import os
import DeepLearnig
import config


cwd = os.getcwd()

training_data = DeepLearnig.TrainingData(cwd)
training_data.open_all_data()
training_data.split_data()
print(training_data)

model = DeepLearnig.Model(cwd)
print(model)
model.train_and_test(training_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE, save=True)
model.save_model('testing_model')


