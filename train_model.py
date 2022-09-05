"""
Run this for training the neural network
"""
# TODO: train the model
import os
import DeepLearnig
import config


cwd = os.getcwd()

training_data = DeepLearnig.TrainingData(cwd)
training_data.open_all_data()
training_data.split_data()

model = DeepLearnig.Model(cwd)
model.set_model(training_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
model.load_callback()
model.train_and_test(training_data, save=True)
print(model)
model.save_model('trained_neural_network')


