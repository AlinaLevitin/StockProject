import os
import methods
import DeepLearnig
import config

REPEATS = 30

cwd = os.getcwd()

training_data = DeepLearnig.OpenTrainingData(cwd)
print(training_data)

model = DeepLearnig.DLModel(training_data, config.NEURONS, config.EPOCHS, config.LEARNING_RATE, config.BATCH_SIZE)
print(model)
opt = DeepLearnig.Optimization(training_data, model)
summary = opt.repeat_train(REPEATS)

methods.save_to_csv(f'summary_for_{REPEATS}_repeats.csv', summary, cwd)


