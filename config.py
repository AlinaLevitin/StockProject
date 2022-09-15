"""
Configuration file for data collection/processing and training the neural network
"""
# TODO reduce steps forward and percent
import os

CWD = os.getcwd()

# =================================================================
"""
Configuration parameters for collecting training data for training the neural network:
-------------------------------------------------------------------

If you change any of these parameters you need to read data again and re-train the model

1. make desired changes
2. use DeepLearning.NeuralNetwork_API.read_data_for_training(copy=True)
3. delete callback folder and trained_neural_network.h5 file
4. use DeepLearning.NeuralNetwork_API.open_data_and_train(from_save=False)
"""

# time points before maximum difference
STEPS_BACK = 21

# time points before maximum difference
STEPS_FORWARD = 20

# the minimum percent for long win
PERCENT_LONG = 1

# the minimum percent for short win
PERCENT_SHORT = 1

# Interval for collecting data
INTERVAL = 10

# =================================================================
"""
Configuration for training data portion split
-------------------------------------------------------------------

If you change any of these parameters you need to re-train the model

1. make desired changes
2. delete callback folder and trained_neural_network.h5 file
3. use DeepLearning.NeuralNetwork_API.open_data_and_train(from_save=False)
"""
# Training data split

TEST_DATA = 0.2

VALIDATION_DATA = 0.1

# =================================================================
"""
Configuration for training neural network architecture:
-------------------------------------------------------------------

If you change any of these parameters you need to re-train the model

1. make desired changes
2. if number of neurons are changed please delete callback folder and trained_neural_network.h5 file
3. use DeepLearning.NeuralNetwork_API.open_data_and_train(from_save=False)
"""
# Layer dense:
NEURONS = 500
"""
Changing the neurons will affect the architecture of the model, therefore, please delete old callback folder
and trained_neural_network.h5 file before training the model
"""
# =================================================================
"""
Configuration for stochastic gradient descent:
-------------------------------------------------------------------

Changing the learning rate will restart the optimizer but training can be resumed
Feel free to change batch size and epochs as needed

"""
# Batch size:
BATCH_SIZE = 256

# Number of epochs:
EPOCHS = 100

# Learning rate:
LEARNING_RATE = 10 ** -2
