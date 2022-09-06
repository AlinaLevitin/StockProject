"""
Configuration file for data collection/processing and training the neural network
"""
import os

CWD = os.getcwd()

# --------------------------------------------------------------

# Configuration parameters for collecting data for DL:

# time points before maximum difference
STEPS_BACK = 21

# time points before maximum difference
STEPS_FORWARD = 20

# the minimum percent for win
PERCENT = 1

# Interval for collecting data
INTERVAL = 10

# --------------------------------------------------------------
# Training data split

TEST_DATA = 0.2

VALIDATION_DATA = 0.1

# --------------------------------------------------------------

# Configuration for training data:

# Batch size:
BATCH_SIZE = 32

# Number of epochs:
EPOCHS = 5

# Layer dense:
NEURONS = 100

# Learning rate:
LEARNING_RATE = 10 ** -5
