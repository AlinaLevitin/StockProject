"""
Configuration file for data collection/processing and training the neural network
"""

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

# Configuration for training data:

# Batch size:
BATCH_SIZE = 32

# Number of epochs:
EPOCHS = 250

# Layer dense:
NEURONS = 300

# Learning rate:
LEARNING_RATE = 10 ** -8
