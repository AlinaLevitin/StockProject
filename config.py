# --------------------------------------------------------------

# Configuration parameters for collecting data for DL:

# time points before maximum difference
STEPS_BACK = 21

# time points before maximum difference
STEPS_FORWARD = 20

# the minimum percent for win
PERCENT = 1

# --------------------------------------------------------------

# Configuration for training data:

# Batch size:

BATCH_SIZE = 32

# Number of epochs:
# Best results with 100 epochs
EPOCHS = 100

# Layer dense:
# Best results with 150 neurons

NEURONS = 150

# Learning rate:
# Best results with 10 ** -7

LEARNING_RATE = 10 ** -7
