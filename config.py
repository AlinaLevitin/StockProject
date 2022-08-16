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
# Best results with batch size of 56, 64, 88 and 104
BATCH_SIZE = 64

# Number of epochs:
# Best results with 150 epochs
EPOCHS = 150

# Layer dense:
# Best results with 350 neurons

NEURONS = 150

# Learning rate:
# Best results with 10 ** -7

LEARNING_RATE = 10 ** -7
