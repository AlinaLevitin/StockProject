"""
main file to read data, train the neural network and predict results
"""
import os

import DeepLearning

CWD = os.getcwd()

DeepLearning.neural_network_api.read_data_for_training(CWD, copy=False)
"""
Use this method when reading the analysis_hist, copy=True will copy all csv files to one folder
for more efficient data reading (do this once then change to copy=False)
this will create a data.csv file with all the data for training.
"""

DeepLearning.neural_network_api.model_opt(CWD)
"""
Use this to optimize the neural network with reduced data volume to 40%.
Optimize layers, activation functions and number or neuron
"""

# DeepLearning.neural_network_api.open_data_and_train(CWD, from_save=False)
"""
Use this to train the neural network either from a callback or saved trained neural network.
* choose from_save=True to continue training from trained_neural_network.h5 file.
* choose from_save=False to train from a callback or create a new neural network (make sure to delete previous 
  callbacks and trained_neural_network.h5).
"""

# DeepLearning.neural_network_api.test_accuracy(CWD)
"""
Use this to test the accuracy of a trained model.
In the config choose the dates you want to test (I suggest no more than two months since it takes 
a long time to read the data)
This will result a csv file named: accuracy_test_{current time and date}.csv in reports folder. 
"""

# DeepLearning.neural_network_api.predict_results(CWD)
"""
Use this to predict results (make sure to copy all desired csv files to predict_data folder)
This will generate a predicted_results.csv file with symbol and suggested position 1/0/-1 for LONG/NO-POSITION/SHORT
"""
