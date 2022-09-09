"""
main file to read data, train the neural network and predict results
"""

import DeepLearning

# DeepLearning.neural_network_api.read_data_for_training(copy=False)
"""
Use this method when reading the analysis_hist, copy=True will copy all csv files to one folder
for more efficient data reading (do this once then change to copy=False)
this will create a data.csv file with all the data for training.
"""

DeepLearning.neural_network_api.open_data_and_train(from_save=False)
"""
Use this to train the neural network either from a callback or saved trained neural network.
* choose from_save=True to continue training from trained_neural_network.h5 file.
* choose from_save=False to train from a callback or create a new neural network (make sure to delete previous 
  callbacks and trained_neural_network.h5).
"""

# DeepLearning.neural_network_api.predict_results()
"""
Use this to predict results (make sure to copy all desired csv files to predict_data folder)
This will generate a predicted_results.csv file with symbol and suggested position 1/0/-1 for LONG/NO-POSITION/SHORT
"""
