"""
DeepLearning API for data processing, training and using the neural network
"""
from .training_data import TrainingData
from .model import Model
from .data import Data
from .neural_network_api import open_data_and_train
from .neural_network_api import predict_results
from .predict_data import PredictData
from .dl_utils import average
from .dl_utils import variance
from .dl_utils import stddev
from .dl_utils import position
from .reports import Reports
from .model_opt import ModelOpt
