"""
DeepLearning API for data processing, training and using the neural network
"""

from .data import Data
from .training_data import TrainingData
from .predict_data import PredictData
from .model import Model
from .model_opt import ModelOpt
from .reports import Reports

from .dl_utils import average
from .dl_utils import variance
from .dl_utils import stddev
from .dl_utils import position

from .utils import open_csv
from .utils import save_to_csv
from .utils import copy_to_one_dir
from .utils import read_config

from .neural_network_api import open_data_and_train
from .neural_network_api import predict_results




