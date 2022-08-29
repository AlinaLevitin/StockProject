import os
import utils
import config
import DeepLearnig

cwd = os.getcwd()

model = DeepLearnig.DLModel(cwd)

model.load_model()


