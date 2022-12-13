# +
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm import tqdm
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
from collections import OrderedDict
import matplotlib.pyplot as plt
from Network import *
from DataSet import *
from train_procedure import *
from Aug import *
from torchvision import transforms
import pickle

# %load_ext autoreload
# %autoreload 2
# -

# Create the augmented dataset
create_augmented_dataset(True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ## Experiment 1

# +
dict_train1 ={"cross_val": True,
            "skip_connection": True,
            "num_epochs": 5,
            "n_splits": 3,
            "batch_size": 10,
            "dict_double_conv": {"BatchNorm": True,
                "activation": nn.ReLU(inplace=True),
                "p_dropout": 0.2,
                "use_dropout": False,
                "bias": False},
            "dict_ups": {"BatchNorm": False,
                "p_dropout": 0.2,
                "use_dropout": False,
                "bias": False},
            "loss": nn.BCEWithLogitsLoss(),
            "optimizer": optim.Adam,
            "param_optimizer": {"weight_decay": None,
                               "lr": 1e-04},
            "use_scheduler": True,
            "type_scheduler": "StepLR",
            "scheduler": torch.optim.lr_scheduler.StepLR,
            "param_scheduler": {"step_size": 4,
                               "gamma": 0.1},
             "scaler": torch.cuda.amp.GradScaler(),
             "device": DEVICE
            }


experiment1 = {"param": dict_train1}
experiment1["convergence_path"] = run_training(dict_train1) 


f = open("result_exp/experiment1.pkl","wb")
pickle.dump(experiment1,f)
f.close()
# -

# ## Experiment 2

# +
dict_train2 ={"cross_val": True,
            "skip_connection": False,
            "num_epochs": 5,
            "n_splits": 3,
            "batch_size": 10,
            "dict_double_conv": {"BatchNorm": True,
                "activation": nn.ReLU(inplace=True),
                "p_dropout": 0.2,
                "use_dropout": False,
                "bias": False},
            "dict_ups": {"BatchNorm": False,
                "p_dropout": 0.2,
                "use_dropout": False,
                "bias": False},
            "loss": nn.BCEWithLogitsLoss(),
            "optimizer": optim.Adam,
            "param_optimizer": {"weight_decay": None,
                               "lr": 1e-04},
            "use_scheduler": True,
            "type_scheduler": "StepLR",
            "scheduler": torch.optim.lr_scheduler.StepLR,
            "param_scheduler": {"step_size": 4,
                               "gamma": 0.1},
             "scaler": torch.cuda.amp.GradScaler(),
             "device": DEVICE
            }


experiment2 = {"param": dict_train2}
experiment2["convergence_path"] = run_training(dict_train2) 


f = open("result_exp/experiment2.pkl","wb")
pickle.dump(experiment2,f)
f.close()
# -


