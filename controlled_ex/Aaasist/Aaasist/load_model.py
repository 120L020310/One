import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", category=FutureWarning)

from .models.AASIST import Model

def get_model(config_name):
    """Define DNN model architecture"""
    
    assert config_name in ['AASIST-L', 'AASIST']
    cur_path = os.path.split(__file__)[0]
    with open(f"{cur_path}/config/{config_name}.conf", "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]


    model = Model(model_config)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model
    