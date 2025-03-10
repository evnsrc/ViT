'''
Train Validate Evaluation Script
'''

import os
import sys
import re
from glob import glob
import time
import numpy as np
import pandas as pd

### TORCH ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import kl_div
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn.conv import PointConv
##############

import logging
import open3d as o3
import yaml
import argparse



### IN-HOUSE CODES ###
from src.utils.train import Trainer
from src.models.VisualTransformerEncoder import *
from src.models.VisualTransformerDecoder import *
from src.models.MultiHeadAttentionBlock import *
from src.models.VisualTransformerGenerator import *
from src.utils import features, utils
from src.data.dataset import DataMNIST

#######################

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import wandb

### HERE ARGS ###
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='config/config.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

#################

### REPRODUCIBILITY ###
torch.seed = config['trainer_parameters']['manual_seed']
#######################

#### W&B INIT ###
#wandb.init(
#    # set the wandb project where this run will be logged
#    project="PointNet-VAE",
#    
#    # track hyperparameters and run metadata
#    config=config
#)
###############



DEVICE = 'cuda'
print('Device is set to :{}'.format(DEVICE))
print(torch.cuda.is_available())    

# model hyperparameters
EPOCHS = config['trainer_parameters']['epochs']
LR = config['trainer_parameters']['lr']
LATENT_DIM = config['model_parameters']['latent_dim']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_HEADS = config['model_parameters']['num_heads']
N_LAYERS = config['model_parameters']['num_layers']
DFFN = config['model_parameters']['d_ffn']
DROPOUT = config['trainer_parameters']['dropout']
NPATCHES = config['model_parameters']['n_patches']

#### LOAD DATA ####
data = DataMNIST(**config["data_parameters"])
data.setup()
train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()
###################

print(f"Utilisation de {DEVICE}")
transformer = Transformer(LATENT_DIM, N_HEADS, N_LAYERS, DFFN, DROPOUT, NPATCHES).to(DEVICE)
model = transformer
model.load_state_dict(torch.load("weights.pth", weights_only=True), strict=False)

model_run = Trainer(model, 
                    criterion=nn.CrossEntropyLoss(),
                    optimizer=optim.Adam(transformer.parameters(), config['trainer_parameters']['lr']),
                    **config['model_parameters']
                    )

tr_LOSS,val_LOSS=append(model_run.fit(train_dataloader, val_dataloader, EPOCHS))

#save the model
torch.save(transformer.state_dict(), 'weights.pth')












