#Imports
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randint
import random 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import random_split
from optuna import Trial, create_study, samplers
from optuna.logging import set_verbosity, DEBUG
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import json
from joblib import Parallel, delayed
import os

#Class: RNN
class RNN(nn.Module):
    #sizes are the number of nodes for respective layers, as ints
    #input_size should be the numbers of variables in input, output_size the number of variables predicted
    #hidden_size should be chosen after experimentation, since we only have one layer it should be more than the number of variables
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, **kwds):
        super().__init__(**kwds)
        #Number of nodes of the hidden layer (used for init)
        self.hidden_size = hidden_size
        #Weights
        self.i2h = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)
    
    #Input must be torch.Tensor and normalized
    def forward(self, input):
        #h0 = self.initHidden()
        h_list, hn = self.i2h(input)
        output_list = self.h2o(h_list)
        return output_list

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

#Parameters to get several scenari
TRAINING_NUMBER = 400
USE_NOISY_DATA = False
USE_MIXED_EFFECT = False
RANDOM_TIME = True
MODEL = RNN        
RE = 'Mixed' if USE_MIXED_EFFECT else 'Fixed'
Time = 'random time' if RANDOM_TIME else 'regular time'

#Imports and preprocessing
CSV_FILE = "../../data/synthetic_bph_1/Simulations" + "/simulation1" + ".csv"
CSV_RES_R = "../../data/synthetic_bph_1/Résultats " + Time
CSV_FILE = os.path.join(os.path.dirname(__file__), CSV_FILE)
CSV_RES_R = os.path.join(os.path.dirname(__file__), CSV_RES_R)
#data loading
data = pd.read_csv(CSV_FILE, sep=";", decimal=",")
predictions_R_test = pd.read_csv(CSV_RES_R + "/Predictions.csv", sep=",", decimal=".")

x_labels = [
    c for c in data.columns if c.startswith("x") and ((("_" in c) is USE_NOISY_DATA and ('obs' in c) is USE_NOISY_DATA))
]
if 'x8' not in x_labels:
    x_labels.append('x8')
#assert len(x_labels) == 8

y_labels = [
    c
    for c in data.columns
    if c.startswith("y")
    and (("_obs" in c))
    and (("_mixed" in c) is USE_MIXED_EFFECT)
]
assert len(y_labels) == 1

data_norm = data.copy()
data_norm = data_norm.dropna()
N_train = random.sample(range(1,501), TRAINING_NUMBER)
N_train.sort()
N_val =  [x for x in range(1,501) if x not in N_train]
data_train = data_norm.loc[data_norm['individus'].isin(N_train)]
data_val = data_norm[~data_norm['individus'].isin(N_train)]

scaler_x = RobustScaler()
data_train.loc[:,x_labels[:-1]] = scaler_x.fit_transform(data_train[x_labels[:-1]])
data_val.loc[:,x_labels[:-1]] = scaler_x.transform(data_val[x_labels[:-1]])

scaler_y = RobustScaler()
data_train.loc[:,y_labels] = scaler_y.fit_transform(data_train[y_labels])
data_val.loc[:,y_labels] = scaler_y.transform(data_val[y_labels])

groupby = data_train.groupby('individus')[x_labels].apply(np.array)
input_train = [torch.Tensor(x) for x in groupby]
input_train = torch.stack(input_train)
groupby = data_train.groupby('individus')[y_labels].apply(np.array)
target_train = [torch.Tensor(x) for x in groupby]
target_train = torch.stack(target_train)

groupby = data_val.groupby('individus')[x_labels].apply(np.array)
input_val = [torch.Tensor(x) for x in groupby]
input_val = torch.stack(input_val)
groupby = data_val.groupby('individus')[y_labels].apply(np.array)
target_val = [torch.Tensor(x) for x in groupby]
target_val = torch.stack(target_val)

criterion = torch.nn.MSELoss()
epoch = 20000
eps = 0.0005

def objective(trial):
    h = trial.suggest_int('hidden_size', 10, 50)
    num_layers = trial.suggest_int('nombre de couches', 1, 3)
    lr = trial.suggest_float('learning rate', 1e-4, 1e-1, log=True)

    #training
    model = MODEL(input_size = 8, hidden_size=h, output_size=1, num_layers=2)
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
    loss_val = [float('inf')]*100
    loss_train = []
    cur_loss_val = 0
    nb_epochs = 0

    while abs(np.max(loss_val[-100:] - np.min(loss_val[-100:] + [cur_loss_val]))) >= eps and loss_val[-100] > cur_loss_val and nb_epochs < epoch:

        y_val = model(input = input_val)
        cur_loss_val = criterion(y_val, target_val).item()
        loss_val.append(cur_loss_val)

        optimizer.zero_grad()
        y_pred = model(input=input_train)
        loss = criterion(y_pred, target_train)
        loss.backward()
        optimizer.step()

        cur_loss_train = loss.item()
        loss_train.append(cur_loss_train)
        
        nb_epochs += 1

    loss_val = loss_val[101:]
    return cur_loss_val

N_STARTUP_TRIALS = 20
N_TPE_TRIALS = 400

study = create_study(direction='minimize')
study.sampler = samplers.TPESampler(n_startup_trials=N_STARTUP_TRIALS, seed=0)
study.optimize(objective, n_trials=N_STARTUP_TRIALS + N_TPE_TRIALS)
print("Best Hyperparameters:", study.best_params)

with open('hyperparameters_ODE_' + RE + '.json', 'w') as fp:
    json.dump(study.best_params, fp)