import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import randint
from numpy.typing import NDArray
import pandas as pd
from random import SystemRandom
from sklearn import model_selection
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
import random

import torch
import torch.nn as nn
import torch.optim as optim
from optuna import Trial, create_study, samplers
from optuna.logging import set_verbosity, DEBUG

import utils
from utils import compute_loss_all_batches

from ode_rnn import *
from ode_func import ODEFunc
from diffeq_solver import DiffeqSolver
from joblib import Parallel, delayed
import os
import json

#Seeding
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
#Settings
USE_NOISY_DATA = False
USE_MIXED_EFFECT = False
RANDOM_TIME = True
RE = 'Mixed' if USE_MIXED_EFFECT else 'Fixed'
Time = 'Random time' if RANDOM_TIME else 'Regular time'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Model parameters
obsrv_std = 0.01
input_dim = 8
output_dim = 1
lr = 1e-2
niters = 5000
eps = 0.0005
train_dict={}
val_dict={}

CSV_FILE = "../../../data/synthetic_bph_1/Simulations/simulation1" + ".csv"
CSV_FILE = os.path.join(os.path.dirname(__file__), CSV_FILE)
data = pd.read_csv(CSV_FILE, sep=";", decimal=",")
#Variables used
x_labels = [
    c for c in data.columns if c.startswith("x") and ((("_" in c) is USE_NOISY_DATA and ('obs' in c) is USE_NOISY_DATA))
]
if 'x8' not in x_labels:
    x_labels.append('x8')
y_labels = [
    c
    for c in data.columns
    if c.startswith("y")
    and (("_obs" in c))
    and (("_mixed" in c) is USE_MIXED_EFFECT)
]
assert len(y_labels) == 1
#preprocessing
data_norm = data.copy()
data_norm = data_norm.dropna()
N_train = random.sample(range(1,501), 400)
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

groupby = data_train.groupby('individus')['temps'].apply(np.array)
observed_tp= [torch.Tensor(x) for x in groupby]
observed_tp = torch.stack(observed_tp)

train_dict["tp_to_predict"] = observed_tp[0] #The time point you want to make a prediction at, it only works at the observed time points
train_dict["observed_data"] = input_train #X
train_dict["observed_tp"] = observed_tp[0] #Time of observations
train_dict["data_to_predict"] = target_train #Y
train_dict["mode"] = None #interpolation or extrapolation, extrapolation isn't implemented
train_dict['labels'] = None #Used for classification
train_dict["observed_mask"] =  torch.ones(input_train.shape) #matrix of the same size as inputs, 0 for the inputs you want to ignore, 1 for the others
train_dict["mask_predicted_data"] = torch.ones(target_train.shape) #same thing for the outputs
val_dict["tp_to_predict"] = observed_tp[0]
val_dict["observed_data"] = input_val
val_dict["observed_tp"] = observed_tp[0]
val_dict["data_to_predict"] = target_val
val_dict["mode"] = None 
val_dict['labels'] = None 
val_dict["observed_mask"] =  torch.ones(input_val.shape)
val_dict["mask_predicted_data"] = torch.ones(target_val.shape)

def objective(trial):
    #Hyperparameters
    n_ode_gru_dims = trial.suggest_int('n_ode_gru_dims', 5, 20)
    n_layers = trial.suggest_int('nombre de couches', 1, 3)
    lr = trial.suggest_float('learning rate', 1e-4, 1e-1, log=True)
    n_units = trial.suggest_int('n_gru_units', 10, 50)

    #training
    ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
        n_layers = n_layers, n_units = n_units, nonlinear = nn.Tanh)

    rec_ode_func = ODEFunc(
        input_dim = input_dim, 
        latent_dim = n_ode_gru_dims,
        ode_func_net = ode_func_net,
        device = device).to(device)

    #ODE solver
    z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", 10, 
        odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

    model = ODE_RNN(input_dim, n_ode_gru_dims, output_dim, device = device, 
        z0_diffeq_solver = z0_diffeq_solver, n_gru_units= n_units,
        concat_mask = True, obsrv_std = obsrv_std,
        use_binary_classif = False,
        classif_per_tp = False,
        n_labels = 1,
        train_classif_w_reconstr =  False).to(device)

    optimizer = optim.Adamax(model.parameters(), lr=lr)

    #training algorithm
    loss_val = [float('inf')]*100
    loss_train = []
    cur_loss_val = 0
    nb_epochs = 0
    criterion = nn.MSELoss()
    while abs(np.max(loss_val[-100:] - np.min(loss_val[-100:] + [cur_loss_val]))) >= eps and loss_val[-100] > cur_loss_val and nb_epochs < niters:
        optimizer.zero_grad()
        utils.update_learning_rate(optimizer, decay_rate = 0.999, lowest = lr / 10)
        train_res = model.compute_all_losses(train_dict, n_traj_samples = 100)
        train_res["loss"].backward()
        loss_train.append(train_res['mse'].item())
        optimizer.step()
        y_val = model.get_reconstruction(time_steps_to_predict= val_dict['tp_to_predict'],
                                        data= val_dict['observed_data'],
                                        truth_time_steps= val_dict['observed_tp'],
                                        mask =  val_dict['observed_mask'],
                                        n_traj_samples=100)
        a = torch.reshape(y_val[0], (100,26,1))
        cur_loss_val = criterion(a, target_val).item()
        loss_val.append(cur_loss_val)
        nb_epochs +=1
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
