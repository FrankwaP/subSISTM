import matplotlib
import matplotlib.pyplot
import matplotlib.pyplot as plt

import numpy as np
from numpy.random import randint
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

import utils
from utils import compute_loss_all_batches

from ode_rnn import *
from ode_func import ODEFunc
from diffeq_solver import DiffeqSolver
from joblib import Parallel, delayed
import os
#Seeding
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
#Settings
USE_NOISY_DATA = False
USE_MIXED_EFFECT = True
RANDOM_TIME = True
RE = 'Mixed' if USE_MIXED_EFFECT else 'Fixed'
Time = 'Random time' if RANDOM_TIME else 'Regular time'
#Device and pathing
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CSV_Dtest = "../../../data/synthetic_bph_1/Simulations random time/01_test.csv"
CSV_Dtest = os.path.join(os.path.dirname(__file__), CSV_Dtest)
dtest = pd.read_csv(CSV_Dtest, sep=";", decimal=",")



#Model parameters
obsrv_std = 0.01
n_ode_gru_dims = 10
input_dim = 8
output_dim = 1
lr = 1e-2
niters = 5000
eps = 0.0005
train_dict={}
val_dict={}
test_dict={}

#fuunction for looping over 
def ODE_GRU_process(n_set = 1):
    print("n_set:", n_set)
    CSV_FILE = "../../../data/synthetic_bph_1/Simulations random time/simulation" + str(n_set) + ".csv"
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

    dtest_norm = dtest.copy()
    dtest_norm = dtest_norm.dropna()

    #scaler_x_test = RobustScaler()
    dtest_norm.loc[:,x_labels[:-1]] = scaler_x.fit_transform(dtest_norm[x_labels[:-1]])

    #scaler_y_test = RobustScaler()
    dtest_norm.loc[:,y_labels] = scaler_y.fit_transform(dtest_norm[y_labels])

    groupby = dtest_norm.groupby('individus')[x_labels].apply(np.array)
    input_test = [torch.Tensor(x) for x in groupby]
    input_test = torch.stack(input_test)
    groupby = dtest_norm.groupby('individus')[y_labels].apply(np.array)
    target_test = [torch.Tensor(x) for x in groupby]
    target_test = torch.stack(target_test)

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
    test_dict["tp_to_predict"] = observed_tp[0]
    test_dict["observed_data"] = input_test
    test_dict["observed_tp"] = observed_tp[0]
    test_dict["data_to_predict"] = target_test
    test_dict["mode"] = None
    test_dict['labels'] = None
    test_dict["observed_mask"] =  torch.ones(input_test.shape)
    test_dict["mask_predicted_data"] = torch.ones(target_test.shape)

    #Model Definition
    #The net used to represent the flucuation of h_t bewteen observations
    ode_func_net = utils.create_net(n_ode_gru_dims, n_ode_gru_dims, 
        n_layers = 2, n_units = 25, nonlinear = nn.Tanh)

    rec_ode_func = ODEFunc(
        input_dim = input_dim, 
        latent_dim = n_ode_gru_dims,
        ode_func_net = ode_func_net,
        device = device).to(device)

    #ODE solver
    z0_diffeq_solver = DiffeqSolver(input_dim, rec_ode_func, "euler", 10, 
        odeint_rtol = 1e-3, odeint_atol = 1e-4, device = device)

    model = ODE_RNN(input_dim, n_ode_gru_dims, output_dim, device = device, 
        z0_diffeq_solver = z0_diffeq_solver, n_gru_units = 25,
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
    min_loss = float('inf')
    criterion = nn.MSELoss()
    while abs(np.max(loss_val[-100:] - np.min(loss_val[-100:] + [cur_loss_val]))) >= eps and nb_epochs < niters:
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
    #plot loss
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(loss_train, label ="loss")
    plt.plot(loss_val, label = "validation")
    plt.legend()

    pred_train = model.get_reconstruction(time_steps_to_predict= train_dict['tp_to_predict'],
                                     data= train_dict['observed_data'],
                                     truth_time_steps= train_dict['observed_tp'],
                                     mask =  train_dict['observed_mask'],
                                     n_traj_samples=400)
    y_pred = pred_train[0]

    #unscale data and aggregate training set
    multi_index = pd.MultiIndex.from_product([N_train, range(y_pred.shape[2])], names=['individus', 'temps'])
    df = pd.DataFrame(index = multi_index, data = y_pred.detach().numpy().flatten(), columns=['y_pred_ODE RNN_' + RE])
    df = df.reset_index()
    df = df.set_index(data_train.index)
    df.loc[:,'y_pred_ODE RNN_' + RE] = scaler_y.inverse_transform(df[['y_pred_ODE RNN_' + RE]])
    data_train.loc[:,y_labels] = scaler_y.inverse_transform(data_train[y_labels])

    multi_index = pd.MultiIndex.from_product([N_val, range(y_val[0].shape[2])], names=['individus', 'temps'])
    df_val = pd.DataFrame(index = multi_index, data = y_val[0].detach().numpy().flatten(), columns=['y_pred_ODE RNN_' + RE])
    df_val = df_val.reset_index()
    df_val = df_val.set_index(data_val.index)
    df_val.loc[:,'y_pred_ODE RNN_' + RE] = scaler_y.inverse_transform(df_val[['y_pred_ODE RNN_' + RE]])
    data_val.loc[:,y_labels] = scaler_y.inverse_transform(data_val[y_labels])

    df = pd.concat((df,df_val))
    data_train = pd.concat((data_train, data_val))
    #Results on training
    MAE_list_train = []
    MSE_list_train = []
    MAE_list_train_obs = []
    MSE_list_train_obs = []
    for k in range(1,499):
        pred_k = df[(df['temps']>5) & (df['individus'] == k)]['y_pred_' + 'ODE RNN' + '_' + RE]
        target_k = data_train[(data_train['temps']>5) & (data_train['individus'] == k)][y_labels[0][:-4]]
        target_k_obs = data_train[(data_train['temps']>5) & (data_train['individus'] == k)][y_labels]
        MAE_list_train.append(MAE(pred_k, target_k))
        MSE_list_train.append(MSE(pred_k, target_k))
        MAE_list_train_obs.append(MAE(pred_k, target_k_obs))
        MSE_list_train_obs.append(MSE(pred_k, target_k_obs))
    #resultats sur le test
    pred_test = model.get_reconstruction(time_steps_to_predict=test_dict['tp_to_predict'],
                                        data=test_dict['observed_data'],
                                        truth_time_steps=test_dict['observed_tp'],
                                        mask = test_dict['observed_mask'],
                                        n_traj_samples=500)
    y_test = pred_test[0]
    multi_index_test = pd.MultiIndex.from_product([range(y_test.shape[1]), range(y_test.shape[2])], names=['individus', 'temps'])
    df_test = pd.DataFrame(index = multi_index_test, data = y_test.detach().numpy().flatten(), columns=['y_test_' + 'ODE RNN' + '_' + RE])
    df_test = df_test.reset_index()
    df_test = df_test.set_index(dtest_norm.index)
    df_test['individus']+=1
    df_test.loc[:,'y_test_' + 'ODE RNN' + '_' + RE] = scaler_y.inverse_transform(df_test[['y_test_' + 'ODE RNN' + '_' + RE]])
    dtest_norm.loc[:,y_labels] = scaler_y.inverse_transform(dtest_norm[y_labels])
    #Scores on test data
    MAE_list_test = []
    MSE_list_test = []
    MAE_list_test_obs = []
    MSE_list_test_obs = []
    for k in range(1,501):
        pred_k = df_test[(df_test['temps']>5) & (df_test['individus'] == k)]['y_test_' +'ODE RNN' + '_' + RE]
        target_k = dtest_norm[(dtest_norm['temps']>5) & (dtest_norm['individus'] == k)][y_labels[0][:-4]]
        target_k_obs = dtest_norm[(dtest_norm['temps']>5) & (dtest_norm['individus'] == k)][y_labels]
        MAE_list_test.append(MAE(pred_k,target_k))
        MSE_list_test.append(MSE(pred_k,target_k))
        MAE_list_test_obs.append(MAE(pred_k,target_k_obs))
        MSE_list_test_obs.append(MSE(pred_k,target_k_obs))

    #Saving the model
    p = "../../../models/ODE-RNN/Résultats/Parameters/POIDS_ODE RNN_" + RE + Time + "_" + str(n_set)
    p = os.path.join(os.path.dirname(__file__), p)
    torch.save(model.state_dict(), p)


    #On met tous les résultats dans un dictionnaire
    res_loop = {}
    res_loop['df'] = df
    res_loop['df test'] = df_test
    res_loop['MAE train'] = np.mean(MAE_list_train)
    res_loop['MSE train'] = np.mean(MSE_list_train)
    res_loop['MAE train obs'] = np.mean(MAE_list_train_obs)
    res_loop['MSE train obs'] = np.mean(MSE_list_train_obs)
    res_loop['MAE test'] = np.mean(MAE_list_test)
    res_loop['MSE test'] = np.mean(MSE_list_test)
    res_loop['MAE test obs'] = np.mean(MAE_list_test_obs)
    res_loop['MSE test obs'] = np.mean(MSE_list_test_obs)
    if n_set == 1:
        p = "../../../models/ODE-RNN/Résultats/"
        p = os.path.join(os.path.dirname(__file__), p)
        df.to_csv(p + "Prédictions_entrainement_ODE RNN_" + RE + Time + ".csv", index=False)
        df_test.to_csv(p + "Prédictions_test_ODE RNN_" + RE + Time + ".csv", index=False)
    return (res_loop)

res = Parallel(n_jobs=4)(delayed(ODE_GRU_process)(n_set = i) for i in range(1,21))

metrics =  ['MAE train', 'MSE train', 'MAE train obs', 'MSE train obs', 'MAE test', 'MSE test', 'MAE test obs', 'MSE test obs']
scores = [{k:x[k] for k in metrics} for x in res]
scores = pd.DataFrame(scores)

#df = res[-1]['df']
#df_test = res[-1]['df test']

m_scores = scores.loc[:,metrics].mean()
m_scores = m_scores.to_list()
m_scores = [RE, "ODE RNN"] + m_scores

results = pd.DataFrame([m_scores], 
                       columns=['Random effect', 'Model', "MAE moyenne sur l'entrainement", "MSE moyenne sur l'entrainement", "MAE moyenne sur l'entrainement bruité", "MSE moyenne sur l'entrainement bruité", "MAE moyenne sur le test", "MSE moyenne sur le test", "MAE moyenne sur le test bruité", "MSE moyenne sur le test bruité"])
p = "../../../models/ODE-RNN/Résultats/ODE RNN_" + RE + Time + ".json"
p = os.path.join(os.path.dirname(__file__), p)
results = results.to_json(path_or_buf= p)

