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
USE_MIXED_EFFECT = True
RANDOM_TIME = True
MODEL = RNN        
RE = 'Mixed' if USE_MIXED_EFFECT else 'Fixed'
Time = 'random time' if RANDOM_TIME else 'regular time'

MAE_train_sims = []
MSE_train_sims = []
MAE_test_sims = []
MSE_test_sims = []
MAE_train_sims_obs = []
MSE_train_sims_obs = []
MAE_test_sims_obs = []
MSE_test_sims_obs = []


def rnn_train_test(n_set):

    res_loop = {}

    CSV_FILE = "../../data/synthetic_bph_1/Simulations" + "/simulation" + str(n_set) + ".csv"
    CSV_Dtest = "../../data/synthetic_bph_1/Simulations" + "/01_test.csv"
    CSV_RES_R = "../../data/synthetic_bph_1/Résultats " + Time
    CSV_FILE = os.path.join(os.path.dirname(__file__), CSV_FILE)
    CSV_Dtest = os.path.join(os.path.dirname(__file__), CSV_Dtest)
    CSV_RES_R = os.path.join(os.path.dirname(__file__), CSV_RES_R)
    #data loading
    data = pd.read_csv(CSV_FILE, sep=";", decimal=",")
    dtest = pd.read_csv(CSV_Dtest, sep=";", decimal=",")
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

    #hyperparamètres
    h = 25
    lr = 0.001
    criterion = torch.nn.MSELoss()
    epoch = 20000
    eps = 0.0005

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
    if n_set == 1:

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(loss_train, label ="loss")
        plt.plot(loss_val, label = "validation")
        plt.legend()
        p = "../../models/ODE-RNN/Résultats/Graphs/Loss_RNN_" + RE +".png"
        p = os.path.join(os.path.dirname(__file__), p)
        plt.savefig(p)


    #unscale data and aggregate training set
    multi_index = pd.MultiIndex.from_product([N_train, range(y_pred.shape[1])], names=['individus', 'temps'])
    df = pd.DataFrame(index = multi_index, data = y_pred.detach().numpy().flatten(), columns=['y_pred_' + MODEL.__name__ + '_' + RE])
    df = df.reset_index()
    df = df.set_index(data_train.index)
    df.loc[:,'y_pred_' + MODEL.__name__ + '_' + RE] = scaler_y.inverse_transform(df[['y_pred_' + MODEL.__name__ + '_' + RE]])
    data_train.loc[:,y_labels] = scaler_y.inverse_transform(data_train[y_labels])

    multi_index = pd.MultiIndex.from_product([N_val, range(y_val.shape[1])], names=['individus', 'temps'])
    df_val = pd.DataFrame(index = multi_index, data = y_val.detach().numpy().flatten(), columns=['y_pred_' + MODEL.__name__ + '_' + RE])
    df_val = df_val.reset_index()
    df_val = df_val.set_index(data_val.index)
    df_val.loc[:,'y_pred_' + MODEL.__name__ + '_' + RE] = scaler_y.inverse_transform(df_val[['y_pred_' + MODEL.__name__ + '_' + RE]])
    data_val.loc[:,y_labels] = scaler_y.inverse_transform(data_val[y_labels])

    df = pd.concat((df,df_val))
    data_train = pd.concat((data_train, data_val))

    #Results on training
    MAE_list_train = []
    MSE_list_train = []
    MAE_list_train_obs = []
    MSE_list_train_obs = []
    for k in range(1,501):
        pred_k = df[(df['temps']>5) & (df['individus'] == k)]['y_pred_' + MODEL.__name__ + '_' + RE]
        target_k = data_train[(data_train['temps']>5) & (data_train['individus'] == k)][y_labels[0][:-4]]
        target_k_obs = data_train[(data_train['temps']>5) & (data_train['individus'] == k)][y_labels]
        MAE_list_train.append(MAE(pred_k, target_k))
        MSE_list_train.append(MSE(pred_k, target_k))
        MAE_list_train_obs.append(MAE(pred_k, target_k_obs))
        MSE_list_train_obs.append(MSE(pred_k, target_k_obs))

    #TEST
    y_test = model(input_test)
    multi_index_test = pd.MultiIndex.from_product([range(y_test.shape[0]), range(y_test.shape[1])], names=['individus', 'temps'])
    df_test = pd.DataFrame(index = multi_index_test, data = y_test.detach().numpy().flatten(), columns=['y_test_' + MODEL.__name__ + '_' + RE])
    df_test = df_test.reset_index()
    df_test = df_test.set_index(dtest_norm.index)
    df_test['individus']+=1
    df_test.loc[:,'y_test_' + MODEL.__name__ + '_' + RE] = scaler_y.inverse_transform(df_test[['y_test_' + MODEL.__name__ + '_' + RE]])
    dtest_norm.loc[:,y_labels] = scaler_y.inverse_transform(dtest_norm[y_labels])

    #Scores on test data
    MAE_list_test = []
    MSE_list_test = []
    MAE_list_test_obs = []
    MSE_list_test_obs = []
    for k in range(1,501):
        pred_k = df_test[(df_test['temps']>5) & (df_test['individus'] == k)]['y_test_' + MODEL.__name__ + '_' + RE]
        target_k = dtest_norm[(dtest_norm['temps']>5) & (dtest_norm['individus'] == k)][y_labels[0][:-4]]
        target_k_obs = dtest_norm[(dtest_norm['temps']>5) & (dtest_norm['individus'] == k)][y_labels]
        MAE_list_test.append(MAE(pred_k,target_k))
        MSE_list_test.append(MSE(pred_k,target_k))
        MAE_list_test_obs.append(MAE(pred_k,target_k_obs))
        MSE_list_test_obs.append(MSE(pred_k,target_k_obs))

    p = "../../models/ODE-RNN/Résultats/Parameters/POIDS_"+ MODEL.__name__ + "_" + RE + Time + "_" + str(n_set)
    p = os.path.join(os.path.dirname(__file__), p)
    torch.save(model.state_dict(), p)

    #On met tous les résultats dans un dictionnaire
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
    print(n_set)
    if n_set == 1:
        p = "../../models/ODE-RNN/Résultats/"
        p = os.path.join(os.path.dirname(__file__), p)
        df.to_csv(p + "Prédictions_entrainement" + MODEL.__name__ + "_" + RE + Time + ".csv", index=False)
        df_test.to_csv(p + "Prédictions_test" + MODEL.__name__ + "_" + RE + Time + ".csv", index=False)
    return (res_loop)

res = Parallel(n_jobs=6)(delayed(rnn_train_test)(n_set = i) for i in range(1,21))

metrics =  ['MAE train', 'MSE train', 'MAE train obs', 'MSE train obs', 'MAE test', 'MSE test', 'MAE test obs', 'MSE test obs']
scores = [{k:x[k] for k in metrics} for x in res]
scores = pd.DataFrame(scores)

df = res[0]['df']
df_test = res[0]['df test']

m_scores = scores.loc[:,metrics].mean()
m_scores = m_scores.to_list()
m_scores = [RE, MODEL.__name__] + m_scores

results = pd.DataFrame([m_scores], 
                       columns=['Random effect', 'Model', "MAE moyenne sur l'entrainement", "MSE moyenne sur l'entrainement", "MAE moyenne sur l'entrainement bruité", "MSE moyenne sur l'entrainement bruité", "MAE moyenne sur le test", "MSE moyenne sur le test", "MAE moyenne sur le test bruité", "MSE moyenne sur le test bruité"])
p = "../../models/ODE-RNN/Résultats/"+ MODEL.__name__ + "_" + RE + Time + ".json"
p = os.path.join(os.path.dirname(__file__), p)
results = results.to_json(path_or_buf= p)
