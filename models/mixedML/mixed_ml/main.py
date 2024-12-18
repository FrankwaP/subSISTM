#!/usr/bin/env python
# coding: utf-8

# # MixedML on Python

# Linear Mixed Effect: $$Y_{ij} = \beta_0 + \beta_i X_{ij} + b_{0j} + b_{ij} X_{ij} + \epsilon_{i}$$
# => MixedML: $$Y_{ij} = ML(X_{ij}) + b_{0j} + b_{ij} X_{ij} + \epsilon_{i}$$
#
#
# with ML: any type of Machine Learning model
#

# LCMM is used for the (pure) random effect model:
#
#
# ```R
# random_hlme <- hlme(
#   e_fixed ~ 1,
#   random = ~  1 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8,
#   idiag = TRUE,
#   data = data,
#   subject = 'individus',
#   var.time = 'temps'
# )
# ```

# Algorithm initialization:  $Y_{fixed} = Y$
#
# Until convergence of $squared\_residuals$:
# |
# |$~~~~~ML.fit(X, Y_{fixed})$
# |$~~~~~Y_{rand} = Y - ML(X)$
# |
# |$~~~~~HLME.fit(X, Y_{rand})$
# |$~~~~~Y_{fixed} = Y - HLME(X)$
# |
# |$~~~~~Y_{pred} = ML(X) + HLME(X)$
# |$~~~~~squared\_residuals = (Y - Y_{pred})^2$

# In[1]:


from sys import path
from pathlib import Path


def add_path(p: str):
    pth = Path(p).resolve().as_posix()
    print(pth)
    path.append(pth)


add_path("../")
add_path("../../")


# In[2]:


import seaborn as sns
from matplotlib.pyplot import subplots
from pandas import DataFrame
import pandas as pd
from reservoirpy import verbosity
from reservoirpy.nodes import Reservoir, Ridge
from sklearn.metrics import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler


from reservoirs_synthetic_bph.utils.reservoirs import ReservoirEnsemble
from reservoirs_synthetic_bph.utils.data import get_dataframe
from reservoirs_synthetic_bph.utils.global_config import N_WARMUPS
from mixed_ml import MixedMLEstimator

pd.options.display.float_format = "{:.3g}".format
verbosity(0)


# In[3]:


data = get_dataframe("../../../data/synthetic_bph_1/01_test.csv")
SERIES = "individus"
TSTEPS = "temps"
#
data_train = data[data[SERIES] > 10]
data_test = data[data[SERIES] <= 10]


# ## Building the model
#
# Standard models can be used:

# In[4]:


model_mlp = MLPRegressor((20, 10, 5), learning_rate="adaptive", max_iter=1000)
mixed_ml_mlp = MixedMLEstimator(model_mlp, recurrent_model=False)


# Also recurrent ones:

# In[5]:


model_rpy = ReservoirEnsemble(
    reservoir_kwargs={"units": 50}, ridge_kwargs={"ridge": 1e-1}
)
mixed_ml_rpy = MixedMLEstimator(model_rpy, recurrent_model=True)


# In[6]:


# ## Training
# with all the features: $x_1,…, x_8$

# In[7]:


X_LABELS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
Y_LABEL = "y_mixed_obs"
Y_LABELS_PRED = ["y_mixed", "y_mixed_obs"]

scaler_x = RobustScaler()
scaler_y = RobustScaler()

data_train_scaled = data_train[[SERIES, TSTEPS]].copy()
data_train_scaled[X_LABELS] = scaler_x.fit_transform(data_train[X_LABELS])
data_train_scaled[[Y_LABEL]] = scaler_y.fit_transform(data_train[[Y_LABEL]])

data_test_scaled = data_test[[SERIES, TSTEPS]].copy()
data_test_scaled[X_LABELS] = scaler_x.transform(data_test[X_LABELS])
data_test_scaled[[Y_LABEL]] = scaler_y.transform(data_test[[Y_LABEL]])


# In[8]:


# In[9]:


results_rpy = mixed_ml_rpy.fit(
    data_train_scaled,
    n_iter_improve=2,
    min_rltv_imprv=0.01,
    fixed_model_options={"warmup": N_WARMUPS},
)
