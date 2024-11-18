#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path

import pandas as pd
import numpy as np

import optuna
from reservoirpy.observables import mse
from sklearn.preprocessing import RobustScaler
from optuna.visualization.matplotlib import (
    plot_slice,
    plot_contour,
    plot_param_importances,
)

from optuna_training import (
    ProcessedData,
    ModelConfiguration,
    get_3D_prediction_list,
    remove_warmup,
)
from optuna_post_processing import (
    get_predictions_dataframe,
    get_worst_individuals,
    plot_individual_results,
    get_statistics,
)

pd.options.display.max_rows = 20
pd.options.display.max_columns = 300


# # Data Loading

# In[2]:


CSV_FILE = "../../data/synthetic_bph_1/Simulations/01_test.csv"


# In[3]:


data = pd.read_csv(CSV_FILE, sep=";", decimal=",")
data = data.sort_values(by=["individus", "temps"])
data


# In[4]:


SERIES_COLUMN_NAME = "individus"
TIMESTEPS_COLUMN_NAME = "temps"


# Formula for `y_fixed` and `y_mixed` (data/synthetic_bph_1/model_oracle_regular_time.R):
#
# ```R
# g_fixed <- mvrnorm(length(ind), µg, diag(numeric(3)))
# df$y_fixed <- g_fixed[df$individus, 1] +
#               g_fixed[df$individus, 2] * df$x2 * df$x5 +
#               g_fixed[df$individus, 3] * df$x4 * df$x7
#
# g_mixed <- mvrnorm(length(ind), µg, diag(c(0.5, 0.5, 0.05)))
# df$y_mixed <- g_mixed[df$individus, 1] +
#               g_mixed[df$individus, 2] * df$x2 * df$x5 +
#               g_mixed[df$individus, 3] * df$x4 * df$x7
# ```
#
#
# So:
# $y_{fixed}(t) = f(x_2(t), x_5(t), x_4(t), x_7(t))$
# $y_{mixed}(t, i) = f(x_2(t), x_5(t), x_4(t), x_7(t), \gamma_{i,25}, )$

# # Summary
#
# ## Objective
#
# Study the capacity of Reservoir Computing (RC) to model longitudinal health data with mixed effects.
#
# In other word: do the modelisation of complex time dependecies can compensate the lack of direct modelisation of the random effect?
#
#
# ## Method
#
# ### Sub-studies
#
# The study is done in incremental steps, starting from this baseline:
# - fixed effects target, without noise
# - using the "oracle" features: $x_2 * x_5$, $x_4 * x_7$
#
# Then we'll see the effect of:
# - addding noise on the target
# - using the mixed effect target
# - using the real features ($x_1, …, x_7$)
#
# For each sub-study we'll do a **capacity** test, training *and* predicting on the same data, to see if the model can learn the data.
# If successful, we'll do a **generalization** test, with a train/test split.
#
# ### Hyper-parameters optimization
#
# From *Unsupervised Reservoir Computing for Solving Ordinary Differential Equations*:
# > Training an RC is efficient, however, finding the optimal hyper-parameters that determine the weight distributions and the network architecture is challenging. RC performance strongly depends on these hyper-parameters and, therefore, finding good hyper-parameters is crucial. The RC has many hyper-parameters making this a computationally expensive process.
#
#
# We will use Optuna for the hyper-parameters optimization.
#
# **trial**: set of hyperparameters suggested by Optuna
# **warmup**: first timesteps that are ignore in the computation of the loss and error
#
#
# **The method**:
# ```
# for each trial:
#     for 5 random seeds:
#         generate the reservoir
#         train the reservoir
#         predict using the reservoir
#         compute the MSE: MSE_seed
#     compute MSE_trial = mean(MSE_seed)
# return the best trial corresponding to min(MSE_trial)
# ```
#
#
# **Note on Optuna's TPESampler**:
#
# https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
# > Sampler using TPE (Tree-structured Parzen Estimator) algorithm.
# On each trial, for each parameter, TPE fits one Gaussian Mixture Model (GMM) $l(x)$ to the set of parameter values associated with the best objective values, and another GMM $g(x$) to the remaining parameter values. It chooses the parameter value x that maximizes the ratio $l(x)/g(x)$.
#
#
# https://anna-ssi.github.io/blog/tpe/
# > Tree-structured Parzen Estimators (TPE) derive their name from the **combination of Parzen estimators to model the probability distributions of hyperparameters and a structured, graph-like approach to represent hyperparameter configurations**. In this tree-like representation, each hyperparameter is a node, and edges denote the dependencies between them.
# For example, the choice of the optimizer (e.g., Adam) and the learning rate can be seen as interconnected nodes.
# This structured representation allows TPE to focus on updating only the relevant parts of the model when new observations are made.
# It also facilitates establishing **dependencies among random variables**, making conditional sampling more efficient and enabling the algorithm to optimize the search space faster.
#

# # Objective function for Optuna
#
#
#
#

# In[5]:


N_WARMUPS = 3
N_SEEDS = 5
N_CPUS = 2


# In[6]:


# https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html#Optimize-hyperparameters


def get_model_list(trial, x_labels):
    reservoir_kwargs = {
        "units": trial.suggest_int("N", 50, 5000),
        # "sr": trial.suggest_float("sr", 1e-2, 1e1, log=True),
        "sr": trial.suggest_float("sr", 1e-3, 1e1, log=True),
        # "lr": trial.suggest_float("lr", 1e-3, 1e0, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 1e0, log=True),
        # SUPER! tu as décalé l'indice de l'input_scaling avec celui du x correspondant <3
        "input_scaling": [
            trial.suggest_float(f"input_scaling_{i}", 1e-5, 1e2, log=True)
            for i, _ in enumerate(x_labels)
        ],
    }

    ridge_kwargs = {
        #  "ridge": trial.suggest_float("ridge", 1e-8, 1e1, log=True),
        "ridge": trial.suggest_float("ridge", 1e-10, 1e2, log=True),
    }

    fit_kwargs = {
        "warmup": N_WARMUPS,
    }

    return [
        ModelConfiguration(
            input_kwargs={},
            reservoir_kwargs=reservoir_kwargs,
            ridge_kwargs=ridge_kwargs,
            fit_kwargs=fit_kwargs,
            input_to_readout=trial.suggest_categorical("input_to_readout", [True, False]),
            readout_feedback_to_reservoir=trial.suggest_categorical(
                "readout_feedback_to_reservoir", [True, False]
            ),
        )
        for reservoir_kwargs["seed"] in range(42, 42 + N_SEEDS)
    ]


# %% Objective function


def optuna_objective(trial, processed_data):

    model_list = get_model_list(trial, processed_data.x_labels)
    list_mse = [
        mse(
            remove_warmup(processed_data.y_test_3D, N_WARMUPS),
            remove_warmup(y_hat_3D, N_WARMUPS),
        )
        for y_hat_3D in get_3D_prediction_list(model_list, processed_data, n_cpus=N_CPUS)
    ]
    return np.mean(list_mse)


# # Baseline => Capacity
#
# We use:
# - features:
#   - "x2_x5" and "x4_x7"
#   - without noise
# - target without random effects

# %% Entraînement


STUDY_NAME = "RELOU"


x_labels = ["x2_x5", "x4_x7"]
data_train = data_test = data
y_labels_train = y_labels_test = ["y_fixed"]

processed_data = ProcessedData(
    data_train,
    data_test,
    SERIES_COLUMN_NAME,
    TIMESTEPS_COLUMN_NAME,
    x_labels,
    y_labels_train,
    y_labels_test,
    RobustScaler(),
)

study = optuna.create_study(study_name=STUDY_NAME, directions=["minimize"])
study.sampler = optuna.samplers.RandomSampler()
study.optimize(lambda x: optuna_objective(x, processed_data), n_trials=1)


# %% Prediction


best_model_list = get_model_list(study.best_trial, x_labels)
df_pred = get_predictions_dataframe(best_model_list, processed_data, n_cpus=N_CPUS)
