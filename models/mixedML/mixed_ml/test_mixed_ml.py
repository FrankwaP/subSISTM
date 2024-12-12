#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import path

path.append("../reservoirs-synthetic_bph/")

from numpy.random import seed

from sklearn.neural_network import MLPRegressor
from reservoirpy.nodes import Reservoir, Ridge


from mixed_ml import MixedMLEstimator
from utils.data import get_dataframe


seed(42)


data = get_dataframe("../../data/synthetic_bph_1/01_test.csv")
data_train = data[(10 < data["individus"]) & (data["individus"] <= 30)]
data_test = data[data["individus"] <= 10]

N_series = len(data["individus"].unique())
N_tsteps = len(data["temps"].unique())

###################

MLP = MLPRegressor(hidden_layer_sizes=(20, 20), max_iter=2000)
mixed_ml_estimator = MixedMLEstimator(MLP, recurrent_model=False)
mixed_ml_estimator.fit(data_train, n_iter_improve=2, max_iter=30)
pred_ss = mixed_ml_estimator.predict(data_test, use_subject_specific=True)
pred_marg = mixed_ml_estimator.predict(data_test, use_subject_specific=False)

###################

res, rid = Reservoir(100), Ridge(ridge=1.5)

ESN = res >> rid
mixed_ml_estimator = MixedMLEstimator(ESN, recurrent_model=True)
mixed_ml_estimator.fit(data_train, n_iter_improve=2, max_iter=30)
pred_ss = mixed_ml_estimator.predict(data_test, use_subject_specific=True)
pred_marg = mixed_ml_estimator.predict(data_test, use_subject_specific=False)
