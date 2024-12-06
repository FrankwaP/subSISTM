# -*- coding: utf-8 -*-

from sys import path
from subprocess import call

from numpy import mean
from numpy.typing import NDArray
from pandas import read_csv, DataFrame
from sklearn.neural_network import MLPRegressor

path.append("../reservoirs-synthetic_bph/")

from utils.data import get_dataframe


# should be the same as in Rscript
X_LABELS = ["x2_x5", "x4_x7", "x6_x8"]

# the paper says "y_fixed" for "y fixed effects"
# but it can be understood as "y not changing"
# so I call it "y_fe"
Y_LABEL = "y_mixed_obs"  # the values are fixed
Y_LABEL_FE = Y_LABEL + "_fixed"  # the value are not fixed


def iterate_mixedml(
    ml_fixed: MLPRegressor,
    df_data: DataFrame,
    lr: float,
) -> tuple[MLPRegressor, DataFrame, bool]:

    ####
    X = df_data[X_LABELS]
    y = df_data[Y_LABEL_FE]
    # we train ml_fixed by ignoring cluster effects (with the target y)
    ml_fixed.fit(X, y)
    # to get an estimate of y_fixed
    try:
        y_fixed = ml_fixed.predict(X)
    except AttributeError:
        y_fixed = ml_fixed.run(X)

    ####
    # based on e_fixed
    df_data["e_fixed"] = y - y_fixed
    df_data.to_csv("ml_pred.csv", index=False)
    # we estimate u
    call(["Rscript", "random_effects_fitting.R"])
    random_preds = read_csv("random_preds.csv")
    random_pred_ss = random_preds["pred_ss"].to_numpy()

    # then we upgrade y_fixed = y-Zu
    # … and re-train ml_fixed with the updated target variable y_fixed
    df_data[Y_LABEL_FE] = df_data[Y_LABEL] - lr * random_pred_ss

    return ml_fixed, df_data, msr(random_preds["resid_ss"])


def mar(resid: NDArray) -> float:
    return mean(abs(resid))


def msr(resid: NDArray) -> float:
    return mean(resid**2)


def loop_mixedml(
    ml_fixed: MLPRegressor,
    df_data: DataFrame,
    eps: float,
    lr: float,
    max_iter: int,
):
    # initialization
    istep = 0
    converged = False
    df_data[Y_LABEL_FE] = df_data[Y_LABEL]
    # iteration
    resid_list = []
    while not converged and istep < max_iter:
        ml_fixed, df_data, resid = iterate_mixedml(ml_fixed, df_data, lr)
        # print(resid)
        converged = resid < eps
        istep += 1
        resid_list.append(resid)
        print(f"mixedML step #{istep:02d}: {resid:8e}")
    return resid_list


# if __name__ == "__main__":

#     data = get_dataframe("../../data/synthetic_bph_1/01_test.csv")
#     data = data[data["individus"] <= 10]

#     model = MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=2000)
#     loop_mixedml(model, data, 1e-1, lr=1e-2, max_iter=1000)
