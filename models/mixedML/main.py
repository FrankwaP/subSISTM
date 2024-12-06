# -*- coding: utf-8 -*-

from sys import path
from pathlib import Path
from subprocess import call
from joblib import dump

from numpy import mean
from numpy.random import seed
from numpy.typing import NDArray
from pandas import read_csv, DataFrame
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse,
)


seed(42)
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
    ml_fixed: MLPRegressor, df_data: DataFrame
) -> tuple[MLPRegressor, DataFrame, bool]:

    #### fitting the Machine Learning model
    X = df_data[X_LABELS]
    y = df_data[Y_LABEL_FE]
    # we train ml_fixed by ignoring cluster effects (with the target y)
    ml_fixed.fit(X, y)
    # to get an estimate of y_fixed
    try:
        y_fixed = ml_fixed.predict(X)
    except AttributeError:
        y_fixed = ml_fixed.run(X)

    #### fitting the Mixed Effect model
    # based on e_fixed
    # !!! l'erreur était là (j'utilisais "y- y_fixed")
    df_data["e_fixed"] = df_data[Y_LABEL] - y_fixed
    df_data.to_csv("ml_pred.csv", index=False)
    # we estimate u
    call(["Rscript", "random_effects_fitting.R"])
    random_preds = read_csv("random_preds.csv")
    y_random = random_preds["pred_ss"].to_numpy()

    # then we upgrade y_fixed = y-Zu
    # … and re-train ml_fixed with the updated target variable y_fixed
    df_data[Y_LABEL_FE] = df_data[Y_LABEL] - y_random

    ####
    # final prediction to monitor convergence
    y_pred = y_fixed + y_random

    # check that it's equivalent
    # assert (
    #     abs(
    #         mse(df_data[Y_LABEL], y_pred) - mean(random_preds["resid_ss"] ** 2)
    #     )
    #     < 1e-6
    # )

    return ml_fixed, df_data, mse(df_data[Y_LABEL], y_pred)


def loop_mixedml(
    ml_fixed: MLPRegressor,
    df_data: DataFrame,
    *,
    n_iter_improve: int,
    max_iter: int,
):
    # initialization
    istep = 0
    df_data[Y_LABEL_FE] = df_data[Y_LABEL]
    # iteration
    metric_list = []
    best_metric = None
    for istep in range(max_iter):
        ml_fixed, df_data, metric = iterate_mixedml(ml_fixed, df_data)
        print(f"mixedML step #{istep:02d}: {metric:8e}", end="")
        #
        if best_metric is None or metric < best_metric:
            print(" (best)")
            best_metric = metric
            n_it_no_improve = 0
            ml_fixed_bak = ml_fixed
            Path("random_hlme.Rds").rename("best_random_hlme.Rds")
        else:
            print("")
            n_it_no_improve += 1
            if n_it_no_improve > n_iter_improve:
                break

        metric_list.append(metric)
        istep += 1

    dump(ml_fixed_bak, "best_fixed_ml.joblib")
    Path("random_hlme.Rds").unlink()

    return metric_list


if __name__ == "__main__":

    data = get_dataframe("../../data/synthetic_bph_1/01_test.csv")
    data = data[data["individus"] <= 10]

    model = MLPRegressor(hidden_layer_sizes=(20, 20, 20), max_iter=2000)
    loop_mixedml(model, data, n_iter_improve=10, max_iter=1000)
