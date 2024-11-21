# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable, ModuleType

from numpy.typing import NDArray
from numpy import array, mean

from optuna import load_study
from sklearn.metrics import mean_squared_error as mse

from .data import remove_warmup_df, get_dataframe, prepare_data
from .hp_optimization import _get_trial_model_list, ESN, N_WARMUPS
from .config import (
    SERIES_COLUMN_NAME,
    TIMESTEPS_COLUMN_NAME,
    N_CPUS,
    N_SEEDS,
    N_WARMUPS,
    JOBLIB_BACKEND,
    SCALER,
)


TEST_FILE = "../../../data/synthetic_bph_1/01_test.csv"
GLOB_SIMU = "../../../data/synthetic_bph_1/simulation1.csv"


def _train_pred_simu(
    model_list: list[ESN],
    x_test_3D_scaled: NDArray,
    y_test_2D: NDArray,
    inverse_transform: Callable,
):
    list_mse_scaled = []

    for model in model_list:
        y_pred_3D_scaled = model.run(x_test_3D_scaled)
        if isinstance(y_pred_3D_scaled, list):
            y_pred_3D_scaled = array(y_pred_3D_scaled)
        y_pred_2D = inverse_transform(y_pred_3D_scaled)

        list_mse_scaled.append(
            mse(
                remove_warmup_df(y_pred_2D, N_WARMUPS),
                remove_warmup_df(y_test_2D, N_WARMUPS),
            )
        )
    return mean(list_mse_scaled)


def train_pred_loop(study_config: ModuleType):

    model_list = _get_trial_model_list(load_study.best_trial)

    df_test = get_dataframe(TEST_FILE)

    for file_simu in Path.rglob(GLOB_SIMU):
        df_simu = get_dataframe(file_simu)

        (
            _,
            _,
            x_test_3D_scaled,
            _,
            inverse_transform,
        ) = prepare_data(
            df_train=df_simu,  # !!!
            df_test=df_test,  # !!!
            serie_column_name=SERIES_COLUMN_NAME,
            tstep_column_name=TIMESTEPS_COLUMN_NAME,
            x_labels=study_config.X_LABELS,
            y_labels=study_config.Y_LABELS,
            x_scaler=SCALER(),
            y_scaler=SCALER(),
        )

        _train_pred_simu(
            model_list,
        )
