#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

values for training:
    xi without noise: "x1", …
    y with noise: "y_fixed_obs" or "y_mixed_obs"
values for metrics:
    y without noise: "y_fixed" or "y_mixed"



HP optimisation step #1:
    train = simulation1
    val = simulation2
    
HP optimisation step #2:
    train = simulation2
    val = simulation1
    
Compare the HPs from both optimisation steps.


Training/prediction steps:
    test = 01_test.csv
    
    for train in simulation*.csv:
        fit on train
        
        prediction on train
        MAE/MSE on train
        
        prediction on test
        MAE/MSE on test

"""
from types import ModuleType
from typing import Union

from numpy import mean, array, isnan
from numpy.typing import NDArray
from optuna import Trial, create_study, samplers
from optuna.logging import set_verbosity, DEBUG
from reservoirpy.observables import mse  # type: ignore


from .reservoirs import get_esn_model_list, ESN
from .data import get_dataframe, prepare_data, remove_warmup_3D
from .global_config import (
    N_WARMUPS,
    SCALER,
)


# optuna
N_STARTUP_TRIALS = 100
N_TPE_TRIALS = 50
set_verbosity(DEBUG)


# %%


def _get_trial_model_list(trial: Trial) -> list[ESN]:

    reservoir_kwargs = dict(
        units=trial.suggest_int("N", 50, 500),
        sr=trial.suggest_float("sr", 1e-4, 1e1, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e0, log=True),
        input_scaling=trial.suggest_float(
            "input_scaling", 1e-1, 1e1, log=True
        ),
    )

    ridge_kwargs = dict(
        ridge=trial.suggest_float("ridge", 1e-8, 1e2, log=True),
    )

    ens_kwargs = dict(
        use_raw_inputs=False,
        # use_raw_inputs=trial.suggest_categorical(
        #     "use_raw_inputs", [True, False]
        # ),
        feedback=False,
        # feedback=trial.suggest_categorical("feedback", [True, False]),
    )

    return get_esn_model_list(reservoir_kwargs, ridge_kwargs, ens_kwargs)


def check_if_use_yprey(
    x_train_3D_scaled: NDArray,
    x_test_3D_scaled: NDArray,
) -> bool:
    # we look for Nan in the first tstep (y @ t-1 is Nan)
    train_tmp = x_train_3D_scaled[:, 0, :]
    test_tmp = x_test_3D_scaled[:, 0, :]

    shp = train_tmp.shape[1]

    for i in range(shp):
        if isnan(train_tmp[0, i]):
            assert all(isnan(train_tmp[:, i]))
            assert all(isnan(test_tmp[:, i]))
            return True


def adpapt_tensors_if_ypred(
    x_train_3D_scaled: NDArray,
    y_train_3D_scaled: NDArray,
    x_test_3D_scaled: NDArray,
    y_test_3D_scaled: Union[NDArray, None],
    n_warmups: int,
) -> tuple[NDArray, NDArray, NDArray, NDArray, int]:
    # !!! could be in data but must find how to handle N_WARMUPS properly
    if check_if_use_yprey(x_train_3D_scaled, x_test_3D_scaled):
        n_warmups -= 1
        x_train_3D_scaled = x_train_3D_scaled[:, 1:, :]
        y_train_3D_scaled = y_train_3D_scaled[:, 1:, :]
        x_test_3D_scaled = x_test_3D_scaled[:, 1:, :]
        if y_test_3D_scaled:
            y_test_3D_scaled = y_test_3D_scaled[:, 1:, :]
    return (
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
        n_warmups,
    )


def _optuna_objective(
    trial: Trial,
    x_train_3D_scaled: NDArray,
    y_train_3D_scaled: NDArray,
    x_test_3D_scaled: NDArray,
    y_test_3D_scaled: NDArray,
) -> tuple[float, int]:
    list_mse_scaled = []
    n_warmups = N_WARMUPS

    (
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
        n_warmups,
    ) = adpapt_tensors_if_ypred(
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
        n_warmups,
    )

    for model in _get_trial_model_list(trial):
        model.fit(x_train_3D_scaled, y_train_3D_scaled, warmup=n_warmups)

        y_pred_3D_scaled = model.run(x_test_3D_scaled)
        if isinstance(y_pred_3D_scaled, list):
            y_pred_3D_scaled = array(y_pred_3D_scaled)

        list_mse_scaled.append(
            mse(
                remove_warmup_3D(y_pred_3D_scaled, n_warmups),
                remove_warmup_3D(y_test_3D_scaled, n_warmups),
            )
        )
    return mean(list_mse_scaled), trial.params["N"]


def _run_study(
    study_name: str,
    x_train_3D_scaled: NDArray,
    y_train_3D_scaled: NDArray,
    x_test_3D_scaled: NDArray,
    y_test_3D_scaled: NDArray,
) -> None:
    db_name = f"{study_name}.db"
    # %rm "$db_name"
    storage_name = f"sqlite:///{db_name}"

    study = create_study(
        study_name=study_name,
        storage=storage_name,
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )

    # study.sampler = CatCmaSampler()
    study.sampler = samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS, seed=0
    )
    study.optimize(
        lambda x: _optuna_objective(
            x,
            x_train_3D_scaled,
            y_train_3D_scaled,
            x_test_3D_scaled,
            y_test_3D_scaled,
        ),
        n_trials=N_STARTUP_TRIALS + N_TPE_TRIALS,
    )


def _run_test_study(
    x_train_3D_scaled: NDArray,
    y_train_3D_scaled: NDArray,
    x_test_3D_scaled: NDArray,
    y_test_3D_scaled: NDArray,
) -> None:
    study = create_study(
        study_name="OSEF",
        directions=["minimize"],
        load_if_exists=True,
    )

    study.sampler = samplers.RandomSampler(seed=0)
    study.optimize(
        lambda x: _optuna_objective(
            x,
            x_train_3D_scaled,
            y_train_3D_scaled,
            x_test_3D_scaled,
            y_test_3D_scaled,
        ),
        n_trials=2,
    )


def run_optimization(study_config: ModuleType) -> None:

    df_1 = get_dataframe("../../../data/synthetic_bph_1/simulation1.csv")
    df_2 = get_dataframe("../../../data/synthetic_bph_1/simulation2.csv")

    #
    study_name = "HP-optimization-01"  # !!!

    (
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
        _,
    ) = prepare_data(
        df_train=df_1,  # !!!
        df_test=df_2,  # !!!
        x_labels=study_config.X_LABELS,
        y_labels=study_config.Y_LABELS,
        x_scaler=SCALER(),
        y_scaler=SCALER(),
    )

    # _run_test_study(
    #     x_train_3D_scaled,
    #     y_train_3D_scaled,
    #     x_test_3D_scaled,
    #     y_test_3D_scaled,
    # )
    _run_study(
        study_name,
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
    )

    #
    study_name = "HP-optimization-02"  # !!!

    (
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
        _,
    ) = prepare_data(
        df_train=df_2,  # !!!
        df_test=df_1,  # !!!
        x_labels=study_config.X_LABELS,
        y_labels=study_config.Y_LABELS,
        x_scaler=SCALER(),
        y_scaler=SCALER(),
    )

    _run_study(
        study_name,
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
    )
