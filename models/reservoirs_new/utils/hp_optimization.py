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

from numpy import mean, array
from numpy.typing import NDArray
from optuna import Trial, create_study, samplers
from optuna.logging import set_verbosity, DEBUG
import optunahub

from reservoirpy.nodes import ESN, Reservoir, Ridge  # type: ignore

from reservoirpy.observables import mse  # type: ignore


from .data import get_dataframe, prepare_data, remove_warmup_3D, FLOAT_DTYPE
from .config import (
    SERIES_COLUMN_NAME,
    TIMESTEPS_COLUMN_NAME,
    N_CPUS,
    N_SEEDS,
    N_WARMUPS,
    JOBLIB_BACKEND,
    SCALER,
)


# optuna
N_STARTUP_TRIALS = 50
N_TPE_TRIALS = 50
set_verbosity(DEBUG)

mod = optunahub.load_module(
    package="samplers/catcma",
)
CatCmaSampler = mod.CatCmaSampler


# %%
def _get_trial_model_list(trial: Trial) -> list[ESN]:
    model_list = []

    # if trial.number < N_STARTUP_TRIALS // 2:
    #     max_units = 500
    # else:
    max_units = 3000

    reservoir_dict = dict(
        units=trial.suggest_int("N", 10, max_units),
        sr=trial.suggest_float("sr", 1e-4, 1e1, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e0, log=True),
        input_scaling=trial.suggest_float(
            "input_scaling", 1e-1, 1e1, log=True
        ),
        dtype=FLOAT_DTYPE,
    )

    ridge_dict = dict(
        ridge=trial.suggest_float("ridge", 1e-8, 1e2, log=True),
    )

    esn_dict = dict(
        use_raw_inputs=False,
        # use_raw_inputs=trial.suggest_categorical(
        #     "use_raw_inputs", [True, False]
        # ),
        feedback=False,
        # feedback=trial.suggest_categorical("feedback", [True, False]),
    )

    for reservoir_seed in range(42, 42 + N_SEEDS):
        reservoir = Reservoir(**reservoir_dict, seed=reservoir_seed)
        readout = Ridge(**ridge_dict)

        model_list.append(
            ESN(
                reservoir=reservoir,
                readout=readout,
                workers=N_CPUS,
                backend=JOBLIB_BACKEND,
                **esn_dict,
            )
        )
    return model_list


def _optuna_objective(
    trial: Trial,
    x_train_3D_scaled: NDArray,
    y_train_3D_scaled: NDArray,
    x_test_3D_scaled: NDArray,
    y_test_3D_scaled: NDArray,
) -> (float, int):
    list_mse_scaled = []

    for model in _get_trial_model_list(trial):
        model.fit(x_train_3D_scaled, y_train_3D_scaled, warmup=N_WARMUPS)

        y_pred_3D_scaled = model.run(x_test_3D_scaled)
        if isinstance(y_pred_3D_scaled, list):
            y_pred_3D_scaled = array(y_pred_3D_scaled)

        list_mse_scaled.append(
            mse(
                remove_warmup_3D(y_pred_3D_scaled, N_WARMUPS),
                remove_warmup_3D(y_test_3D_scaled, N_WARMUPS),
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
        serie_column_name=SERIES_COLUMN_NAME,
        tstep_column_name=TIMESTEPS_COLUMN_NAME,
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
        serie_column_name=SERIES_COLUMN_NAME,
        tstep_column_name=TIMESTEPS_COLUMN_NAME,
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
