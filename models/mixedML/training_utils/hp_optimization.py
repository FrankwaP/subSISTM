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
from sys import path

path.append("..")


from numpy import mean
from pandas import DataFrame
from optuna import Trial, create_study, samplers
from optuna.logging import set_verbosity, DEBUG

from reservoirpy.observables import mse  # type: ignore


from .trial_model_multiprocessing import multi_proc_fit_and_predict_trial
from .data import get_dataframe, remove_warmup_1D
from .global_config import N_WARMUPS, SCALER
from mixed_ml.mixed_ml import (
    SERIES,
    TSTEPS,
    X_LABELS,
    Y_LABEL,
)


# optuna
N_STARTUP_TRIALS = 20
N_TPE_TRIALS = 10
set_verbosity(DEBUG)


# %%


def _optuna_objective(
    trial: Trial,
    df_train_scaled: DataFrame,
    df_val_scaled: DataFrame,
) -> tuple[float, int]:

    options_fit = dict(
        df_data=df_train_scaled,
        n_iter_improve=1,
        min_ratio_improve=0.01,
        fixed_model_options={"warmup": N_WARMUPS},
    )
    option_pred = dict(
        df_data=df_val_scaled,
        use_subject_specific=True,
    )
    list_y_pred_3D_scaled = multi_proc_fit_and_predict_trial(
        trial, options_fit, option_pred
    )

    #################
    # old way: no multiprocessing
    # for model in _get_trial_model_list(trial):
    #     model.fit(
    #         df_train_scaled,
    #         n_iter_improve=3,
    #         fixed_model_options={"warmup": N_WARMUPS},
    #     )
    #     y_pred_3D_scaled = model.predict(
    #         df_val_scaled, use_subject_specific=True
    #     )
    #     list_y_pred_3D_scaled.append(y_pred_3D_scaled)
    #################
    y_pred_3D_scaled_mean = mean(list_y_pred_3D_scaled, axis=0)
    y_val_3D_scaled = df_val_scaled[Y_LABEL]
    return (
        mse(
            remove_warmup_1D(y_pred_3D_scaled_mean, N_WARMUPS),
            remove_warmup_1D(y_val_3D_scaled, N_WARMUPS),
        ),
        trial.params["N"],
    )


def _run_study(
    study_name: str,
    df_train_scaled: DataFrame,
    df_val_scaled: DataFrame,
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
            df_train_scaled,
            df_val_scaled,
        ),
        n_trials=N_STARTUP_TRIALS + N_TPE_TRIALS,
    )


def _run_test_study(
    df_train_scaled: DataFrame,
    df_val_scaled: DataFrame,
) -> None:
    study = create_study(
        study_name="OSEF",
        directions=["minimize"],
        load_if_exists=True,
    )

    study.sampler = samplers.RandomSampler(seed=0)
    study.optimize(
        lambda x: _optuna_objective(x, df_train_scaled, df_val_scaled),
        n_trials=2,
    )


def run_optimization(opti_idx: int) -> None:

    df_1 = get_dataframe("../../../data/synthetic_bph_1/simulation1.csv")
    df_2 = get_dataframe("../../../data/synthetic_bph_1/simulation2.csv")

    scaler = SCALER()
    all_labels = [SERIES, TSTEPS] + X_LABELS + [Y_LABEL]

    #
    if opti_idx == 1:
        study_name = "HP-optimization-01"  # !!!
        df_train, df_test = df_1, df_2
        df_train_scaled = DataFrame(columns=all_labels)
        df_test_scaled = DataFrame(columns=all_labels)
        df_train_scaled[all_labels] = scaler.fit_transform(
            df_train[all_labels]
        )
        df_test_scaled[all_labels] = scaler.transform(df_test[all_labels])

        _run_study(study_name, df_train_scaled, df_test_scaled)

    elif opti_idx == 2:
        study_name = "HP-optimization-02"  # !!!

        df_train, df_test = df_2, df_1
        df_train_scaled = DataFrame(columns=all_labels)
        df_test_scaled = DataFrame(columns=all_labels)
        df_train_scaled[all_labels] = scaler.fit_transform(
            df_train[all_labels]
        )
        df_test_scaled[all_labels] = scaler.transform(df_test[all_labels])

        _run_study(study_name, df_train_scaled, df_test_scaled)

    else:
        raise UserWarning("Asshole!")
