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
import sys
from pathlib import Path


def add_path(p: str) -> None:
    pth = Path(p).resolve().as_posix()
    print("Added to path:", pth)
    sys.path.append(pth)


add_path("../../")

from pandas import DataFrame
from optuna import Trial, create_study, samplers
from optuna.logging import set_verbosity, DEBUG
from optuna.pruners import MedianPruner

from reservoirpy.observables import mse  # type: ignore

from reservoirs_synthetic_bph.utils.reservoirs import ReservoirEnsemble
from reservoirs_synthetic_bph.utils.data import get_dataframe
from reservoirs_synthetic_bph.utils.global_config import (
    N_WARMUPS,
    SCALER,
    SERIES,
    TSTEPS,
)


from mixedML.mixed_ml.mixed_ml import (
    MixedMLEstimator,
    X_LABELS,
    Y_LABEL,
)


# optuna
N_STARTUP_TRIALS = 50
N_TPE_TRIALS = 50
set_verbosity(DEBUG)


# %%


def _get_trial_model(trial: Trial, study_name: str) -> MixedMLEstimator:
    reservoir_dict = dict(
        units=trial.suggest_int("N", 5, 500),
        sr=trial.suggest_float("sr", 1e-4, 1e1, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e0, log=True),
        input_scaling=trial.suggest_float(
            "input_scaling", 1e-1, 1e1, log=True
        ),
    )
    ridge_dict = dict(
        ridge=trial.suggest_float("ridge", 1e-8, 1e2, log=True),
    )
    return MixedMLEstimator(
        ReservoirEnsemble(reservoir_dict, ridge_dict),
        recurrent_model=True,
        specific_dir=study_name,
    )


def _optuna_objective(
    trial: Trial,
    study_name: str,
    df_train_scaled: DataFrame,
    df_val_scaled: DataFrame,
) -> tuple[float, int]:

    model = _get_trial_model(trial, study_name)
    model.fit(
        df_train_scaled,
        n_iter_improv=1,
        min_rltv_imrov=0.01,
        trial_for_pruning=trial,
        fixed_model_fit_options={"warmup": N_WARMUPS},
    )
    y_pred_3D_scaled = model.predict(df_val_scaled, use_subject_specific=True)
    y_val_3D_scaled = df_val_scaled[Y_LABEL].to_numpy()
    assert y_pred_3D_scaled.ndim == 1
    return mse(
        y_pred_3D_scaled[N_WARMUPS:],
        y_val_3D_scaled[N_WARMUPS:],
    )
    # NotImplementedError: Trial.report is not supported for multi-objective optimization.
    # return (
    #     mse(
    #         y_pred_3D_scaled[N_WARMUPS:],
    #         y_val_3D_scaled[N_WARMUPS:],
    #     ),
    #     trial.params["N"],
    # )


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
        directions=["minimize"],
        # NotImplementedError: Trial.report is not supported for multi-objective optimization.
        # directions=["minimize", "minimize"],
        load_if_exists=True,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )

    study.sampler = samplers.TPESampler(
        n_startup_trials=N_STARTUP_TRIALS, seed=0
    )
    study.optimize(
        lambda x: _optuna_objective(
            x,
            study_name,
            df_train_scaled,
            df_val_scaled,
        ),
        n_trials=N_STARTUP_TRIALS + N_TPE_TRIALS,
    )


def run_optimization(opti_idx: int) -> None:

    df_1 = get_dataframe(
        "../../../../data/synthetic_bph_1/simulation1.csv", X_LABELS, [Y_LABEL]
    )
    df_2 = get_dataframe(
        "../../../../data/synthetic_bph_1/simulation2.csv", X_LABELS, [Y_LABEL]
    )

    #
    if opti_idx == 1:
        study_name = "HP-optimization-01"  # !!!
        df_train, df_test = df_1, df_2  # !!!
    elif opti_idx == 2:
        study_name = "HP-optimization-02"  # !!!
        df_train, df_test = df_2, df_1  # !!!
    else:
        raise UserWarning("Asshole!")

    scaler = SCALER()
    all_labels = X_LABELS + [Y_LABEL]

    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[all_labels] = scaler.fit_transform(df_train[all_labels])
    df_test_scaled[all_labels] = scaler.transform(df_test[all_labels])

    _run_study(study_name, df_train_scaled, df_test_scaled)
