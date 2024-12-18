from sys import path

path.append("..")

import __main__
from pathlib import Path
from os import chdir, makedirs
from itertools import repeat
from multiprocessing import Pool, current_process

from numpy.typing import NDArray
from reservoirpy.nodes import Reservoir, Ridge

from optuna import Trial
from global_config import N_SEEDS


from mixed_ml.mixed_ml import MixedMLEstimator


def _get_trial_model_list(trial: Trial) -> list[MixedMLEstimator]:

    # before the loop!!!
    reservoir_dict = dict(
        units=trial.suggest_int("N", 10, 200),
        sr=trial.suggest_float("sr", 1e-4, 1e1, log=True),
        lr=trial.suggest_float("lr", 1e-4, 1e0, log=True),
        input_scaling=trial.suggest_float(
            "input_scaling", 1e-1, 1e1, log=True
        ),
    )
    ridge_dict = dict(
        ridge=trial.suggest_float("ridge", 1e-8, 1e2, log=True),
    )

    list_model = []
    for seed in range(42, 42 + N_SEEDS):
        reservoir = Reservoir(**reservoir_dict, seed=seed)
        ridge = Ridge(**ridge_dict)
        esn = reservoir >> ridge
        list_model.append(MixedMLEstimator(esn, recurrent_model=True))
    return list_model


def _mp_initializer():
    proc = current_process()
    main_name = Path(__main__.__file__).stem
    proc.name = f"{main_name}__{proc.name}"


def _mp_fit_and_predict(
    model: MixedMLEstimator, options_fit: dict, options_pred: dict
) -> NDArray:
    name = current_process().name
    makedirs(name + "/results", exist_ok=True)
    chdir(name)
    model.fit(**options_fit)
    return model.predict(**options_pred)


def multi_proc_fit_and_predict_trial(
    trial: Trial, options_fit: dict, options_pred: dict
) -> list[NDArray]:

    model_list = _get_trial_model_list(trial)
    with Pool(N_SEEDS, initializer=_mp_initializer) as p:
        return p.starmap(
            _mp_fit_and_predict,
            zip(model_list, repeat(options_fit), repeat(options_pred)),
        )
