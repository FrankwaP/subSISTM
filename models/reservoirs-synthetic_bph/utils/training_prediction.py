# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable
from types import ModuleType
from json import load

from numpy.typing import NDArray
from numpy import array
from pandas import DataFrame, concat

from sklearn.metrics import (  # type:ignore
    mean_squared_error as mse,  # type:ignore
    mean_absolute_error as mae,  # type:ignore
)  # type:ignore

from .data import get_dataframe, prepare_data
from .reservoirs import get_esn_model_list, ESN
from .global_config import (
    SERIES_COLUMN_NAME,
    TIMESTEPS_COLUMN_NAME,
    N_WARMUPS,
    SCALER,
    DATA_DIR,
    TEST_FILE,
    SIMU_PATTERN,
    HP_JSON_FILE,
    PRED_CSV_FILE,
    PRED_CSV_FILE_TRAIN,
)


def _get_model_list() -> list[ESN]:

    hp_json = load(open(HP_JSON_FILE, "r"))

    reservoir_kwargs = {
        k: v
        for k, v in hp_json.items()
        if k in ["N", "sr", "lr", "input_scaling"]
    }
    reservoir_kwargs["units"] = reservoir_kwargs.pop("N")

    ridge_kwargs = {k: v for k, v in hp_json.items() if k in ["ridge"]}

    esn_kwargs = {
        k: v for k, v in hp_json.items() if k in ["use_raw_inputs", "feedback"]
    }

    return get_esn_model_list(reservoir_kwargs, ridge_kwargs, esn_kwargs)


def _train_pred_simu(
    simulation_name: str,
    tsteps: NDArray,
    model_list: list[ESN],
    x_train_3D_scaled: NDArray,
    y_train_3D_scaled: NDArray,
    x_test_3D_scaled: NDArray,
    inverse_transform_y_pred_3D_scaled: Callable,
) -> DataFrame:

    simu_df = DataFrame()

    for iseed, model in enumerate(model_list):
        model.fit(x_train_3D_scaled, y_train_3D_scaled, warmup=N_WARMUPS)
        y_pred_3D_scaled = model.run(x_test_3D_scaled)
        if isinstance(y_pred_3D_scaled, list):
            y_pred_3D_scaled = array(y_pred_3D_scaled)
        df = inverse_transform_y_pred_3D_scaled(y_pred_3D_scaled)
        df["iseed"] = iseed
        simu_df = concat([simu_df, df])
    simu_df["simulation"] = simulation_name
    return simu_df


# def _train_pred_simu(
#     model_list: list[ESN],
#     x_train_3D_scaled: NDArray,
#     y_train_3D_scaled: NDArray,
#     x_test_3D_scaled: NDArray,
#     y_test_2D: NDArray,
#     y_test_2D_obs: NDArray,
#     inverse_transform_y_pred_3D_scaled: Callable,
#     remove_warmup_test_or_pred_2D: Callable,
# ) -> dict[str, float]:

#     y_test_2D_ = remove_warmup_test_or_pred_2D(y_test_2D, N_WARMUPS)

#     result_dict = {}

#     for iseed, model in enumerate(model_list):
#         model.fit(x_train_3D_scaled, y_train_3D_scaled, warmup=N_WARMUPS)
#         y_pred_3D_scaled = model.run(x_test_3D_scaled)
#         if isinstance(y_pred_3D_scaled, list):
#             y_pred_3D_scaled = array(y_pred_3D_scaled)
#         y_pred_2D = inverse_transform_y_pred_3D_scaled(y_pred_3D_scaled)
#         y_pred_2D_ = remove_warmup_test_or_pred_2D(y_pred_2D, N_WARMUPS)

#         result_dict[f"mse-seed-{iseed:02d}"] = mse(y_pred_2D_, y_test_2D_)
#         result_dict[f"mae-seed-{iseed:02d}"] = mae(y_pred_2D_, y_test_2D_)

#     return result_dict


def train_pred_loop(study_config: ModuleType):

    df_test = get_dataframe(DATA_DIR + "/" + TEST_FILE)
    tsteps = df_test[TIMESTEPS_COLUMN_NAME].to_numpy()

    results_df = DataFrame()
    for file_simu in Path(DATA_DIR).glob(SIMU_PATTERN):
        simulation_name = file_simu.name

        print(f"\n{simulation_name}")
        # reinitialize the models
        model_list = _get_model_list()

        # load the simulation data
        df_simu = get_dataframe(file_simu)

        (
            x_train_3D_scaled,
            y_train_3D_scaled,
            x_test_3D_scaled,
            _,
            inverse_transform_y_pred_3D_scaled,
        ) = prepare_data(
            df_train=df_simu,
            df_test=df_test,
            x_labels=study_config.X_LABELS,
            y_labels=study_config.Y_LABELS,
            x_scaler=SCALER(),
            y_scaler=SCALER(),
        )

        df_pred_simu = _train_pred_simu(
            simulation_name,
            tsteps,
            model_list,
            x_train_3D_scaled,
            y_train_3D_scaled,
            x_test_3D_scaled,
            inverse_transform_y_pred_3D_scaled,
        )

        results_df = concat([results_df, df_pred_simu])

    results_df.to_csv(PRED_CSV_FILE, index=False)


def train_pred_loop_omg_so_ugly(study_config: ModuleType):

    df_test = get_dataframe(DATA_DIR + "/" + TEST_FILE)
    tsteps = df_test[TIMESTEPS_COLUMN_NAME].to_numpy()

    results_df = DataFrame()
    for file_simu in Path(DATA_DIR).glob(SIMU_PATTERN):
        simulation_name = file_simu.name

        print(f"\n{simulation_name}")
        # reinitialize the models
        model_list = _get_model_list()

        # load the simulation data
        df_simu = get_dataframe(file_simu)

        (
            x_train_3D_scaled,
            y_train_3D_scaled,
            x_test_3D_scaled,
            _,
            inverse_transform_y_pred_3D_scaled,
        ) = prepare_data(
            df_train=df_simu,
            df_test=df_simu,  # !!!
            x_labels=study_config.X_LABELS,
            y_labels=study_config.Y_LABELS,
            x_scaler=SCALER(),
            y_scaler=SCALER(),
        )

        df_pred_simu = _train_pred_simu(
            simulation_name,
            tsteps,
            model_list,
            x_train_3D_scaled,
            y_train_3D_scaled,
            x_test_3D_scaled,
            inverse_transform_y_pred_3D_scaled,
        )

        results_df = concat([results_df, df_pred_simu])

    results_df.to_csv(PRED_CSV_FILE_TRAIN, index=False)  # !!!
