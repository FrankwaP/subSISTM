# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Callable
from types import ModuleType
from json import load

from numpy.typing import NDArray
from numpy import array
from pandas import DataFrame, concat
from sklearn.metrics import (
    mean_absolute_error as mae,
    mean_squared_error as mse,
)


from .data import get_dataframe, prepare_data
from .reservoirs import get_esn_model_list, ESN, ReservoirEnsemble, Model

# from .hp_optimization import adpapt_tensors_if_ypred
from .global_config import (
    TSTEPS,
    N_WARMUPS,
    SCALER,
    DATA_DIR,
    TEST_FILE,
    SIMU_PATTERN,
    HP_JSON_FILE,
    PRED_CSV_FILE,
    SORT_COLUMNS,
)


# def _get_model_list() -> list[ESN]:

#     hp_json = load(open(HP_JSON_FILE, "r"))

#     reservoir_kwargs = {
#         k: v
#         for k, v in hp_json.items()
#         if k in ["N", "sr", "lr", "input_scaling"]
#     }
#     reservoir_kwargs["units"] = reservoir_kwargs.pop("N")

#     ridge_kwargs = {k: v for k, v in hp_json.items() if k in ["ridge"]}

#     esn_kwargs = {
#         k: v for k, v in hp_json.items() if k in ["use_raw_inputs", "feedback"]
#     }

#     return get_esn_model_list(reservoir_kwargs, ridge_kwargs, esn_kwargs)


# def _train_pred_simu(
#     simulation_name: str,
#     tsteps: NDArray,
#     model_list: list[ESN],
#     x_train_3D_scaled: NDArray,
#     y_train_3D_scaled: NDArray,
#     x_test_3D_scaled: NDArray,
#     inverse_transform_y_pred_3D_scaled: Callable,
#     inverse_transform_y_pred_train_3D_scaled: Callable,
#     n_warmups_offset: float,
# ) -> DataFrame:
#     list_df_result = []
#     for iseed, model in enumerate(model_list):
#         model.fit(
#             x_train_3D_scaled,
#             y_train_3D_scaled,
#             warmup=N_WARMUPS + n_warmups_offset,
#         )
#         for name_x, x_3D_scaled, inv_func in [
#             (
#                 "train",
#                 x_train_3D_scaled,
#                 inverse_transform_y_pred_train_3D_scaled,
#             ),
#             (
#                 "test",
#                 x_test_3D_scaled,
#                 inverse_transform_y_pred_3D_scaled,
#             ),
#         ]:
#             y_pred_3D_scaled = model.run(x_3D_scaled)
#             if isinstance(y_pred_3D_scaled, list):
#                 y_pred_3D_scaled = array(y_pred_3D_scaled)
#             df_pred = inv_func(y_pred_3D_scaled)
#             df_pred["simulation"] = simulation_name
#             df_pred["time"] = tsteps
#             df_pred["iseed"] = iseed
#             df_pred["dataset"] = name_x
#             list_df_result.append(df_pred)

#     return concat(list_df_result)


def _get_model() -> list[ESN]:

    hp_json = load(open(HP_JSON_FILE, "r"))

    assert ("use_raw_inputs" not in hp_json) or (
        hp_json["use_raw_inputs"] is False
    )
    assert ("feedback" not in hp_json) or (hp_json["feedback"] is False)

    reservoir_kwargs = {
        k: v
        for k, v in hp_json.items()
        if k in ["N", "sr", "lr", "input_scaling"]
    }
    reservoir_kwargs["units"] = reservoir_kwargs.pop("N")
    ridge_kwargs = {k: v for k, v in hp_json.items() if k in ["ridge"]}
    return ReservoirEnsemble(reservoir_kwargs, ridge_kwargs)


def _train_pred_simu(
    study_config: ModuleType,
    model: Model,
    df_train: DataFrame,
    df_test: DataFrame,
) -> DataFrame:

    (
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
        y_train_2D,
        y_test_2D,
        inverse_transform_y_test_3D_scaled,
        inverse_transform_y_train_3D_scaled,
    ) = prepare_data(
        df_train=df_train,
        df_test=df_test,  # !!!
        x_labels=study_config.X_LABELS,
        y_labels=study_config.Y_LABELS,
        x_scaler=SCALER(),
        y_scaler=SCALER(),
    )

    model.fit(
        x_train_3D_scaled,
        y_train_3D_scaled,
        warmup=N_WARMUPS,
    )

    list_dict_result = []
    for dataset, x_3D_scaled, y_2D, inv_func in [
        (
            "train",
            x_train_3D_scaled,
            y_train_2D,
            inverse_transform_y_train_3D_scaled,
        ),
        (
            "test",
            x_test_3D_scaled,
            y_test_2D,
            inverse_transform_y_test_3D_scaled,
        ),
    ]:
        y_pred_3D_scaled = model.run(x_3D_scaled)
        if isinstance(y_pred_3D_scaled, list):
            y_pred_3D_scaled = array(y_pred_3D_scaled)

        y_pred_2D = inv_func(y_pred_3D_scaled)
        assert y_pred_2D[SORT_COLUMNS].equals(y_2D[SORT_COLUMNS])
        for metname, metric in [("mae", mae), ("mse", mse)]:
            list_dict_result.append(
                {
                    "dataset": dataset,
                    "metric": metname,
                    "value": metric(
                        y_pred_2D[study_config.Y_LABELS],
                        y_2D[study_config.Y_LABELS],
                    ),
                }
            )

    return DataFrame(list_dict_result)


def train_pred_loop(study_config: ModuleType):

    df_test = get_dataframe(
        DATA_DIR + "/" + TEST_FILE,
        x_labels=study_config.X_LABELS,
        y_labels=study_config.Y_LABELS,
    )

    results_df = DataFrame()
    for file_simu in Path(DATA_DIR).glob(SIMU_PATTERN):
        simulation_name = file_simu.name

        print(f"\n{simulation_name}")
        # reinitialize the model
        model = _get_model()

        # load the simulation data
        df_simu = get_dataframe(
            file_simu,
            x_labels=study_config.X_LABELS,
            y_labels=study_config.Y_LABELS,
        )

        df_simu = _train_pred_simu(
            study_config,
            model,
            df_simu,
            df_test,
        )

        df_simu["simulation"] = simulation_name

        results_df = concat([results_df, df_simu])

    results_df.to_csv(PRED_CSV_FILE, index=False)
