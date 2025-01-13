# -*- coding: utf-8 -*-
from pathlib import Path

from types import ModuleType
from json import load

from numpy import array
from pandas import DataFrame, concat


from .data import get_dataframe, prepare_data
from .reservoirs import ReservoirEnsemble

from .global_config import (
    N_WARMUPS,
    SCALER,
    DATA_DIR,
    TEST_FILE,
    SIMU_PATTERN,
    HP_JSON_FILE,
    PRED_CSV_FILE,
)

# Columns names
SIMS = "simulation"
DSET = "dataset"
TRAIN = "train"
TEST = "test"


def _get_model() -> ReservoirEnsemble:

    with open(HP_JSON_FILE, "r", encoding="utf-8") as fjson:
        hp_json = load(fjson)

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
    model: ReservoirEnsemble,
    df_train: DataFrame,
    df_test: DataFrame,
) -> DataFrame:

    (
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        _,
        inverse_transform_y_test_3D_scaled,
        inverse_transform_y_train_3D_scaled,
    ) = prepare_data(
        df_train=df_train,
        df_test=df_test,
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

    df_result = DataFrame()
    for dataset, x_3D_scaled, inv_func in [
        (
            TRAIN,
            x_train_3D_scaled,
            inverse_transform_y_train_3D_scaled,
        ),
        (
            TEST,
            x_test_3D_scaled,
            inverse_transform_y_test_3D_scaled,
        ),
    ]:
        y_pred_3D_scaled = model.run(x_3D_scaled)
        if isinstance(y_pred_3D_scaled, list):
            y_pred_3D_scaled = array(y_pred_3D_scaled)
        y_pred_2D = inv_func(y_pred_3D_scaled)

        y_pred_2D[DSET] = dataset
        df_result = concat([df_result, y_pred_2D])

    return df_result


def train_pred_loop(study_config: ModuleType):

    df_test = get_dataframe(
        DATA_DIR + "/" + TEST_FILE,
        x_labels=study_config.X_LABELS,
        y_labels=study_config.Y_LABELS_TGT,
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
            y_labels=study_config.Y_LABELS_TGT,
        )

        df_simu = _train_pred_simu(
            study_config,
            model,
            df_simu,
            df_test,
        )

        df_simu[SIMS] = simulation_name

        results_df = concat([results_df, df_simu])

    results_df.to_csv(PRED_CSV_FILE, index=False)
