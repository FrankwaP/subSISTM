# -*- coding: utf-8 -*-
import sys
from pathlib import Path

# from types import ModuleType
from json import load

from numpy import array
from pandas import DataFrame, concat


def add_path(p: str) -> None:
    pth = Path(p).resolve().as_posix()
    print("Added to path:", pth)
    sys.path.append(pth)


add_path("../../")

from mixedML.mixed_ml.mixed_ml import (
    MixedMLEstimator,
    X_LABELS,
    Y_LABEL,
)
from reservoirs_synthetic_bph.utils.data import get_dataframe, prepare_data
from reservoirs_synthetic_bph.utils.reservoirs import ReservoirEnsemble
from reservoirs_synthetic_bph.utils.global_config import (
    N_WARMUPS,
    SCALER,
    DATA_DIR,
    TEST_FILE,
    SIMU_PATTERN,
    HP_JSON_FILE,
    PRED_CSV_FILE,
)

# same as in the R script
SIMS = "simulation"
DSET = "dataset"
TRAIN = "train"
TEST = "test"


def _get_model(study_name: str) -> MixedMLEstimator:

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
    reservoir = ReservoirEnsemble(reservoir_kwargs, ridge_kwargs)

    return MixedMLEstimator(
        reservoir, recurrent_model=True, specific_dir=study_name
    )


def _train_pred_simu(
    model: MixedMLEstimator,
    df_train: DataFrame,
    df_test: DataFrame,
) -> DataFrame:

    scaler = SCALER()
    all_labels = X_LABELS + [Y_LABEL]

    df_train_scaled = df_train.copy()
    df_test_scaled = df_test.copy()
    df_train_scaled[all_labels] = scaler.fit_transform(df_train[all_labels])
    df_test_scaled[all_labels] = scaler.transform(df_test[all_labels])

    model.fit(
        df_train_scaled,
        n_iter_improv=1,
        min_rltv_imrov=0.01,
        trial_for_pruning=None,
        fixed_model_fit_options={"warmup": N_WARMUPS},
    )

    df_result = DataFrame()
    for dataset, df_scaled in [
        (
            TRAIN,
            df_train_scaled,
        ),
        (
            TEST,
            df_test_scaled,
        ),
    ]:
        y_pred_3D_scaled = model.run(df_scaled, use_subject_specific=True)
        if isinstance(y_pred_3D_scaled, list):
            y_pred_3D_scaled = array(y_pred_3D_scaled)

        df_inv_trans = df_scaled[all_labels].copy()
        df_inv_trans[Y_LABEL] = y_pred_3D_scaled

        df_inv_trans[all_labels] = scaler.inverse_transform(
            df_inv_trans[all_labels]
        )
        y_pred_2D = df_inv_trans[Y_LABEL]

        y_pred_2D[DSET] = dataset
        df_result = concat([df_result, y_pred_2D])

    return df_result


def train_pred_loop(study_name: str):

    df_test = get_dataframe(
        DATA_DIR + "/" + TEST_FILE,
        x_labels=X_LABELS,
        y_labels=[Y_LABEL],
    )

    results_df = DataFrame()
    for file_simu in Path(DATA_DIR).glob(SIMU_PATTERN):
        simulation_name = file_simu.name

        print(f"\n{simulation_name}")
        # reinitialize the model
        model = _get_model(study_name)

        # load the simulation data
        df_simu = get_dataframe(
            file_simu,
            x_labels=X_LABELS,
            y_labels=[Y_LABEL],
        )

        df_simu = _train_pred_simu(
            model,
            df_simu,
            df_test,
        )

        df_simu[SIMS] = simulation_name

        results_df = concat([results_df, df_simu])

    results_df.to_csv(PRED_CSV_FILE, index=False)
