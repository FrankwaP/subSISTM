# -*- coding: utf-8 -*-
from pathlib import Path
from types import ModuleType

from pandas import read_csv, DataFrame, merge, concat
from sklearn.metrics import (  # type: ignore
    mean_absolute_error as mae,
    mean_squared_error as mse,
)

from .global_config import (
    SERIES,
    TSTEPS,
    N_WARMUPS,
    DATA_DIR,
    TEST_FILE,
    SIMU_PATTERN,
    PRED_CSV_FILE,
    METRIC_CSV_FILE,
)
from .training_prediction import DSET, TRAIN, TEST, SIMS

SIM_PREF = "simulation"
PRED = "pred"
TGT = "target"
METRIC = "metric"
VAL = "value"
QUADBIAS = "quad-bias"
VAR = "variance"


def _remove_warmup(df: DataFrame) -> DataFrame:
    tsteps = sorted(df[TSTEPS].unique())
    return df[df[TSTEPS] >= tsteps[N_WARMUPS]]


def _get_study_name(study_config: ModuleType) -> str:
    return Path(study_config.__file__).parent.name


# %% datasets loading
def _get_train_df(study_config: ModuleType) -> DataFrame:
    df_out = DataFrame()
    for file_simu in Path(DATA_DIR).glob(SIMU_PATTERN):
        simulation_name = file_simu.name
        df_simu = read_csv(file_simu, sep=";", decimal=",")
        df_simu = df_simu[[SERIES, TSTEPS] + study_config.Y_LABELS_TGT]
        df_simu[SIMS] = simulation_name
        df_out = concat([df_out, df_simu])
    df_out[DSET] = TRAIN
    return _remove_warmup(df_out)


def _get_test_df(study_config: ModuleType) -> DataFrame:
    filename = DATA_DIR + "/" + TEST_FILE
    df = read_csv(filename, sep=";", decimal=",")
    df = df[[SERIES, TSTEPS] + study_config.Y_LABELS_TGT]
    df[DSET] = TEST
    return _remove_warmup(df)


def _get_pred_df(study_config: ModuleType) -> DataFrame:
    filename = Path(study_config.__file__).parent / Path(PRED_CSV_FILE)
    df_pred = read_csv(filename)
    assert len(study_config.Y_LABELS)
    y_pred = study_config.Y_LABELS[0]
    df_pred = df_pred.rename(columns={y_pred: PRED})
    # !!! with the (y-1) covariate, the warmup steps are offset by one
    return _remove_warmup(df_pred)


def _get_full_merged_df(
    df_train: DataFrame, df_test: DataFrame, df_pred: DataFrame
) -> DataFrame:
    df_merge_train = merge(df_train, df_pred, on=[SERIES, TSTEPS, DSET, SIMS])
    df_merge_test = merge(df_test, df_pred, on=[SERIES, TSTEPS, DSET])
    assert len(df_merge_train) == len(df_merge_test) == len(df_pred) // 2
    return concat([df_merge_train, df_merge_test])


# %%
def _get_metrics(
    study_config: ModuleType,
    df_train: DataFrame,
    df_test: DataFrame,
    df_pred: DataFrame,
) -> DataFrame:
    df_merged = _get_full_merged_df(df_train, df_test, df_pred)
    list_dict = []
    for dset in (TRAIN, TEST):
        df = df_merged[df_merged[DSET] == dset]
        for ytgt in study_config.Y_LABELS_TGT:
            for met in (mae, mse):
                list_dict.append(
                    {
                        DSET: dset,
                        TGT: ytgt,
                        METRIC: met.__name__,
                        VAL: met(df[PRED], df[ytgt]),
                    }
                )
    return DataFrame(list_dict)


# %%
def _compute_squared_bias(
    study_config: ModuleType, df_test: DataFrame, df_pred_test: DataFrame
) -> list[dict]:
    df_mean = df_pred_test.groupby([SERIES, TSTEPS])[PRED].mean()
    df_merged = merge(df_mean, df_test, on=[SERIES, TSTEPS])
    assert len(df_mean) == len(df_merged)
    return [
        {
            DSET: TEST,
            METRIC: QUADBIAS,
            TGT: ytgt,
            VAL: (df_merged[PRED] - df_merged[ytgt]).pow(2).mean(),
        }
        for ytgt in study_config.Y_LABELS_TGT
    ]


def _compute_variance(df_pred_test: DataFrame) -> dict:
    return {
        DSET: TEST,
        METRIC: VAR,
        VAL: df_pred_test.groupby([SERIES, TSTEPS])[PRED].var().mean(),
    }


def _get_bias_and_variance(
    study_config: ModuleType,
    df_test: DataFrame,
    df_pred: DataFrame,
) -> DataFrame:
    df_pred_test = df_pred[df_pred[DSET] == TEST]
    list_dict = _compute_squared_bias(study_config, df_test, df_pred_test)
    list_dict.append(_compute_variance(df_pred_test))
    return DataFrame(list_dict)


# %%
def process(study_config: ModuleType):
    df_train = _get_train_df(study_config)
    df_test = _get_test_df(study_config)
    df_pred = _get_pred_df(study_config)
    df_metrics = _get_metrics(study_config, df_train, df_test, df_pred)
    df_bias_var = _get_bias_and_variance(study_config, df_test, df_pred)
    concat([df_metrics, df_bias_var]).to_csv(METRIC_CSV_FILE)
