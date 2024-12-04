from types import ModuleType
from pathlib import Path


from pandas import DataFrame, MultiIndex, merge, read_csv

from .data import get_dataframe, remove_warmup_df
from .global_config import (
    SERIES_COLUMN_NAME,
    TIMESTEPS_COLUMN_NAME,
    N_WARMUPS,
    DATA_DIR,
    TEST_FILE,
    PRED_CSV_FILE,
)


def compute_errors_df(study_config: ModuleType) -> DataFrame:
    study_path = Path(study_config.__file__).parent

    study_code = study_path.name
    # reading the targets
    df_tgt = get_dataframe(DATA_DIR + "/" + TEST_FILE)
    df_tgt = remove_warmup_df(df_tgt, N_WARMUPS)
    df_tgt = df_tgt[
        [SERIES_COLUMN_NAME, TIMESTEPS_COLUMN_NAME]
        + study_config.Y_LABELS_PRED
    ]
    df_tgt = df_tgt.set_index([SERIES_COLUMN_NAME, TIMESTEPS_COLUMN_NAME])

    # reading the prediction
    df_pred = read_csv(study_path / Path(PRED_CSV_FILE))
    df_pred = remove_warmup_df(df_pred, N_WARMUPS)
    df_pred = df_pred.set_index(
        ["simulation", "iseed", SERIES_COLUMN_NAME, TIMESTEPS_COLUMN_NAME]
    )

    # merging predictions and targets
    tsteps_pred = set(df_pred.index.get_level_values("temps").unique())
    tsteps_tgt = set(df_tgt.index.get_level_values("temps").unique())
    assert tsteps_pred == tsteps_tgt

    indiv_pred = set(df_pred.index.get_level_values("individus").unique())
    indiv_tgt = set(df_tgt.index.get_level_values("individus").unique())
    assert indiv_pred == indiv_tgt
    # we use how='left' because we copy the left index back
    df_mrg = merge(
        left=df_pred, right=df_tgt, on=["individus", "temps"], how="left"
    )
    df_mrg.index = df_pred.index

    # computing the error for noise/no noise targets
    for col in study_config.Y_LABELS_PRED:
        # petite vérif du merge… normalement les targets sont dupliquées pour chaque simulation
        assert all(
            df_mrg[[col]].groupby(["iseed", "individus", "temps"]).std() == 0
        )
        #
        df_mrg["error_" + col] = df_mrg[col] - df_mrg["pred"]
    # print(df_mrg)

    # mean on the seeds and simulation
    # we have the same seed number of seed for each indiv/tsep, so  mean(mean) == mean
    df_dict = {}

    for col in study_config.Y_LABELS_PRED:
        df_dict[(col, "mae")] = (
            df_mrg["error_" + col].abs().groupby("simulation").mean().mean()
        )
        df_dict[(col, "mse")] = (
            df_mrg["error_" + col].pow(2).groupby("simulation").mean().mean()
        )

    df_final = DataFrame(
        [df_dict],
        columns=MultiIndex.from_tuples(df_dict.keys()),
        index=[study_code],
    )

    return df_final
