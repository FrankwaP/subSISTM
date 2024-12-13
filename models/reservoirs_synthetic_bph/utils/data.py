from typing import NamedTuple, Callable, Union
from pathlib import Path

from numpy.typing import NDArray
from numpy import float32
from pandas import read_csv, DataFrame

from reservoirpy import set_seed, verbosity  # type: ignore
from sklearn.base import TransformerMixin  # type: ignore
from sklearn.utils.validation import check_is_fitted  # type: ignore
from sklearn.exceptions import NotFittedError  # type: ignore

from .global_config import SERIES_COLUMN_NAME, TIMESTEPS_COLUMN_NAME

set_seed(42)
verbosity(0)

FLOAT_DTYPE = "float32"

DTYPES = {
    "x1": FLOAT_DTYPE,
    "x2": FLOAT_DTYPE,
    "x3": FLOAT_DTYPE,
    "x4": FLOAT_DTYPE,
    "x5": FLOAT_DTYPE,
    "x6": FLOAT_DTYPE,
    "x7": FLOAT_DTYPE,
    "x1_obs": FLOAT_DTYPE,
    "x2_obs": FLOAT_DTYPE,
    "x3_obs": FLOAT_DTYPE,
    "x4_obs": FLOAT_DTYPE,
    "x5_obs": FLOAT_DTYPE,
    "x6_obs": FLOAT_DTYPE,
    "x7_obs": FLOAT_DTYPE,
    "x2_x5": FLOAT_DTYPE,
    "x4_x7": FLOAT_DTYPE,
    "x6_x8": FLOAT_DTYPE,
    "x8": "category",
    "y_mixed": FLOAT_DTYPE,
    "y_mixed_obs": FLOAT_DTYPE,
    "y_fixed": FLOAT_DTYPE,
    "y_fixed_obs": FLOAT_DTYPE,
}


def get_dataframe(filename: Union[str, Path]) -> DataFrame:
    df = read_csv(filename, sep=";", decimal=",", dtype=DTYPES)
    for ylab in ["y_mixed", "y_mixed_obs", "y_fixed", "y_fixed_obs"]:
        df[ylab + "-1"] = df.groupby(SERIES_COLUMN_NAME)[ylab].shift(+1)
    return df


def remove_warmup_df(df: DataFrame, n_warmups: int) -> DataFrame:
    tsteps = sorted(df[TIMESTEPS_COLUMN_NAME].unique())
    return df[df[TIMESTEPS_COLUMN_NAME] > tsteps[n_warmups - 1]]


def remove_warmup_3D(array_3D: NDArray, n_warmups: int) -> NDArray:
    return array_3D[:, n_warmups:, :]


def _check_fitted(scaler: TransformerMixin) -> bool:
    try:
        check_is_fitted(scaler)
    except NotFittedError:
        return False
    return True


class _Data(NamedTuple):
    x_train_3D_scaled: NDArray
    y_train_3D_scaled: NDArray
    x_test_3D_scaled: NDArray
    y_test_3D_scaled: NDArray  # for optuna val MSE
    inverse_transform_pred: Callable  # for final (real) MSE


def prepare_data(
    *,
    df_train: DataFrame,
    df_test: DataFrame,
    x_labels: list[str],
    y_labels: list[str],
    x_scaler: TransformerMixin,
    y_scaler: TransformerMixin,
) -> _Data:

    Nx = len(x_labels)
    Ny = len(y_labels)
    sort_columns = [SERIES_COLUMN_NAME, TIMESTEPS_COLUMN_NAME]

    df_train.sort_values(sort_columns, inplace=True)
    Ns_train = len(df_train[SERIES_COLUMN_NAME].unique())
    Nt_train = len(df_train[TIMESTEPS_COLUMN_NAME].unique())
    x_train_3D_scaled = x_scaler.fit_transform(df_train[x_labels]).reshape(
        (Ns_train, Nt_train, Nx)
    )
    y_train_3D_scaled = y_scaler.fit_transform(df_train[y_labels]).reshape(
        (Ns_train, Nt_train, Ny)
    )
    print(f"Train set is {Ns_train} individuals")

    df_test.sort_values(sort_columns, inplace=True)
    Ns_test = len(df_test[SERIES_COLUMN_NAME].unique())
    Nt_test = len(df_test[TIMESTEPS_COLUMN_NAME].unique())
    x_test_3D_scaled = x_scaler.transform(df_test[x_labels]).reshape(
        (Ns_test, Nt_test, Nx)
    )
    y_test_3D_scaled = y_scaler.transform(df_test[y_labels]).reshape(
        (Ns_test, Nt_test, Ny)
    )
    print(f"Train set is {Ns_test} individuals")

    def inverse_transform_y_pred_3D_scaled(
        y_pred_3D_scaled: NDArray,
    ) -> DataFrame:
        df = df_test[[SERIES_COLUMN_NAME, TIMESTEPS_COLUMN_NAME]].copy()
        df["pred"] = y_scaler.inverse_transform(
            y_pred_3D_scaled.reshape((Ns_test * Nt_test, Ny))
        )
        return df

    return _Data(
        x_train_3D_scaled,
        y_train_3D_scaled,
        x_test_3D_scaled,
        y_test_3D_scaled,
        inverse_transform_y_pred_3D_scaled,
    )
