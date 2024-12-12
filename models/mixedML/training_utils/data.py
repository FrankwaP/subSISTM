from typing import Union
from pathlib import Path

from numpy.typing import NDArray
from pandas import read_csv, DataFrame


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
    return read_csv(filename, sep=";", decimal=",", dtype=DTYPES)


def remove_warmup_3D(array_3D: NDArray, n_warmups: int) -> NDArray:
    return array_3D[:, n_warmups:, :]


def remove_warmup_1D(array_1D: NDArray, n_warmups: int) -> NDArray:
    return array_1D[n_warmups:]
