import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from MAIN_ import TimeSerieShapeChanger

SERIE_NAME = "person"
TIMESTEP_NAME = "time"

DF = pd.DataFrame.from_dict(
    {
        SERIE_NAME: [0, 0, 1, 1, 2, 2],
        TIMESTEP_NAME: [0, 1, 0, 1, 0, 1],
        "x1": [1, 2, 3, 4, 5, 6],
        "x2": [7, 8, 9, 10, 11, 12],
        "x3": [13, 14, 15, 16, 17, 18],
    }
)

#    person  time  x1  x2  x3
# 0       0     0   1   7  13
# 1       0     1   2   8  14
# 2       1     0   3   9  15
# 3       1     1   4  10  16
# 4       2     0   5  11  17
# 5       2     1   6  12  18

OTHER_NAMES = [c for c in DF.columns if c not in [SERIE_NAME, TIMESTEP_NAME]]

ARR = np.array(
    [  # [0,:]
        [[1, 7, 13], [2, 8, 14]],  # [0, 0, :]  # [0, 1, :]
        # [1,:]
        [[3, 9, 15], [4, 10, 16]],  # [1, 0, :]  # [1, 1, :]
        # [2,:]
        [[5, 11, 17], [6, 12, 18]],  # [2, 0, :]  # [2, 1, :]
    ]
)


incorrect_dim_array3D = np.concatenate([ARR, ARR])
n, t, p = ARR.shape
n_, t_, p_ = incorrect_dim_array3D.shape
assert n_ == n * 2
assert t_ == t_
assert p_ == p


def test_time_serie_shape_changer_00() -> None:

    shpchng = TimeSerieShapeChanger(SERIE_NAME, TIMESTEP_NAME)
    arr3D = shpchng.fit_transform(DF)

    for iserie, serie in enumerate(DF[SERIE_NAME].unique()):
        for itime, time in enumerate(DF[TIMESTEP_NAME].unique()):
            df_features = DF[(DF[SERIE_NAME] == serie) & (DF[TIMESTEP_NAME] == time)][
                OTHER_NAMES
            ].to_numpy()

            arr_features = arr3D[iserie, itime, :]

            assert np.all(df_features == arr_features)

    df2D = shpchng.inverse_transform(arr3D)

    assert df2D.equals(DF.sort_values([SERIE_NAME, TIMESTEP_NAME]))

    with pytest.raises(AssertionError):
        _ = shpchng.inverse_transform(incorrect_dim_array3D)


def test_time_serie_shape_changer_01() -> None:
    shpchng = TimeSerieShapeChanger(SERIE_NAME, TIMESTEP_NAME)

    with pytest.raises(AttributeError):
        _ = shpchng.inverse_transform(ARR)


def test_time_serie_shape_changer_02() -> None:
    shpchng = TimeSerieShapeChanger(SERIE_NAME, TIMESTEP_NAME)

    pipe = Pipeline([("scaler", StandardScaler()), ("shape_changer", shpchng)])

    arr3D = pipe.fit_transform(DF)
    df2D = pipe.inverse_transform(arr3D)

    assert df2D.equals(DF.sort_values([SERIE_NAME, TIMESTEP_NAME]))


if __name__ == "__main__":
    test_time_serie_shape_changer_00()
    test_time_serie_shape_changer_01()
    test_time_serie_shape_changer_02()