import numpy as np
import pandas as pd

from tools.time_series_modules import TimeSerieShapeChanger


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


DF = DF.sort_values(by=[TIMESTEP_NAME])  # to fuck around


# def test_easy_time_serie() -> None:

#     tsdf = TimeSeriesDataFrame(DF, SERIE_NAME, TIMESTEP_NAME)

#     arr_test = tsdf.get_3D_array(OTHER_NAMES)
#     assert np.all(arr_test == ARR)

#     iser, istep, ifeat = 2, 1, 0
#     arr_test[iser, istep, ifeat] = 666
#     tsdf.set_from_3D_array(OTHER_NAMES, arr_test)
#     val_test = tsdf[(tsdf[SERIE_NAME] == iser) & (tsdf[TIMESTEP_NAME] == istep)][
#         OTHER_NAMES[ifeat]
#     ]
#     assert val_test.shape == (1,) and val_test.values[0] == 666


def test_time_serie_shape_changer() -> None:

    shpchng = TimeSerieShapeChanger(SERIE_NAME, TIMESTEP_NAME)
    arr3D = shpchng.df2D_to_array3D(DF)

    for iserie, serie in enumerate(DF[SERIE_NAME].unique()):
        for itime, time in enumerate(DF[TIMESTEP_NAME].unique()):
            df_features = DF[(DF[SERIE_NAME] == serie) & (DF[TIMESTEP_NAME] == time)][
                OTHER_NAMES
            ].to_numpy()

            arr_features = arr3D[iserie, itime, :]

            assert np.all(df_features == arr_features)

    df2D = shpchng.array3D_to_df2D(arr3D)

    assert df2D.equals(DF.sort_values([SERIE_NAME, TIMESTEP_NAME]))
