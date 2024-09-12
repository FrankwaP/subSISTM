import numpy as np
import pandas as pd

from tools.time_series_modules import TimeSerieShapeChanger


serie_column_name = "person"
timestep_column_name = "time"

df = pd.DataFrame.from_dict(
    {
        serie_column_name: [0, 0, 1, 1, 2, 2],
        timestep_column_name: [0, 1, 0, 1, 0, 1],
        "x1": [1, 2, 3, 4, 5, 6],
        "x2": [7, 8, 9, 10, 11, 12],
        "x3": [13, 14, 15, 16, 17, 18],
    }
)

others_column_names = [
    c for c in df.columns if c not in [serie_column_name, timestep_column_name]
]


def test_time_serie_shape_changer() -> None:

    shpchng = TimeSerieShapeChanger(serie_column_name, timestep_column_name)
    arr3D = shpchng.df2D_to_array3D(df)

    for iserie, serie in enumerate(df[serie_column_name].unique()):
        for itime, time in enumerate(df[timestep_column_name].unique()):
            df_features = df[
                (df[serie_column_name] == serie) & (df[timestep_column_name] == time)
            ][others_column_names].to_numpy()

            arr_features = arr3D[iserie, itime, :]

            assert np.all(df_features == arr_features)

    df2D = shpchng.array3D_to_df2D(arr3D)

    assert df2D.equals(df)
