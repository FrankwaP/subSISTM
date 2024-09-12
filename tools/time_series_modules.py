# %%


from typing import Callable, Union


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


# class EasyTimeSeries:

#     def __init__(
#         self,
#         dataframe: pd.DataFrame,
#         *,
#         serie_column_name: str,
#         timestep_column_name: str,
#         feature_column_names: list[str],
#         target_column_names: list[str],
#     ):

#         assert isinstance(dataframe, pd.DataFrame)
#         assert isinstance(serie_column_name, str)
#         assert isinstance(timestep_column_name, str)
#         assert isinstance(feature_column_names, list)
#         assert all(isinstance(c, str) for c in feature_column_names)
#         assert isinstance(target_column_names, list)
#         assert all(isinstance(c, str) for c in target_column_names)

#         assert serie_column_name in dataframe.columns
#         assert timestep_column_name in dataframe.columns
#         assert all(c in dataframe.columns for c in feature_column_names)
#         assert all(c in dataframe.columns for c in target_column_names)

#         filter_columns = (
#             [serie_column_name, timestep_column_name]
#             + feature_column_names
#             + target_column_names
#         )

#         self.dataframe = dataframe[filter_columns].sort_values(
#             by=[serie_column_name, timestep_column_name]
#         )
#         self.serie_column_name = serie_column_name
#         self.timestep_column_name = timestep_column_name
#         self.feature_column_names = feature_column_names
#         self.target_column_names = target_column_names

#         self.series = self.dataframe[self.serie_column_name].unique()
#         self.N_series = len(self.series)

#         self.timesteps = self.dataframe[self.timestep_column_name].unique()
#         self.N_timesteps = len(self.timesteps)

#     def __repr__(self) -> str:
#         return repr(self.dataframe)

#     def train_test_split_on_series(
#         self, test_size=None, train_size=None, random_state=None, shuffle=True
#     ) -> tuple["EasyTimeSeries", "EasyTimeSeries"]:

#         train_series, test_series = train_test_split(
#             self.series,
#             test_size=test_size,
#             train_size=train_size,
#             random_state=random_state,
#             shuffle=shuffle,
#         )

#         train_dataframe = self.dataframe[
#             self.dataframe[self.serie_column_name].isin(train_series)
#         ]
#         test_dataframe = self.dataframe[
#             self.dataframe[self.serie_column_name].isin(test_series)
#         ]

#         return (
#             EasyTimeSeries(
#                 dataframe=train_dataframe,
#                 serie_column_name=self.serie_column_name,
#                 timestep_column_name=self.timestep_column_name,
#                 feature_column_names=self.feature_column_names,
#                 target_column_names=self.target_column_names,
#             ),
#             EasyTimeSeries(
#                 dataframe=test_dataframe,
#                 serie_column_name=self.serie_column_name,
#                 timestep_column_name=self.timestep_column_name,
#                 feature_column_names=self.feature_column_names,
#                 target_column_names=self.target_column_names,
#             ),
#         )

#     def convert_2D_to_3D_array(self, array_2D: np.ndarray) -> np.ndarray:
#         return array_2D.reshape([self.N_series, self.N_timesteps, -1])

#     def convert_3D_to_2D_array(self, array_3D: np.ndarray) -> np.ndarray:
#         return array_3D.reshape([self.N_series * self.N_timesteps, -1])

#     def get_3D_feature_array(self) -> np.ndarray:
#         return self.convert_2D_to_3D_array(
#             self.dataframe[self.feature_column_names].to_numpy()
#         )

#     def get_3D_target_array(self) -> np.ndarray:
#         return self.convert_2D_to_3D_array(
#             self.dataframe[self.target_column_names].to_numpy()
#         )

#     def apply_on_2D(
#         self, function: Callable, columns_names: list[str], in_place: bool
#     ) -> Union[None, np.ndarray]:
#         results = function(self.dataframe[columns_names])
#         if in_place:
#             self.dataframe[columns_names] = results
#         else:
#             return results


def _check_dataframe(
    df: pd.DataFrame, serie_column_name: str, timestep_column_name: str
) -> None:

    N_series = len(df[serie_column_name].unique())
    N_timesteps = len(df[timestep_column_name].unique())

    if len(df) != N_series * N_timesteps:
        raise UserWarning(
            "This tool considers that all the series have the same timesteps (so the dataframe is Nseries x Ntimesteps)."
        )


def train_test_split_on_series(
    df: pd.DataFrame,
    serie_column_name: str,
    timestep_column_name: str,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    _check_dataframe(df, serie_column_name, timestep_column_name)

    train_series, test_series = train_test_split(
        df[serie_column_name].unique(),
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
    )

    train_dataframe = df[df[serie_column_name].isin(train_series)]
    test_dataframe = df[df[serie_column_name].isin(test_series)]

    return (train_dataframe, test_dataframe)


class TimeSerieShapeChanger:

    def __init__(
        self,
        serie_column_name: str,
        timestep_column_name: str,
    ):
        self.serie_column_name = serie_column_name
        self.timestep_column_name = timestep_column_name

    def df2D_to_array3D(self, df: pd.DataFrame) -> np.ndarray:

        _check_dataframe(df, self.serie_column_name, self.timestep_column_name)

        x = df.sort_values(by=[self.serie_column_name, self.timestep_column_name])

        self.series = x[self.serie_column_name]
        self.series_unq = self.series.unique()
        self.timesteps = x[self.timestep_column_name]
        self.timesteps_unq = self.timesteps.unique()

        if len(x) != len(self.series_unq) * len(self.timesteps_unq):
            raise UserWarning(
                "This tool considers that all the series have the same timesteps."
            )

        self.other_column_names = [
            c
            for c in x.columns
            if c not in [self.serie_column_name, self.timestep_column_name]
        ]
        return (
            x[self.other_column_names]
            .to_numpy()
            .reshape(
                [
                    len(self.series_unq),
                    len(self.timesteps_unq),
                    len(self.other_column_names),
                ]
            )
        )

    def array3D_to_df2D(self, arr: np.ndarray) -> pd.DataFrame:

        user_warn = UserWarning(
            "The 2D-to-3D conversion has been done on another type of data so we cannot do the 3D-to-2D."
        )
        try:
            shape_test = (
                len(self.series_unq),
                len(self.timesteps_unq),
                len(self.other_column_names),
            )
        except AttributeError:
            raise user_warn

        if not np.all(arr.shape == shape_test):
            raise user_warn

        df = pd.DataFrame(
            data=arr.reshape(
                [
                    len(self.series_unq) * len(self.timesteps_unq),
                    len(self.other_column_names),
                ]
            ),
            columns=self.other_column_names,
        )

        df[self.serie_column_name] = self.series
        df[self.timestep_column_name] = self.timesteps
        return df[
            [self.serie_column_name, self.timestep_column_name]
            + self.other_column_names
        ]
