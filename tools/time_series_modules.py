# %%


from typing import Callable, Union, NamedTuple


from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def _get_and_check_series_and_timesteps(
    df: pd.DataFrame, serie_column_name: str, timestep_column_name: str
) -> tuple[np.ndarray, np.ndarray]:

    print('Checking the "series x timesteps" hypothesis…')

    series_unq = df[serie_column_name].unique()
    N_series = len(series_unq)
    timesteps_unq = df[timestep_column_name].unique()
    N_timesteps = len(timesteps_unq)

    if len(df) != N_series * N_timesteps:
        raise UserWarning(
            "This tool considers that all the series have the same timesteps (so the dataframe is Nseries x Ntimesteps)."
        )

    print(
        f"The dataframe has {N_series} unique series and {N_timesteps} unique timesteps, for a total of {len(df)}=={N_series}x{N_timesteps}"
    )
    return (series_unq, timesteps_unq)


# class TimeSeriesDataFrame(pd.DataFrame):

#     def __init__(
#         self,
#         dataframe: pd.DataFrame,
#         serie_column_name: str,
#         timestep_column_name: str,
#     ):
#         print("Checking the input…")
#         assert isinstance(dataframe, pd.DataFrame)

#         assert isinstance(serie_column_name, str)
#         assert serie_column_name in dataframe.columns

#         assert isinstance(timestep_column_name, str)
#         assert timestep_column_name in dataframe.columns

#         sort_columns = [serie_column_name, timestep_column_name]
#         super().__init__(dataframe.sort_values(by=sort_columns))

#         self.series, self.timesteps = _get_and_check_series_and_timesteps(
#             dataframe, serie_column_name, timestep_column_name
#         )

#         self.serie_column_name = serie_column_name
#         self.timestep_column_name = timestep_column_name

#     def __getitem__(self, key):
#         return TimeSeriesDataFrame(
#             super().__getitem__(key), self.serie_column_name, self.timestep_column_name
#         )

#     ######
#     # trick to handle the fact that after initialization (or instantiation?) "dataframe.name" is equivalent to "dataframe['name']"
#     # so we cannot create new object attributes using "self.attribute = something" and we need to work with "self.__dict__["attribute"]"
#     # also for some reason inheriting from pd.Dataframe make setting a value pass by the property caller… so I've used a try/except

#     @property
#     def serie_column_name(self) -> str:
#         try:
#             return self.__dict__["_serie_column_name"]
#         except KeyError:
#             return None

#     @serie_column_name.setter
#     def serie_column_name(self, val) -> None:
#         self.__dict__["_serie_column_name"] = val

#     @property
#     def timestep_column_name(self) -> str:
#         try:
#             return self.__dict__["_timestep_column_name"]
#         except KeyError:
#             return None

#     @timestep_column_name.setter
#     def timestep_column_name(self, val) -> None:
#         self.__dict__["_timestep_column_name"] = val

#     @property
#     def series(self) -> np.ndarray:
#         try:
#             return self.__dict__["_series"]
#         except KeyError:
#             return None

#     @series.setter
#     def series(self, val) -> None:
#         self.__dict__["_series"] = val

#     @property
#     def timesteps(self) -> np.ndarray:
#         try:
#             return self.__dict__["_timesteps"]
#         except KeyError:
#             return None

#     @timesteps.setter
#     def timesteps(self, val) -> None:
#         self.__dict__["_timesteps"] = val

#     @property
#     def N_series(self) -> int:
#         return len(self.series)

#     @property
#     def N_timesteps(self) -> int:
#         return len(self.timesteps)

#     # end trick
#     ######

#     def _convert_2D_to_3D_array(
#         self, array_2D: np.ndarray, N_columns: int
#     ) -> np.ndarray:
#         return array_2D.reshape([self.N_series, self.N_timesteps, N_columns])

#     def _convert_3D_to_2D_array(
#         self, array_3D: np.ndarray, N_columns: int
#     ) -> np.ndarray:
#         return array_3D.reshape([self.N_series * self.N_timesteps, N_columns])

#     def get_3D_array(self, column_names: list[str]) -> np.ndarray:
#         return self._convert_2D_to_3D_array(
#             self[column_names].to_numpy(), len(column_names)
#         )

#     def set_from_3D_array(self, column_names: list[str], arr: np.ndarray) -> None:
#         self[column_names] = self._convert_3D_to_2D_array(arr, len(column_names))


def train_test_split_on_series(
    df: pd.DataFrame,
    serie_column_name: str,
    timestep_column_name: str,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    _ = _get_and_check_series_and_timesteps(df, serie_column_name, timestep_column_name)

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
    """
    Object that hopefully helps using Time Series with Scikit Learn pipelines.

    instead of being initialized (during the __init__), the dimensions of the time series are calculated
    each time we use 2D-to-3D conversions, and are used during 3D-to-2D conversions
    that way it can be used for datasets of different size,
    providing we always start with using a 2D-to-3D conversion on them
    """

    def __init__(
        self,
        serie_column_name: str,
        timestep_column_name: str,
    ):
        self.serie_column_name = serie_column_name
        self.timestep_column_name = timestep_column_name

    def df2D_to_array3D(self, df: pd.DataFrame) -> np.ndarray:
        _ = _get_and_check_series_and_timesteps(
            df, self.serie_column_name, self.timestep_column_name
        )
        x = df.sort_values(by=[self.serie_column_name, self.timestep_column_name])
        self.series_col = x[self.serie_column_name]
        self.series_unq = self.series_col.unique()
        self.timesteps_col = x[self.timestep_column_name]
        self.timesteps_unq = self.timesteps_col.unique()

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

        df[self.serie_column_name] = self.series_col
        df[self.timestep_column_name] = self.timesteps_col
        return df[
            [self.serie_column_name, self.timestep_column_name]
            + self.other_column_names
        ]
