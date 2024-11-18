#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

values for training:
    xi without noise: "x1", …
    y with noise: "y_fixed_obs" or "y_mixed_obs"
values for metrics:
    y without noise: "y_fixed" or "y_mixed"



HP optimisation step #1:
    train = simulation1
    val = simulation2
    
HP optimisation step #2:
    train = simulation2
    val = simulation1
    
Compare the HPs from both optimisation steps.


Training/prediction steps:
    test = 01_test.csv
    
    for train in simulation*.csv:
        fit on train
        
        prediction on train
        MAE/MSE on train
        
        prediction on test
        MAE/MSE on test

"""
from sklearn.base import TransformerMixin  # type: ignore

from pandas import DataFrame
from numpy.typing import NDArray
from optuna import Trial


class TimeSerieShapeChanger(TransformerMixin):
    """
    Object that hopefully helps using Time Series with Scikit Learn pipelines.

    Instead of being initialized (during the __init__), the dimensions of the time
    series are calculated each time we use the 2D-to-3D conversion.
    Then the same dimensions are used for the following 3D-to-2D conversions.

    That way it can be used for datasets of different size (like train and test dataset)
    providing we always do a 2D-to-3D conversion first then immediately the 3D-to-2D
    conversion on the same dataset.
    """

    def __init__(
        self,
        series_column_name: str,
        timesteps_column_name: str,
    ):
        self.series_column_name = series_column_name
        self.timesteps_column_name = timesteps_column_name

    def fit(self, X: DataFrame) -> None:
        sort_cols = [self.series_column_name, self.timesteps_column_name]
        X.sort_values(by=sort_cols, inplace=True)

        self._series_col = X[self.series_column_name]
        self.series = self._series_col.unique().tolist()
        self.N_s = len(self.series)

        self._timesteps_col = X[self.timesteps_column_name]
        self.timesteps = self._timesteps_col.unique().tolist()
        self.N_t = len(self.timesteps)

        self.other_column_names = [c for c in X.columns if c not in sort_cols]
        self.N_x = len(self.other_column_names)
        return self

    def transform(self, X: DataFrame) -> NDArray:
        return X[self.other_column_names].to_numpy().reshape([self.N_s, self.N_t, self.N_x])

    def inverse_transform(self, X: NDArray) -> DataFrame:
        assert X.shape == (self.N_s, self.N_t, self.N_x)  # just in case…

        df = DataFrame(
            data=X.reshape([self.N_s * self.N_t, self.N_x]),
            columns=self.other_column_names,
        )
        df.insert(0, self.series_column_name, self._series_col)
        df.insert(1, self.timesteps_column_name, self._timesteps_col)
        return df


class Data:

    def __init__(
        self,
        dataframe: DataFrame,
        serie_column_name: str,
        timestep_column_name: str,
        x_labels: list[str],
        y_labels: list[str],
    ):

        sort_columns = [serie_column_name, timestep_column_name]

        self.dataframe = dataframe.sort_values(by=sort_columns)
        self.serie_column_name = serie_column_name
        self.timestep_column_name = timestep_column_name

        self.x_labels = x_labels
        self.N_x = len(x_labels)
        self.y_labels = y_labels
        self.N_y = len(y_labels)

        self.series = dataframe[serie_column_name].unique().list()
        self.N_s = len(self.series)

        self.timesteps = dataframe[timestep_column_name].unique().list()
        self.N_t = len(self.timesteps)

    def fit_transform(self, scaler: TransformerMixin):
        scaled_x = scaler.fit_transform(self.dataframe[self.x_labels])
        self._x_3D_scaled = scaled_x.reshape([self.N_s, self.N_t, self.N_x])
        # we only need to unscale the Y value (prediction) so we can just reuse the scaler
        scaled_y = scaler.fit_transform(self.dataframe[self.y_labels])
        self._y_3D_scaled = scaled_y.reshape([self.N_s, self.N_t, self.N_y])
        self.scaler = scaler

    def transform(self, scaler: TransformerMixin):
        scaled_x = scaler.transform(self.dataframe[self.x_labels])
        self._x_3D_scaled = scaled_x.reshape([self.N_s, self.N_t, self.N_x])
        # we only need to unscale the Y value (prediction) so we can just reuse the scaler
        scaled_y = scaler.transform(self.dataframe[self.y_labels])
        self._y_3D_scaled = scaled_y.reshape([self.N_s, self.N_t, self.N_y])
        self.scaler = scaler

    def x_3D_scaled(self) -> NDArray:
        return self._x_3D_scaled

    def y_3D_scaled(self) -> NDArray:
        return self._y_3D_scaled

    def to_2D_unscaled(self, *, ypred_3D_scaled) -> DataFrame:
        ypred_2D_scaled = ypred_3D_scaled.reshape([self.N_s, self.N_t, self.N_y])
        ypred_2D_unscaled = self.scaler.inverse_transform(ypred_2D_scaled)
        return DataFrame(ypred_2D_unscaled, columns=self.y_labels)


# def hp_optimization(csv_file1: str, csv_file2: str) -> Trial:

#     data = pd.read_csv(csv_file1, sep=";", decimal=",")
#     data = data.sort_values(by=["individus", "temps"])


#     def get_


#     def fit_transform(self)
