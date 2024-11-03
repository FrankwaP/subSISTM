from typing import Union
from multiprocessing import Pool
from itertools import repeat

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from reservoirpy.nodes import Reservoir, Ridge, Input  # type: ignore
from reservoirpy.model import Model  # type: ignore
from reservoirpy import set_seed, verbosity  # type: ignore
from sklearn.base import TransformerMixin  # type: ignore

from optuna.trial import FrozenTrial, Trial

set_seed(42)
verbosity(0)


AnyTrial = Union[Trial, FrozenTrial]


def remove_warmup(array_3D: NDArray, n_warmups: int) -> NDArray:
    return array_3D[:, n_warmups:, :]


class ProcessedData:

    def __init__(
        self,
        data_train: DataFrame,
        data_test: DataFrame,
        series_column_name: str,
        timestep_column_name: str,
        x_labels: list[str],
        y_labels_train: list[str],
        y_labels_test: list[str],
        scaler: TransformerMixin,
    ):
        # N: number of series/individuals
        # T: number of timesteps
        # P: number of features
        N_train, T_train = self._get_data_shape(
            data_train, series_column_name, timestep_column_name
        )
        N_test, T_test = self._get_data_shape(
            data_test, series_column_name, timestep_column_name
        )
        assert len(y_labels_train) == len(y_labels_test)
        P_x, P_y = len(x_labels), len(y_labels_train)

        x_train = data_train[x_labels]
        x_test = data_test[x_labels]
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        self.x_train_3D = x_train_scaled.reshape([N_train, T_train, P_x])
        self.x_test_3D = x_test_scaled.reshape([N_test, T_test, P_x])

        y_train = data_train[y_labels_train]
        y_test = data_test[y_labels_test]
        y_train_scaled = scaler.fit_transform(y_train.values)
        y_test_scaled = scaler.transform(y_test.values)
        self.y_train_3D = y_train_scaled.reshape([N_train, T_train, P_y])
        self.y_test_3D = y_test_scaled.reshape([N_test, T_test, P_y])

        self.scaler = scaler
        self.N_test = N_test
        self.T_test = T_test
        self.P_y = P_y
        self.data_test = data_test
        self.x_labels = x_labels
        self.y_labels_train = y_labels_train
        self.y_labels_test = y_labels_test
        self.series_column_name = series_column_name
        self.timestep_column_name = timestep_column_name

        print(f"Train and test data prepared with scaler {type(scaler)}.")

    def to_unscaled_test_2D(self, pred_array_3D: NDArray) -> NDArray:
        return self.scaler.inverse_transform(
            pred_array_3D.reshape([self.N_test * self.T_test, self.P_y])
        )

    @staticmethod
    def _get_data_shape(
        data: DataFrame, series_column_name: str, timesteps_column_name: str
    ) -> tuple[int, int]:
        if not data.equals(
            data.sort_values([series_column_name, timesteps_column_name])
        ):
            raise UserWarning("Dataframe is not sorted by [series, timesteps].")
        N_series = len(data[series_column_name].unique())
        N_timesteps = len(data[timesteps_column_name].unique())
        if not len(data) == N_series * N_timesteps:
            raise UserWarning("Dataframe is not in N_series * N_timesteps format.")
        return N_series, N_timesteps


class ModelConfiguration:
    """Helper to generate various Reservoir model configurations.

    Please see: https://reservoirpy.readthedocs.io/en/latest/user_guide/advanced_demo.html
    (quotes in comments/docstrings will refer to this pages)
    """

    def __init__(
        self,
        input_kwargs: dict,
        reservoir_kwargs: dict,
        ridge_kwargs: dict,
        fit_kwargs: dict,
        *,
        input_to_readout: bool,
        readout_feedback_to_reservoir: bool,
    ):
        assert isinstance(input_to_readout, bool)
        assert isinstance(readout_feedback_to_reservoir, bool)
        #
        self.input_to_readout = input_to_readout
        self.readout_feedback_to_reservoir = readout_feedback_to_reservoir
        self.fit_kwargs = fit_kwargs
        #
        self.data = Input(**input_kwargs)
        self.reservoir = Reservoir(**reservoir_kwargs)
        self.readout = Ridge(**ridge_kwargs)
        self.model = self._get_model()

    def set_reservoir_seed(self):
        self.reservoir.se

    def _get_model(self) -> Model:
        if self.readout_feedback_to_reservoir:
            self._set_readout_feedback()

        if self.input_to_readout:
            return self._get_input_in_readout_model()

        return self._get_simple_model()

    def _get_simple_model(self) -> Model:
        return self.data >> self.reservoir >> self.readout

    def _get_input_in_readout_model(self) -> Model:
        # https://reservoirpy.readthedocs.io/en/latest/user_guide/advanced_demo.html#Input-to-readout-connections
        return [self.data, self.data >> self.reservoir] >> self.readout

    def _set_readout_feedback(self) -> None:
        # https://reservoirpy.readthedocs.io/en/latest/user_guide/advanced_demo.html#Feedback-connections
        self.reservoir <<= self.readout

    def fit(self, X_train: NDArray, Y_train: NDArray) -> None:
        assert "force_teachers" not in self.fit_kwargs
        self.model.fit(
            X=X_train,
            Y=Y_train,
            # force_teacher is set to True by default,
            # and setting it to False makes the fit method "crash"
            #   (yeah sorry it is not more accurate so far)
            # force_teachers=self.readout_feedback,
            **self.fit_kwargs,
        )

    def run(self, X: NDArray) -> list[NDArray]:
        return self.model.run(X)


def get_3D_prediction(
    model: ModelConfiguration, processed_data: ProcessedData
) -> NDArray:
    # !!! 3D prediction are scaled
    model.fit(processed_data.x_train_3D, processed_data.y_train_3D)
    y_hat_run = model.run(processed_data.x_test_3D)
    if isinstance(y_hat_run, np.ndarray):
        y_hat_run = [y_hat_run]
    return np.array(y_hat_run)


def get_3D_prediction_list(
    model_list: list[ModelConfiguration], processed_data: ProcessedData, n_cpus: int
) -> list[NDArray]:
    if n_cpus == 1:
        return [get_3D_prediction(model, processed_data) for model in model_list]
    with Pool(n_cpus) as p:
        return p.starmap(get_3D_prediction, zip(model_list, repeat(processed_data)))
