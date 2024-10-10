# %%
from typing import Any, Type, Union
from pprint import pp

import numpy as np
import pandas as pd
from reservoirpy.nodes import Reservoir, Ridge, Input
from reservoirpy.model import Model
from reservoirpy import set_seed, verbosity
from reservoirpy.observables import nrmse, rsquare
from sklearn.base import TransformerMixin


set_seed(42)
verbosity(0)


class ModelConfiguration:
    """Helper to generate various Revservoir model configurations.

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
        verbose: bool = False,
    ):
        assert isinstance(input_to_readout, bool)
        assert isinstance(readout_feedback_to_reservoir, bool)
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
        if verbose:
            self.print_summary()

    def print_summary(self) -> None:
        pp("=MODEL SUMMARY==")
        pp(self.model)
        pp("=Hyper-parameters=")
        pp(self.model.hypers)
        pp("=Input=")
        pp(self.data)
        pp("=Reservoir=")
        pp(self.reservoir)
        pp("=Readout=")
        pp(self.readout)
        pp("=Feedback nodes=")
        pp(self.model.feedback_nodes)

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

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
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

    def run(self, X: np.ndarray) -> list[np.ndarray]:
        return self.model.run(X)


def objective(
    dataset,
    config,
    *,
    input_scaling,
    N,
    sr,
    lr,
    ridge,
    seed,
    input_to_readout,
    readout_feedback_to_reservoir,
    warmup,
) -> dict[str, float]:
    """Objective function for HP tuning in reservoirpy.

    from: https://reservoirpy.readthedocs.io/en/latest/user_guide/hyper.html#Step-1:-define-the-objective
    """

    # This step may vary depending on what you put inside 'dataset'
    x_train, y_train, x_test, y_test = dataset

    # You can access anything you put in the config
    # file from the 'config' parameter.
    instances = config["instances_per_trial"]

    # The seed should be changed across the instances,
    # to be sure there is no bias in the results
    # due to initialization.
    variable_seed = seed

    losses = []
    r2s = []
    for variable_seed in range(instances):
        # Build your model given the input parameters
        input_kwargs: dict[str, Any] = {}
        reservoir_kwargs = {
            "units": N,
            "lr": lr,
            "sr": sr,
            "input_scaling": input_scaling,
            "seed": variable_seed,
        }
        ridge_kwargs = {"ridge": ridge}
        fit_kwargs = {"warmup": warmup}

        model = ModelConfiguration(
            input_kwargs,
            reservoir_kwargs,
            ridge_kwargs,
            fit_kwargs,
            input_to_readout=input_to_readout,
            readout_feedback_to_reservoir=readout_feedback_to_reservoir,
        )
        # Train your model and test your model.
        model.fit(x_train, y_train)
        predictions = model.run(x_test)
        loss = nrmse(y_test, predictions)
        r2 = rsquare(y_test, predictions)

        # Change the seed between instances
        variable_seed += 1

        losses.append(loss)
        r2s.append(r2)

    # Return a dictionnary of metrics. The 'loss' key is mandatory when
    # using hyperopt.
    return {"loss": np.mean(losses), "r2": np.mean(r2s)}


data_2D = Union[np.ndarray, pd.DataFrame]


class ScalingData:

    def __init__(
        self,
        x_train: data_2D,
        y_train: data_2D,
        x_test: data_2D,
        y_test: data_2D,
        x_scaler: TransformerMixin,
        y_scaler: TransformerMixin,
    ):

        self.x_scaler = x_scaler
        self.x_train = self.x_scaler.fit_transform(x_train.to_numpy())
        self.x_test = self.x_scaler.transform(x_test.to_numpy())

        self.y_scaler = y_scaler
        self.y_train = self.y_scaler.fit_transform(y_train.to_numpy())
        self.y_test = self.y_scaler.transform(y_test.to_numpy())


if __name__ == "__main__":
    # %%
    input_kwargs = {"input_dim": 8}
    reservoir_kwargs = {"units": 100, "lr": 1.0, "sr": 1.0}
    ridge_kwargs = {"ridge": 1e-6, "output_dim": 1}
    fit_kwargs = {"warmup": 2}

    modconf = ModelConfiguration(
        input_to_readout=True,
        readout_feedback_to_reservoir=True,
        input_kwargs=input_kwargs,
        reservoir_kwargs=reservoir_kwargs,
        ridge_kwargs=ridge_kwargs,
        fit_kwargs=fit_kwargs,
    )

    X_train = np.random.random((2, 50, 8))
    Y_train = np.random.random((2, 50, 1))

    # modconf.fit(X_train, Y_train)

    # %%

    data1 = Input(**input_kwargs)
    reservoir1 = Reservoir(**reservoir_kwargs)
    readout1 = Ridge(**ridge_kwargs)

    readout1 <<= reservoir1

    model1 = data1 >> reservoir1 >> readout1

    data2 = Input(**input_kwargs)
    reservoir2 = Reservoir(**reservoir_kwargs)
    readout2 = Ridge(**ridge_kwargs)

    readout2 <<= data2

    model2 = data2 >> reservoir2 >> readout2

    # %%
