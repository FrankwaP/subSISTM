# %%
import numpy as np
from sklearn import metrics, preprocessing
from reservoirpy.nodes import Reservoir, Ridge, Input, ESN
from reservoirpy.model import Model


from reservoirpy import set_seed, verbosity


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
        print("\n==MODEL SUMMARY==")
        print(self.model)
        print("=Hyper-parameters=")
        print(self.model.hypers)
        print("=Input=")
        print(self.data)
        print("=Reservoir=")
        print(self.reservoir)
        print("=Readout=")
        print(self.readout)
        print("=Feedback nodes=")
        print(self.model.feedback_nodes)

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

    def _set_readout_feedback(self):
        # https://reservoirpy.readthedocs.io/en/latest/user_guide/advanced_demo.html#Feedback-connections
        self.reservoir <<= self.readout

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray) -> None:
        assert "force_teachers" not in self.fit_kwargs
        self.model.fit(
            X=X_train,
            Y=Y_train,
            # force_teacher is set to True by default,
            # and setting it to False makes the fit method "crash" (yeah sorry it is not more accurate so far)
            # force_teachers=self.readout_feedback,
            **self.fit_kwargs,
        )

    def run(self, X: np.ndarray) -> list[np.ndarray]:
        return self.model.run(X)


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
