# -*- coding: utf-8 -*-
from pathlib import Path
from subprocess import check_call
from copy import deepcopy

from joblib import dump, load  # type: ignore
from numpy import array
from numpy.typing import NDArray
from pandas import DataFrame, read_csv
from reservoirpy.nodes import ESN
from sklearn.metrics import mean_squared_error as mse  # type: ignore

# this is to avoid doing it in other scripts/notebooks
from sys import path

path.append("../reservoirs-synthetic_bph/")
from utils.data import get_dataframe


# !!! please set similar values in the R scripts
LOCAL_DIR = Path(__file__).parent.as_posix()


R_FIT = LOCAL_DIR + "/random_effects_fit.R"
R_PREDICT = LOCAL_DIR + "/random_effects_predict.R"


RSLT_DIR = "results"
Path(RSLT_DIR).mkdir(exist_ok=True)

PY_JBLB_BEST = RSLT_DIR + "/best_fixed_effects.joblib"
R_RDS = RSLT_DIR + "/random_hlme.Rds"
R_RDS_BEST = RSLT_DIR + "/best_random_hlme.Rds"

PY_CSV_FIT_RESID = RSLT_DIR + "/fixed_effect_fit_residuals.csv"
PY_CSV_PRED = RSLT_DIR + "/dataframe_predict.csv"

R_CSV_FIT = RSLT_DIR + "/random_effect_fit.csv"
R_CSV_PRED = RSLT_DIR + "/random_effect_predict.csv"


###

SERIES = "individus"
TSTEPS = "temps"
X_LABELS = ["x2_x5", "x4_x7", "x6_x8"]
Y_LABEL = "y_mixed_obs"  # the values are fixed

# the paper says "y_fixed" for "y fixed effects"
# but it can be understood as "y not changing"
# so I call it "y_fe"

Y_LABEL_FE = Y_LABEL + "__fixed"  # the value are not fixed

PY_E_FIXED_COLUMN = "e_fixed"
R_PRED_SS_COLUMN = "pred_ss"


class _FixedEffectsEstimator:
    """Class to handle classical ML models."""

    # reservoirpy uses "run" for the prediction
    PREDICT_NAMES = ["predict", "run"]

    def __init__(self, model):
        self.model = model
        #
        meth_name = [m for m in self.PREDICT_NAMES if hasattr(self.model, m)]
        assert len(meth_name) == 1
        self._model_predict = getattr(self.model, meth_name[0])

    def fit(self, X: NDArray, y: NDArray) -> None:
        self.model.fit(X, y)

    def predict(self, X: NDArray) -> NDArray:
        return self._model_predict(X)

    def fit_and_predict(self, X: NDArray, y: NDArray) -> NDArray:
        self.fit(X, y)
        return self.predict(X)


class _RecurrentFixedEffectsEstimator(_FixedEffectsEstimator):
    """Class to handle recurrent ML models."""

    def __init__(self, model):
        if isinstance(model, ESN):
            raise UserWarning("Nope! ESN are boring to save/load: build one.")
        super().__init__(model)
        self.n_series, self.n_tsteps = None, None

    def prepare(self, df: DataFrame):
        df.sort_values(by=[SERIES, TSTEPS], inplace=True)
        self.n_series = len(df[SERIES].unique())
        self.n_tsteps = len(df[TSTEPS].unique())

    def trans_input(self, X_or_y: NDArray) -> NDArray:
        if self.n_series is None:
            raise UserWarning(
                f"Use the '{self.prepare.__name__}' method first"
            )
        return X_or_y.reshape((self.n_series, self.n_tsteps, -1))

    def trans_y_output(self, y: NDArray) -> NDArray:
        if isinstance(y, list):
            y = array(y)
        y = y.reshape((self.n_series * self.n_tsteps,))
        self.n_series, self.n_tsteps = None, None
        return y

    def fit(self, X: NDArray, y: NDArray) -> None:
        super().fit(self.trans_input(X), self.trans_input(y))

    def predict(self, X: NDArray) -> NDArray:
        return self.trans_y_output(super().predict(self.trans_input(X)))


class _RandomEffectsEstimator:

    def fit_and_predict(self) -> NDArray:
        return self._call_and_read(R_FIT, R_CSV_FIT)

    def predict(self) -> NDArray:
        return self._call_and_read(R_PREDICT, R_CSV_PRED)

    @staticmethod
    def _call_and_read(r_file, r_csv_file: str) -> NDArray:
        check_call(["Rscript", r_file])
        df = read_csv(r_csv_file)
        return df[R_PRED_SS_COLUMN].to_numpy()


class MixedMLEstimator:

    def __init__(
        self,
        fixed_effect_estimator,
        # random_effects_estimator: RandomEffectsEstimator,
        *,
        recurrent_model: bool,
    ):
        assert isinstance(recurrent_model, bool)
        if recurrent_model:
            model_class = _RecurrentFixedEffectsEstimator
        else:
            model_class = _FixedEffectsEstimator

        self.ml_fixed = model_class(fixed_effect_estimator)
        # self.lcmm = random_effects_estimator
        self.lcmm = _RandomEffectsEstimator()

    def fit_iteration(self, df: DataFrame) -> tuple[DataFrame, float]:
        #### fitting the Machine Learning model
        X = df[X_LABELS].to_numpy()
        y = df[Y_LABEL_FE].to_numpy()
        # we train ml_fixed by ignoring cluster effects (with the target y)
        # to get an estimate of y_fixed
        if isinstance(self.ml_fixed, _RecurrentFixedEffectsEstimator):
            self.ml_fixed.prepare(df)
        y_fixed = self.ml_fixed.fit_and_predict(X, y)

        #### fitting the Random Effect model
        # based on e_fixed
        # !!! l'erreur était là (j'utilisais "y - y_fixed")
        df[PY_E_FIXED_COLUMN] = df[Y_LABEL] - y_fixed
        df.to_csv(PY_CSV_FIT_RESID, index=False)

        # we estimate u
        y_random = self.lcmm.fit_and_predict()

        # then we upgrade y_fixed = y-Zu
        # … and re-train ml_fixed with the updated target variable y_fixed
        df[Y_LABEL_FE] = df[Y_LABEL] - y_random

        ####
        # final prediction to monitor convergence
        # we have checked that it's equivalent to using directly the "resid_ss" value
        # but the form "mse(df[Y_LABEL], y_pred)" has been prefered to "mean(resid_ss**2)
        y_pred = y_fixed + y_random
        return df, mse(df[Y_LABEL], y_pred)

    def fit(
        self,
        df_data: DataFrame,
        *,
        n_iter_improve: int,
        max_iter: int,
    ) -> list[float]:
        # initialization
        istep = 0
        df = df_data.copy()  # we are going to modify it for fitting
        df[Y_LABEL_FE] = df[Y_LABEL]
        # iteration
        metric_list = []
        best_metric = None
        for istep in range(max_iter):
            df, metric = self.fit_iteration(df)
            print(f"mixedML step #{istep:02d}: {metric:8e}", end="")
            #
            if best_metric is None or metric < best_metric:
                print(" (best)")
                best_metric = metric
                n_it_no_improve = 0
                best_ml_fixed = deepcopy(self.ml_fixed)
                Path(R_RDS).rename(R_RDS_BEST)
            else:
                print("")
                n_it_no_improve += 1
                if n_it_no_improve > n_iter_improve:
                    break
            metric_list.append(metric)
            istep += 1
        #
        self.ml_fixed = best_ml_fixed
        dump(best_ml_fixed, PY_JBLB_BEST)
        Path(R_RDS).unlink(missing_ok=True)
        return metric_list

    def predict(
        self, df_data: DataFrame, *, use_subject_specific: bool
    ) -> NDArray:
        assert isinstance(use_subject_specific, bool)
        df = df_data.copy()  # we are going to modify it for predicting
        if isinstance(self.ml_fixed, _RecurrentFixedEffectsEstimator):
            self.ml_fixed.prepare(df)
        X = df[X_LABELS].to_numpy()
        y_fixed = self.ml_fixed.predict(X)
        #
        if use_subject_specific:
            df[PY_E_FIXED_COLUMN] = df[Y_LABEL] - y_fixed
        else:
            df[PY_E_FIXED_COLUMN] = 0
        df.to_csv(PY_CSV_PRED, index=False)
        y_random = self.lcmm.predict()
        #
        return y_fixed + y_random

    # aliases for compatibility
    run = predict
