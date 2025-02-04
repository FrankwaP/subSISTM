# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from os import chdir
from subprocess import check_call
from copy import deepcopy
from typing import Optional, Callable

from joblib import dump  # type: ignore
from numpy import array, inf
from numpy.typing import NDArray
from pandas import DataFrame, read_csv
from reservoirpy.nodes import ESN  # type: ignore
from sklearn.metrics import mean_squared_error as mse  # type: ignore
from optuna import Trial, TrialPruned


def add_path(p: str) -> None:
    pth = Path(p).resolve().as_posix()
    if pth not in sys.path:
        print("Added to sys.path:", pth)
        sys.path.append(pth)


add_path(Path(__file__).parent.as_posix() + "../../")

from reservoirs_synthetic_bph.utils.global_config import SERIES, TSTEPS


# !!! please set similar values in the R scripts
LOCAL_DIR = Path(__file__).parent.as_posix()
R_FIT = LOCAL_DIR + "/random_effects_fit.R"
R_PREDICT = LOCAL_DIR + "/random_effects_predict.R"


# RSLT_DIR = "results"
CVG_LOG_CSV = "convergence-log.csv"

PY_JBLB_BEST = "best_fixed_effects.joblib"
R_RDS = "random_hlme.Rds"
R_RDS_BEST = "best_random_hlme.Rds"

PY_CSV_FIT_RESID = "fixed_effect_fit_residuals.csv"
PY_CSV_PRED = "dataframe_predict.csv"

R_CSV_FIT = "random_effect_fit.csv"
R_CSV_PRED = "random_effect_predict.csv"


###
X_LABELS = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
Y_LABEL = "y_mixed_obs"


Y_LABEL_FE = Y_LABEL + "__fixed"

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

    def fit(self, X: NDArray, y: NDArray, **options) -> None:
        self.model.fit(X, y, **options)

    def predict(self, X: NDArray) -> NDArray:
        return self._model_predict(X)


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

    def fit(self, X: NDArray, y: NDArray, **options) -> None:
        super().fit(self.trans_input(X), self.trans_input(y), **options)

    def predict(self, X: NDArray) -> NDArray:
        return self.trans_y_output(super().predict(self.trans_input(X)))


class _RandomEffectsEstimator:
    """Call R scripts and read their output files"""

    def fit_and_predict(self) -> NDArray:
        # !!! prediction is done on R_CSV_FIT (train)
        return self._call_and_read(R_FIT, R_CSV_FIT)

    def predict(self) -> NDArray:
        # !!! prediction is done on R_CSV_PRED (val)
        return self._call_and_read(R_PREDICT, R_CSV_PRED)

    @staticmethod
    def _call_and_read(r_file, r_csv_file: str) -> NDArray:
        check_call(["Rscript", r_file])
        df = read_csv(r_csv_file)
        assert len(df.columns) == 1
        return df.iloc[:, 0].to_numpy()


class MixedMLEstimator:

    def __init__(
        self,
        fixed_effect_estimator,
        # random_effects_estimator: RandomEffectsEstimator,
        *,
        recurrent_model: bool,
        specific_dir: str,
    ):
        assert isinstance(recurrent_model, bool)
        if not recurrent_model:  # in this order for mypy
            model_class = _FixedEffectsEstimator
        else:
            model_class = _RecurrentFixedEffectsEstimator

        self.ml_fixed = model_class(fixed_effect_estimator)
        # self.lcmm = random_effects_estimator
        self.lcmm = _RandomEffectsEstimator()
        self.specific_dir = Path(specific_dir).resolve().as_posix()
        Path(self.specific_dir).mkdir(exist_ok=True)

    def _fit_iteration(
        self, df: DataFrame, fixed_model_option: dict
    ) -> tuple[DataFrame, float]:
        #### fitting the Machine Learning model
        X = df[X_LABELS].to_numpy()
        y = df[Y_LABEL_FE].to_numpy()
        # we train ml_fixed by ignoring cluster effects (with the target y)
        # to get an estimate of y_fixed
        if isinstance(self.ml_fixed, _RecurrentFixedEffectsEstimator):
            self.ml_fixed.prepare(df)
        print("\tFitting fixed effects with ML model.")
        self.ml_fixed.fit(X, y, **fixed_model_option)
        y_fixed = self.ml_fixed.predict(X)

        #### fitting the Random Effect model
        # based on e_fixed
        # !!! l'erreur était là (j'utilisais "y - y_fixed")
        df[PY_E_FIXED_COLUMN] = df[Y_LABEL] - y_fixed
        df.to_csv(PY_CSV_FIT_RESID, index=False)

        # we estimate u
        print("\tFitting random effects with mixed model.")
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
        n_iter_improv: int,
        min_rltv_imrov: float,
        fixed_model_fit_options: dict,
        trial_for_pruning: Optional[Trial] = None,
    ):
        wd = Path(".").resolve()
        chdir(self.specific_dir)
        self._fit(
            df_data,
            n_iter_improv,
            min_rltv_imrov,
            fixed_model_fit_options,
            trial_for_pruning,
        )
        chdir(wd)

    def _fit(
        self,
        df_data: DataFrame,
        n_iter_improv: int,
        min_rltv_imrov: float,
        fixed_model_fit_options: dict,
        trial_for_pruning: Optional[Trial] = None,
    ):
        #
        istep = 0
        df = df_data.copy()  # we are going to modify it for fitting
        df[Y_LABEL_FE] = df[Y_LABEL]
        # iteration
        best_metric = inf
        n_it_no_improv = 0
        log_dict_list = []
        while True:
            print(f"mixedML step #{istep:02d}")
            df, metric = self._fit_iteration(df, fixed_model_fit_options)
            log_dict = {"step": istep, "metric": metric}
            if trial_for_pruning:
                log_dict["trial"] = trial_for_pruning.number
            log_dict_list.append(log_dict)
            print(f"\tresulting MSE: {metric:8e}", end="")
            #
            test_best_metric = best_metric
            if metric < test_best_metric:
                best_metric = metric
                best_ml_fixed = deepcopy(self.ml_fixed)
                Path(R_RDS).rename(R_RDS_BEST)
            if metric < (1 - min_rltv_imrov) * test_best_metric:
                print("")
                n_it_no_improv = 0
            else:
                print(" (no improvement)")
                n_it_no_improv += 1
                if n_it_no_improv > n_iter_improv:
                    break
            #
            if trial_for_pruning:
                trial_for_pruning.report(metric, istep)
                # Handle pruning based on the intermediate value.
                if trial_for_pruning.should_prune():
                    raise TrialPruned()
            istep += 1
        #
        self.ml_fixed = best_ml_fixed
        dump(best_ml_fixed, PY_JBLB_BEST)
        Path(R_RDS).unlink(missing_ok=True)
        DataFrame(log_dict_list).to_csv(CVG_LOG_CSV, index=False)

    def predict(
        self, df_data: DataFrame, *, use_subject_specific: bool
    ) -> NDArray:
        wd = Path(".").resolve()
        chdir(self.specific_dir)
        result = self._predict(df_data, use_subject_specific)
        chdir(wd)
        return result

    def _predict(
        self, df_data: DataFrame, use_subject_specific: bool
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
            df.to_csv(PY_CSV_PRED, index=False)
            y_random = self.lcmm.predict()
            return y_fixed + y_random
        else:
            return y_fixed

    # aliases for compatibility
    run = predict
