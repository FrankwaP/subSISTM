# -*- coding: utf-8 -*-
from typing import Literal

from reservoirpy import set_seed, verbosity
from sklearn.preprocessing import RobustScaler  # type: ignore

SERIES_COLUMN_NAME = "individus"
TIMESTEPS_COLUMN_NAME = "temps"
SCALER = RobustScaler
# reservoirpy
N_WARMUPS = 5
N_SEEDS = 5
N_CPUS = 5  # negative number => CPU_USED = MAX_CPU + N_CPUS
JOBLIB_BACKEND: Literal["threading", "multiprocessing", "loky"] = "loky"
set_seed(0)
verbosity(1)
