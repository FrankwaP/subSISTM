# -*- coding: utf-8 -*-
from typing import Literal
from pathlib import Path

from reservoirpy import set_seed, verbosity  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore

DATA_DIR = Path("../../../data/synthetic_bph_1/").resolve().as_posix()
print("Data folder:", DATA_DIR)
TEST_FILE = "01_test.csv"
SIMU_PATTERN = "simulation*.csv"
SERIES = "individus"
TSTEPS = "temps"
SORT_COLUMNS = [SERIES, TSTEPS]
SCALER = RobustScaler


# reservoirpy
FLOAT_DTYPE = "float32"
N_WARMUPS = 5
N_SEEDS = 5
N_CPUS = N_SEEDS  # negative number => CPU_USED = MAX_CPU + N_CPUS
JOBLIB_BACKEND: Literal["threading", "multiprocessing", "loky"] = "loky"
set_seed(0)
verbosity(0)
# HP optimization
HP_JSON_FILE = "HP_medians.json"

# MAE/MSE
PRED_CSV_FILE = "predictions.csv"
PRED_CSV_FILE_TRAIN = "predictions_train.csv"
