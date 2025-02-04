# -*- coding: utf-8 -*-
from typing import Literal
from pathlib import Path

from reservoirpy import set_seed, verbosity  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore

DATA_DIR = (
    (Path(__file__).parent / Path("../../../data/synthetic_bph_1/"))
    .resolve()
    .as_posix()
)
assert Path(DATA_DIR).exists()
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

# results files
PRED_CSV_FILE = "predictions.csv"
METRIC_CSV_FILE = "metrics.csv"
