# -*- coding: utf-8 -*-
from pathlib import Path

from reservoirpy import set_seed, verbosity  # type: ignore
from sklearn.preprocessing import RobustScaler  # type: ignore

DATA_DIR = (
    (Path(__file__).parent / Path("../../../data/synthetic_bph_1/"))
    .resolve()
    .as_posix()
)
TEST_FILE = "01_test.csv"
SIMU_PATTERN = "simulation*.csv"
###########
# they are now defined in the mixedml module
# becvause they must correspond to the Rscript (so it's a better place to set them)
# SERIES_COLUMN_NAME = "individus"
# TIMESTEPS_COLUMN_NAME = "temps"
###########
SCALER = RobustScaler
# reservoirpy
N_WARMUPS = 5
N_SEEDS = 5
set_seed(0)
verbosity(0)
# HP optimization
HP_JSON_FILE = "HP_medians.json"

# MAE/MSE
PRED_CSV_FILE = "predictions.csv"
PRED_CSV_FILE_TRAIN = "predictions_train.csv"
