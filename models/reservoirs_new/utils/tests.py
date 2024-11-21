import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from optuna_training import Data

SERIE_NAME = "person"
TIMESTEP_NAME = "time"

DF = pd.DataFrame.from_dict(
    {
        SERIE_NAME: [0, 0, 1, 1, 2, 2],
        TIMESTEP_NAME: [0, 1, 0, 1, 0, 1],
        "x1": [1, 2, 3, 4, 5, 6],
        "x2": [7, 8, 9, 10, 11, 12],
        "x3": [13, 14, 15, 16, 17, 18],
        "y": [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
    }
)

# 	person	time	x1	x2	x3	y
# 0	0	0	1	7	13	20
# 1	0	1	2	8	14	21
# 2	1	0	3	9	15	22
# 3	1	1	4	10	16	23
# 4	2	0	5	11	17	24
# 5	2	1	6	12	18	25


OTHER_NAMES = [c for c in DF.columns if c not in [SERIE_NAME, TIMESTEP_NAME]]


X_LABELS = ["x1", "x2", "x3"]
Y_LABELS = ["y"]


def test_data() -> None:

    data = Data(DF, SERIE_NAME, TIMESTEP_NAME, X_LABELS, Y_LABELS)
    data.apply_fit_transform(StandardScaler())

    y_fakepred = data.y_3D_scaled
    y_test = data.to_2D_unscaled(ypred_3D_scaled=y_fakepred)

    assert y_test.equals(DF[Y_LABELS])


if __name__ == "__main__":
    test_data()
