# -*- coding: utf-8 -*-
from sys import path

path.append("../")

from utils.training_prediction import (
    train_pred_loop,
    train_pred_loop_omg_so_ugly,
)
import study_config


# train_pred_loop(study_config)
train_pred_loop_omg_so_ugly(study_config)
