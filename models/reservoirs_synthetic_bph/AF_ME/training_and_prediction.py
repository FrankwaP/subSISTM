# -*- coding: utf-8 -*-
from sys import path

path.append("../")

from utils.training_prediction import train_pred_loop
import study_config


train_pred_loop(study_config)
