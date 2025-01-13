# -*- coding: utf-8 -*-
from sys import path

path.append("../")

from utils.hp_optimization import run_optimization
import study_config


run_optimization(study_config)
