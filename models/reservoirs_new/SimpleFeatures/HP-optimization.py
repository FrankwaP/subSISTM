# -*- coding: utf-8 -*-
from sys import path

path.append("../")

from utils.hp_optimization import run_optimization
import config


run_optimization(config)
