# -*- coding: utf-8 -*-
from sys import path

path.append("../")

from utils.errors_computing import computer_errors
import study_config


computer_errors(study_config)
