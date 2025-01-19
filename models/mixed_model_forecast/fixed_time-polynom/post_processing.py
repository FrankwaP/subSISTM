# -*- coding: utf-8 -*-
from sys import path

path.append("../")

from utils.post_processing import process
import study_config


process(study_config)
