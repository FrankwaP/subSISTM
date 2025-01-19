# -*- coding: utf-8 -*-
import sys
from pathlib import Path


def add_path(p: str) -> None:
    pth = Path(p).resolve().as_posix()
    if pth not in sys.path:
        print("Added to sys.path:", pth)
        sys.path.append(pth)


add_path("../../")


from mixedML.training_utils.hp_optimization import run_optimization


run_optimization(1)
# run_optimization(2)
