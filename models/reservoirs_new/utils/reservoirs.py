#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from reservoirpy.nodes import ESN, Reservoir, Ridge  # type: ignore
from reservoirpy import verbosity  # type: ignore

from .global_config import N_CPUS, N_SEEDS, JOBLIB_BACKEND
from .data import FLOAT_DTYPE

verbosity(level=1)


def get_esn_model_list(
    reservoir_kwargs: dict, ridge_kwargs: dict, ens_kwargs: dict
) -> list[ESN]:

    return [
        ESN(
            reservoir=Reservoir(
                **reservoir_kwargs, seed=reservoir_seed, dtype=FLOAT_DTYPE
            ),
            readout=Ridge(**ridge_kwargs),
            **ens_kwargs,
            workers=N_CPUS,
            backend=JOBLIB_BACKEND,
        )
        for reservoir_seed in range(42, 42 + N_SEEDS)
    ]
