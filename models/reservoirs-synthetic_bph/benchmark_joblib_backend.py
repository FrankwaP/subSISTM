#!/usr/bin/env python
# coding: utf-8

# In[41]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sys import path
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from reservoirpy.nodes import ESN, Reservoir, Ridge  # type
from sklearn.preprocessing import RobustScaler

from utils.data import Data


# # Data

# In[42]:


SERIES_COLUMN_NAME = "individus"
TIMESTEPS_COLUMN_NAME = "temps"
X_LABELS = ["x2_x5", "x4_x7"]
Y_LABELS = ["y_fixed_obs"]


# In[43]:


file_01 = "../../data/synthetic_bph_1/simulation1.csv"


train_data = Data(
    file_01,
    SERIES_COLUMN_NAME,
    TIMESTEPS_COLUMN_NAME,
    X_LABELS,
    Y_LABELS,
)

train_data.apply_fit_transform(RobustScaler(), RobustScaler())


# # Benchmark

# In[47]:


from itertools import repeat
from multiprocessing import Pool


# In[74]:


# In[69]:


def _training(model, train_data: Data):
    print(model)
    model.fit(train_data.x_3D_scaled, train_data.y_3D_scaled)
    return model


def simple_training(model_list, train_data: Data):
    trained_models = [_training(model, train_data) for model in model_list]
    return trained_models


def pool_training(model_list, train_data: Data, n_cpus: int):
    with Pool(n_cpus) as p:
        trained_models = p.starmap(
            _training, zip(model_list, repeat(train_data))
        )
    return trained_models


# In[71]:


csv = "benchmark_2.csv"


try:
    df = pd.read_csv(csv)
except:
    df_list = []
    for units in [100, 500, 1000]:
        ####
        list_models = []
        for seed in range(5):
            reservoir = Reservoir(
                units=units,
                sr=0.1,
                lr=0.1,
                seed=seed,
            )
            readout = Ridge(
                ridge=0.1,
            )
            model = ESN(
                reservoir=reservoir,
                readout=readout,
                use_raw_inputs=False,
                feedback=False,
                name="tarace" + str(seed),
            )
            list_models.append(model)
        ####
        for n_cpus in [3, 5]:
            print(f"pool_training, {n_cpus} cpus")

            print(list_models[0])

            t0 = time()
            pool_training(list_models, train_data, n_cpus)
            df_list.append(
                {
                    "units": units,
                    "n_cpus": n_cpus,
                    "function": "pool_training",
                    "time (s)": time() - t0,
                }
            )
        ######
        print("simple_training")
        n_cpus = 1

        t0 = time()
        simple_training(list_models, train_data)
        df_list.append(
            {
                "units": units,
                "n_cpus": n_cpus,
                "function": "simple_training",
                "time (s)": time() - t0,
            }
        )

    df = pd.DataFrame(df_list)
    df.to_csv(csv)


# In[ ]:


units_list = df["units"].unique()
n_plots = len(units_list)

_, axs = plt.subplots(nrows=n_plots, ncols=1, figsize=(10, 10 * n_plots))


for i, units in enumerate(units_list):
    df_plot = df[df["units"] == units]
    sns.barplot(df_plot, x="n_cpus", y="time (s)", hue="function", ax=axs[i])
    axs[i].set_title(f"N={units}")


# In[ ]:
