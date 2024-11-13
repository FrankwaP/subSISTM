from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
import seaborn as sns
from pandas import DataFrame
from optuna.trial import FrozenTrial

from optuna_training import ModelConfiguration, get_3D_prediction_list, ProcessedData

sns.set_style("darkgrid")
sns.set(rc={"figure.figsize": (15, 10)})


def get_2D_prediction_list(
    model_list: list[ModelConfiguration], processed_data: ProcessedData, n_cpus: int
) -> list[NDArray]:
    # !!! 2D prediction are unscaled
    return [
        processed_data.to_unscaled_test_2D(pred_3D)
        for pred_3D in get_3D_prediction_list(model_list, processed_data, n_cpus)
    ]


def get_predictions_dataframe(
    model_list: list[ModelConfiguration], processed_data: ProcessedData, n_cpus: int
) -> DataFrame:
    y_labels = processed_data.y_labels_test
    assert len(y_labels) == 1
    y_label = y_labels[0]
    sercol = processed_data.series_column_name
    tstepcol = processed_data.timestep_column_name
    x_labels = processed_data.x_labels
    data_test = processed_data.data_test[[sercol, tstepcol] + y_labels]
    # used to avoid "A value is trying to be set on a copy of a slice from a DataFrame."
    data_test = data_test.copy()
    ###
    err_cols = []
    for seed, pred in enumerate(
        get_2D_prediction_list(model_list, processed_data, n_cpus)
    ):
        pred_name = f"prediction-{seed}"
        data_test.loc[:, pred_name] = pred
        err_name = f"error-{seed}"
        data_test.loc[:, err_name] = data_test[y_label] - data_test[pred_name]
        err_cols.append(err_name)

    data_test.loc[:, "mean-absolute-error"] = data_test[err_cols].abs().mean(axis=1)

    return data_test


def remove_warmups(
    pred_dataframe: DataFrame, timestep_column_name: str, N_warmups: int
) -> DataFrame:
    times = pred_dataframe[timestep_column_name].unique()
    return pred_dataframe[pred_dataframe[timestep_column_name] > times[N_warmups]]


def get_statistics(
    pred_dataframe: DataFrame,
    serie_column_name: str,
    timestep_column_name: str,
    N_warmups: int,
) -> None:

    err_name = [c for c in pred_dataframe.columns if c.startswith("error")]
    data = remove_warmups(pred_dataframe, timestep_column_name, N_warmups)

    mae_seeds = data[err_name].abs().mean(axis=1)
    mse_seeds = (data[err_name] ** 2).mean(axis=1)

    mae_global = data[err_name].abs().mean(axis=0).mean()
    print(f"\tMAE: {mae_global}")
    mse_global = (data[err_name] ** 2).mean(axis=0).mean()
    print(f"\tMSE: {mse_global}")

    plot_data = DataFrame()

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(5 * 2, 5))
    sns.boxplot(mae_seeds, ax=axs[0])
    axs[0].set_yscale("log")
    axs[0].set_title(f"MAE = {mae_global:2g}")
    sns.boxplot(mse_seeds, ax=axs[1])
    axs[1].set_yscale("log")
    axs[1].set_title(f"MSE = {mse_global:2g}")


def get_worst_individuals(
    pred_dataframe: DataFrame,
    serie_column_name: str,
    timestep_column_name: str,
    N_warmups: int,
    N_worst: int,
) -> list[int]:
    print(
        f"Returning the {N_worst} worst MAE individuals (mean over seeds and timesteps)."
    )
    return (
        remove_warmups(pred_dataframe, timestep_column_name, N_warmups)
        .loc[:, [serie_column_name, "mean-absolute-error"]]
        .groupby(serie_column_name)
        .mean()
        .sort_values("mean-absolute-error", ascending=False)
        .iloc[:N_worst]
        .index.tolist()
    )


def plot_individual_results(
    pred_dataframe: DataFrame,
    ind_idx: int,
    serie_column_name: str,
    timestep_column_name: str,
    y_label: str,
) -> None:
    assert isinstance(y_label, str)
    indiv_dataframe = pred_dataframe[pred_dataframe[serie_column_name] == ind_idx]

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10 * 2, 5))

    AX, PREFIX = 0, "prediction"
    list_cols = [c for c in indiv_dataframe.columns if c.startswith(PREFIX)]
    plot_dataframe = indiv_dataframe[
        [serie_column_name, timestep_column_name] + list_cols + [y_label]
    ]
    _plot_individual_single_result(
        axs[AX],
        PREFIX,
        ind_idx,
        plot_dataframe,
        serie_column_name,
        timestep_column_name,
    )

    AX, PREFIX = 1, "error"
    list_cols = [c for c in indiv_dataframe.columns if c.startswith(PREFIX)]
    plot_dataframe = indiv_dataframe[
        [serie_column_name, timestep_column_name] + list_cols
    ]
    plot_dataframe.loc[:, list_cols] = plot_dataframe[list_cols].abs()
    axs[AX].set_yscale("log")
    _plot_individual_single_result(
        axs[AX],
        PREFIX,
        ind_idx,
        plot_dataframe,
        serie_column_name,
        timestep_column_name,
    )


def _plot_individual_single_result(
    ax: Axes,
    prefix: str,
    serie_idx: int,
    plot_dataframe: DataFrame,
    serie_column_name: str,
    timestep_column_name: str,
) -> None:

    VAR_NAME = "variable"
    VALUE_NAME = "value"
    plot_dataframe = plot_dataframe.melt(
        id_vars=[serie_column_name, timestep_column_name],
        var_name=VAR_NAME,
        value_name=VALUE_NAME,
    )

    sns.lineplot(
        plot_dataframe,  # .sort_values([VAR_NAME]),
        x=timestep_column_name,
        y=VALUE_NAME,
        hue=VAR_NAME,
        style=VAR_NAME,
        markers=True,
        ax=ax,
    )
    ax.set_title(f"{prefix} for individual #{serie_idx}")


# def _fix_axes(axs: Union[Axes, NDArray]) -> NDArray:
#     # correct the irregular output of subplots for ax
#     # so it's always a 2D NDArray of Axes
#     if isinstance(axs, Axes):
#         axs = np.array([axs])
#     if len(axs.shape) != 2:
#         axs = np.expand_dims(axs, axis=0)
#     return axs
