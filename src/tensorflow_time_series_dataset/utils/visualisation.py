# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : visualisation.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 16:50:58 (Marcel Arpogaus)
# changed : 2024-02-19 12:07:12 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# ...
###############################################################################
from typing import Any, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.data import Dataset


def plot_patch(
    ds: Dataset,
    history_size: int,
    prediction_size: int,
    history_columns: List[str],
    prediction_columns: List[str],
    meta_columns: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 8),
    **kwargs: Any,
) -> plt.Figure:
    """Plot history and prediction data from a dataset.

    Parameters
    ----------
    ds : Dataset
        Dataset to extract data from.
    history_size : int
        Number of timesteps in the history.
    prediction_size : int
        Number of timesteps in the prediction.
    history_columns : List[str]
        Columns to use for history data.
    prediction_columns : List[str]
        Columns to use for prediction data.
    meta_columns : List[str], optional
        Columns for additional meta data. Defaults to [].
    figsize : Tuple[int, int], optional
        Size of the figure. Defaults to (16, 8).

    Returns
    -------
    plt.Figure
        The generated plot figure.

    """
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    x, y = next(ds.as_numpy_iterator())
    x1, x2 = None, None

    if history_size and len(history_columns) and len(meta_columns):
        x1, x2 = x
    elif history_size and len(history_columns):
        x1 = x
    elif len(meta_columns):
        x2 = x

    x1_columns = sorted(history_columns)
    x2_columns = sorted(meta_columns)
    y_columns = sorted(prediction_columns)

    x1_column_ch = {k: c for c, k in enumerate(x1_columns)}
    y_column_ch = {k: c for c, k in enumerate(y_columns)}

    fig = plt.figure(figsize=figsize)
    if x1 is not None:
        for c in x1_columns:
            ch = x1_column_ch[c]
            plt.plot(
                np.arange(history_size),
                x1[0, ..., ch],
                color=colors[ch % len(colors)],
                label=c,
            )

    if x2 is not None:
        plt.table(
            cellText=x2[0],
            rowLabels=["meta data"],
            colLabels=x2_columns,
            colLoc="right",
            loc="top",
            edges="horizontal",
        )
    for c in y_columns:
        ch = y_column_ch[c]
        color_idx = x1_column_ch.get(c, ch)
        plt.scatter(
            np.arange(prediction_size) + history_size,
            y[0, ..., ch],
            color=colors[color_idx % len(colors)],
            label=c + " (target)",
            s=32,
            edgecolors="k",
        )

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.1, top=1.2)
    plt.legend(loc="lower right")
    plt.xlabel("Time Steps")
    plt.ylabel("Observations")
    plt.tight_layout()

    return fig
