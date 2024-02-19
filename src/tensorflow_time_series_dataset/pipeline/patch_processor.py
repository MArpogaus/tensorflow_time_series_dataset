# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : patch_processor.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-19 12:41:38 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# Copyright 2022 Marcel Arpogaus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
from typing import Dict, List, Tuple

import tensorflow as tf


class PatchPreprocessor:
    """A class for preprocessing patches of time series data.

    Parameters
    ----------
    history_size : int
        The number of past time steps to include for each sample.
    history_columns : List[str]
        The names of columns in the dataset that are part of the historical data.
    meta_columns : List[str]
        The names of columns in the dataset that are part of the meta data.
    prediction_columns : List[str]
        The names of columns in the dataset that are to be predicted.

    Attributes
    ----------
    history_size : int
        The number of past time steps to include for each sample.
    history_columns : List[str]
        The names of columns in the dataset that are part of the historical data.
    meta_columns : List[str]
        The names of columns in the dataset that are part of the meta data.
    prediction_columns : List[str]
        The names of columns in the dataset that are to be predicted.
    column_idx : Dict[str, int]
        A mapping from column names to their indices in the input dataset.

    """

    def __init__(
        self,
        history_size: int,
        history_columns: List[str],
        meta_columns: List[str],
        prediction_columns: List[str],
    ) -> None:
        self.history_size: int = history_size
        self.history_columns: List[str] = history_columns
        self.meta_columns: List[str] = meta_columns
        self.prediction_columns: List[str] = prediction_columns

        assert (
            len(set(history_columns + meta_columns)) != 0
        ), "No feature columns provided"
        if len(meta_columns) == 0:
            assert history_size > 0, (
                "history_size must be a positive integer greater than zero"
                ", when no meta date is used"
            )
        else:
            assert history_size >= 0, "history_size must be a positive integer"
        assert len(prediction_columns) != 0, "No prediction columns provided"

        columns = sorted(list(set(history_columns + prediction_columns + meta_columns)))
        self.column_idx: Dict[str, int] = {c: i for i, c in enumerate(columns)}

    def __call__(self, patch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Process a single patch of data.

        Parameters
        ----------
        patch : tf.Tensor
            A tensor representing a patch of time series data.

        Returns
        -------
        Tuple[tf.Tensor, tf.Tensor]
            A tuple containing the input features (as one or more tensors) and
            the target values as tensors.

        """
        y: List[tf.Tensor] = []
        x_hist: List[tf.Tensor] = []
        x_meta: List[tf.Tensor] = []

        x_columns = sorted(set(self.history_columns + self.meta_columns))
        y_columns = sorted(self.prediction_columns)

        assert (
            len(set(x_columns + y_columns)) == patch.shape[-1]
        ), "Patch shape does not match column number"

        for c in y_columns:
            column = patch[self.history_size :, self.column_idx[c]]
            y.append(column)

        for c in x_columns:
            column = patch[:, self.column_idx[c], None]
            if self.history_size and c in self.history_columns:
                x_hist.append(column[: self.history_size, 0])
            if c in self.meta_columns:
                x_meta.append(column[self.history_size, None, ...])

        y = tf.stack(y, axis=-1)
        x: List[tf.Tensor] = []
        if len(x_hist):
            x.append(tf.stack(x_hist, axis=-1))
        if len(x_meta):
            x.append(tf.concat(x_meta, axis=-1))

        if len(x) > 1:
            return tuple(x), y
        else:
            return x[0], y
