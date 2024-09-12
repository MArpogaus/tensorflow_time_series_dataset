# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : windowed_time_series_pipeline.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-09-12 16:01:27 (Marcel Arpogaus)
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
# REQUIRED PYTHON MODULES #####################################################
from typing import List, Union

import tensorflow as tf
from tensorflow.data import Dataset

from tensorflow_time_series_dataset.pipeline.patch_generator import PatchGenerator
from tensorflow_time_series_dataset.pipeline.patch_processor import PatchPreprocessor


class WindowedTimeSeriesPipeline:
    """A Pipeline to process time-series data in a rolling window for TensorFlow models.

    This class provides functionality to generate windowed datasets from time series
    data for training, validating, and testing machine learning models. It allows for
    the generation of datasets that include historical data windows along with
    corresponding future values (for either regression or classification tasks).

    Parameters
    ----------
    history_size : int
        The size of the history window.
    prediction_size : int
        The size of the prediction window.
    history_columns : List[str]
        The names of the columns to be used for the history window.
    meta_columns : List[str]
        The names of the columns to be used for metadata in each window.
    prediction_columns : List[str]
        The names of the columns from which the future values are predicted.
    shift : int
        The shift (in time units) for the sliding window.
    batch_size : int
        The size of the batch for each dataset iteration.
    cycle_length : int
        The number of data files that the dataset can process in parallel.
    shuffle_buffer_size : int
        The buffer size for data shuffling.
    cache : Union[str, bool]
        Whether to cache the dataset in memory or to a specific file.
    drop_remainder : bool
        Whether to drop the remainder of batches that are not equal to the batch size.
    filter_nans : int
        Apply a filter function to drop patches containing NaN values.

    Raises
    ------
    AssertionError
        If the prediction_size is not greater than zero.

    """

    def __init__(
        self,
        history_size: int,
        prediction_size: int,
        history_columns: List[str],
        meta_columns: List[str],
        prediction_columns: List[str],
        shift: int,
        batch_size: int,
        cycle_length: int,
        shuffle_buffer_size: int,
        cache: Union[str, bool],
        drop_remainder: bool,
        filter_nans: bool,
    ) -> None:
        assert (
            prediction_size > 0
        ), "prediction_size must be a positive integer greater than zero"
        self.history_size = history_size
        self.prediction_size = prediction_size
        self.window_size = history_size + prediction_size
        self.history_columns = history_columns
        self.meta_columns = meta_columns
        self.prediction_columns = prediction_columns
        self.shift = shift
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.cache = cache
        self.drop_remainder = drop_remainder
        self.filter_nans = filter_nans

    def __call__(self, ds: Dataset) -> Dataset:
        """Applies the pipeline operations to the given dataset.

        Parameters
        ----------
        ds : Dataset
            The input dataset to transform.

        Returns
        -------
        Dataset
            The transformed dataset, ready for model training or evaluation.

        """
        ds = ds.interleave(
            PatchGenerator(self.window_size, self.shift, self.filter_nans),
            cycle_length=self.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        ds = ds.map(
            PatchPreprocessor(
                self.history_size,
                self.history_columns,
                self.meta_columns,
                self.prediction_columns,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if self.cache:
            ds = ds.cache()

        if self.shuffle_buffer_size:
            ds = ds.shuffle(self.shuffle_buffer_size)

        ds = ds.batch(self.batch_size, drop_remainder=self.drop_remainder)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds
