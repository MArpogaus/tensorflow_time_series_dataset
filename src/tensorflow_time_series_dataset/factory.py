# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : windowed_time_series_dataset_factory.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-09-12 16:21:24 (Marcel Arpogaus)
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
from typing import Any, Dict, List

import tensorflow as tf

from .pipeline import WindowedTimeSeriesPipeline


class WindowedTimeSeriesDatasetFactory:
    """A factory to initialize a new WindowedTimeSeriesPipeline.

    Parameters
    ----------
    history_columns : List[str]
        List of columns representing the history data.
    prediction_columns : List[str]
        List of columns representing the prediction data.
    meta_columns : List[str], optional
        List of columns representing the meta data, by default [].
    dtype : tf.DType, optional
        Data type of the dataset, by default tf.float32.
    **pipeline_kwargs : Any
        Keyword arguments for the data pipeline.

    """

    default_pipeline_kwargs: Dict[str, Any] = {
        "shift": None,
        "batch_size": 32,
        "cycle_length": 1,
        "shuffle_buffer_size": 1000,
        "cache": True,
        "filter_nans": False,
    }

    def __init__(
        self,
        history_columns: List[str],
        prediction_columns: List[str],
        meta_columns: List[str] = [],
        dtype: tf.DType = tf.float32,
        **pipeline_kwargs: Any,
    ) -> None:
        self.columns: set = set(history_columns + meta_columns + prediction_columns)
        self.preprocessors: List[Any] = []
        self.data_loader: Any = None
        self.dtype: tf.DType = dtype
        self.data_pipeline: WindowedTimeSeriesPipeline = WindowedTimeSeriesPipeline(
            history_columns=history_columns,
            meta_columns=meta_columns,
            prediction_columns=prediction_columns,
            **{**self.default_pipeline_kwargs, **pipeline_kwargs},
        )

    def add_preprocessor(self, preprocessor: Any) -> None:
        """Add a preprocessor to the list of preprocessors.

        Parameters
        ----------
        preprocessor : Any
            The preprocessor function to be added.

        """
        self.preprocessors.append(preprocessor)

    def set_data_loader(self, data_loader: Any) -> None:
        """Set a data loader for the dataset.
        bk.

        Parameters
        ----------
        data_loader : Any
            The data loader function.

        """
        self.data_loader = data_loader

    def get_dataset(self, data: Any = None) -> tf.data.Dataset:
        """Get the dataset for the given data.

        Parameters
        ----------
        data : Any, optional
            The data to create the dataset from, by default None.

        Returns
        -------
        tf.data.Dataset
            The created TensorFlow dataset.

        """
        if self.data_loader is not None and data is None:
            data = self.data_loader()
        elif data is not None:
            data = data
        else:
            raise ValueError("No data provided")

        for preprocessor in self.preprocessors:
            data = preprocessor(data)

        if not isinstance(data, tf.data.Dataset):
            data = data[sorted(self.columns)]
            tensors = tf.convert_to_tensor(data, dtype=self.dtype)
            data = tf.data.Dataset.from_tensors(tensors)

        ds = self.data_pipeline(data)
        return ds

    def __call__(self, data: Any = None) -> tf.data.Dataset:
        """Call method to get the dataset.

        Parameters
        ----------
        data : Any, optional
            The data to create the dataset from, by default None.

        Returns
        -------
        tf.data.Dataset
            The created TensorFlow dataset.

        """
        return self.get_dataset(data)
