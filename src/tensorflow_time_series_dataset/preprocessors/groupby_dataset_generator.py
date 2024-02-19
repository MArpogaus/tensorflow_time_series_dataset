# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : groupby_dataset_generator.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-19 12:57:02 (Marcel Arpogaus)
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

from typing import Callable, List

import pandas as pd
import tensorflow as tf


class GroupbyDatasetGenerator:
    """Creates a generator that yields the data for each group in the given column.

    Parameters
    ----------
    groupby : str
        The column name to group the dataframe by.
    columns : List[str]
        The list of columns to keep in the dataset.
    dtype : tf.dtypes.DType, optional
        The dtype of the output tensors, by default tf.float32.
    shuffle : bool, optional
        Whether to shuffle the dataset or not, by default False.
    test_mode : bool, optional
        Whether the generator is in test mode or not, by default False.

    """

    def __init__(
        self,
        groupby: str,
        columns: List[str],
        dtype: tf.dtypes.DType = tf.float32,
        shuffle: bool = False,
        test_mode: bool = False,
    ) -> None:
        self.groupby: str = groupby
        self.columns: List[str] = sorted(list(set(columns)))
        self.dtype: tf.dtypes.DType = dtype
        self.shuffle: bool = shuffle
        self.test_mode: bool = test_mode

    def get_generator(self, df: pd.DataFrame) -> Callable[[], tf.Tensor]:
        """Returns the dataset generator with the given parameters.

        Parameters
        ----------
        df : pd.DataFrame
            The source DataFrame to generate data from.

        Returns
        -------
        Callable[[], tf.Tensor]
            A function that when called returns a generator yielding batches of
            data as tensors.

        """
        df.sort_index(inplace=True)
        if self.test_mode:
            ids = df[self.groupby].unique()
            ids = ids[:2]
            df = df[df[self.groupby].isin(ids)]

        grpd = df.groupby(self.groupby)

        def generator() -> tf.Tensor:
            for _, d in grpd:
                yield d[self.columns].values.astype(self.dtype.as_numpy_dtype)

        return generator

    def __call__(self, df: pd.DataFrame) -> tf.data.Dataset:
        """Makes the class instance callable and returns a TensorFlow dataset.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to create the dataset from.

        Returns
        -------
        tf.data.Dataset
            A TensorFlow dataset created from the DataFrame.

        """
        ds: tf.data.Dataset = tf.data.Dataset.from_generator(
            self.get_generator(df),
            output_signature=(
                tf.TensorSpec(shape=[None, len(self.columns)], dtype=self.dtype)
            ),
        )
        if self.shuffle:
            len_ids = df[self.groupby].unique().size
            ds = ds.shuffle(len_ids)
        return ds
