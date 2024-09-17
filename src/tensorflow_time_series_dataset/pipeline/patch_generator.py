# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : patch_generator.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-09-12 15:52:32 (Marcel Arpogaus)
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

import tensorflow as tf


class PatchGenerator:
    """A generator class that creates windows from provided time-series data.

    Attributes
    ----------
    window_size : int
        The size of each patch.
    shift : int
        The shift between patches.
    filter_nans : int
        Apply a filter function to drop patches containing NaN values.

    """

    def __init__(self, window_size: int, shift: int, filter_nans: bool) -> None:
        """Parameters
        ----------
        window_size : int
            The size of each patch.
        shift : int
            The shift between patches.
        filter_nans : int
            If True, apply a filter function to drop patches containing NaN values.

        """
        self.window_size: int = window_size
        self.shift: int = shift
        self.filter_nans: bool = filter_nans

    def __call__(self, data: tf.Tensor) -> tf.data.Dataset:
        """Converts input data into patches of provided window size.

        Parameters
        ----------
        data : tf.Tensor
            The input data to generate patches from.

        Returns
        -------
        tf.data.Dataset
            A dataset of patches generated from the input data.

        """
        data_set: tf.data.Dataset = tf.data.Dataset.from_tensor_slices(data)

        data_set = data_set.window(
            size=self.window_size,
            shift=self.shift,
            drop_remainder=True,
        )

        def sub_to_patch(sub):
            return sub.batch(self.window_size, drop_remainder=True)

        data_set = data_set.flat_map(sub_to_patch)

        if self.filter_nans:

            def filter_func(patch):
                return tf.reduce_all(tf.logical_not(tf.math.is_nan(patch)))

            data_set = data_set.filter(filter_func)

        return data_set
