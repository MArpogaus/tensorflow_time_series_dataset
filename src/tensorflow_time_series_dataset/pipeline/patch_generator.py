# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : patch_generator.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-19 12:52:06 (Marcel Arpogaus)
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

    """

    def __init__(self, window_size: int, shift: int) -> None:
        """Parameters
        ----------
        window_size : int
            The size of each patch.
        shift : int
            The shift between patches.

        """
        self.window_size: int = window_size
        self.shift: int = shift

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
        ).flat_map(lambda sub: sub.batch(self.window_size, drop_remainder=True))

        return data_set
