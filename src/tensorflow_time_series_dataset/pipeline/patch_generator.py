# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : patch_generator.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2022-01-07 09:02:38 (Marcel Arpogaus)
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
    def __init__(self, window_size, shift):
        self.window_size = window_size
        self.shift = shift

    def __call__(self, data):
        def sub_to_patch(sub):
            return sub.batch(self.window_size, drop_remainder=True)

        data_set = tf.data.Dataset.from_tensor_slices(data)

        data_set = data_set.window(
            size=self.window_size,
            shift=self.shift,
            drop_remainder=True,
        )
        data_set = data_set.flat_map(sub_to_patch)

        return data_set
