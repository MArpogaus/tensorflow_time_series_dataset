# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : groupby_dataset_generator.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2022-09-02 16:03:22 (Marcel Arpogaus)
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


class GroupbyDatasetGenerator:
    def __init__(
        self, groupby, columns, dtype=tf.float32, shuffle=False, test_mode=False
    ):
        self.groupby = groupby
        self.columns = sorted(list(set(columns)))
        self.dtype = dtype
        self.shuffle = shuffle
        self.test_mode = test_mode

    def get_generator(self, df):
        df.sort_index(inplace=True)
        if self.test_mode:
            ids = df[self.groupby].unique()
            ids = ids[:2]
            df = df[df[self.groupby].isin(ids)]

        grpd = df.groupby(self.groupby)

        def generator():
            for _, d in grpd:
                yield d[self.columns].values

        return generator

    def __call__(self, df):
        ds = tf.data.Dataset.from_generator(
            self.get_generator(df),
            output_signature=(
                tf.TensorSpec(shape=[None, len(self.columns)], dtype=self.dtype)
            ),
        )
        if self.shuffle:
            len_ids = df[self.groupby].unique().size
            ds = ds.shuffle(len_ids)
        return ds
