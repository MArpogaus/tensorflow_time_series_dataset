# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : windowed_time_series_dataset_factory.py
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
from .pipeline import WindowedTimeSeriesPipeline


class WindowedTimeSeriesDataSetFactory:
    default_pipline_kwds = dict(
        shift=None,
        batch_size=32,
        cycle_length=100,
        shuffle_buffer_size=1000,
        seed=42,
    )

    def __init__(self, **pipline_kwds):
        self.preprocessors = []
        self.data_loader = None
        self.data_pipeline = WindowedTimeSeriesPipeline(
            **{**self.default_pipline_kwds, **pipline_kwds}
        )

    def add_preprocessor(self, preprocessor):
        self.preprocessors.append(preprocessor)

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def get_dataset(self, data=None):
        if self.data_loader is not None and data is None:
            data = self.data_loader()
        elif data is not None:
            data = data
        else:
            raise ValueError("No data provided")

        for preprocessor in self.preprocessors:
            data = preprocessor(data)

        if not isinstance(data, tf.data.Dataset):
            data = tf.data.Dataset.from_tensors(data)

        ds = self.data_pipeline(data)
        return ds

    def __call__(self, data=None):
        return self.get_dataset(data)
