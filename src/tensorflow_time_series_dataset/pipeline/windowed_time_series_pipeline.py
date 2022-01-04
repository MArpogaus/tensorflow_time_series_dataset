#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : windowed_time_series_pipeline.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2022-01-04 12:11:45 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# Probabilistic Short-Term Low-Voltage Load Forecasting using
# Bernstein-Polynomial Normalizing Flows
# LICENSE #####################################################################
# Copyright (C) 2021 Marcel Arpogaus
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
###############################################################################
# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf

from tensorflow_time_series_dataset.pipeline.batch_processor import BatchPreprocessor
from tensorflow_time_series_dataset.pipeline.patch_generator import PatchGenerator


class WindowedTimeSeriesPipeline:
    def __init__(
        self,
        history_size,
        prediction_size,
        history_columns,
        meta_columns,
        prediction_columns,
        shift,
        batch_size,
        cycle_length,
        shuffle_buffer_size,
        seed,
    ):
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
        self.seed = seed

    def __call__(self, ds):

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(
                self.cycle_length * self.shuffle_buffer_size, seed=self.seed
            )

        ds = ds.interleave(
            PatchGenerator(self.window_size, self.shift),
            cycle_length=self.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(self.batch_size * self.shuffle_buffer_size, seed=self.seed)

        ds = ds.batch(self.batch_size, drop_remainder=True)

        ds = ds.map(
            BatchPreprocessor(
                self.history_size,
                self.history_columns,
                self.meta_columns,
                self.prediction_columns,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds
