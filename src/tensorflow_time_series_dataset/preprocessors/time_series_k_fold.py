#!/usr/bin/env python3
# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : time_series_k_fold.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2021-07-29 18:02:20 (Marcel Arpogaus)
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
import numpy as np
import pandas as pd


class TimeSeriesKFold():
    def __init__(self, fold, n_folds=20):
        self.n_folds = n_folds
        self.fold = fold

    def __call__(self, data):
        days = pd.date_range(data.index.min(), data.index.max(), freq='D')
        fold_idx = np.array_split(days.to_numpy(), self.n_folds)
        folds = {f: (idx[[0, -1]].astype('datetime64[m]') + [0, 60 * 24 - 1]
                     ).astype(str).tolist() for f, idx in enumerate(fold_idx)}

        return data[folds[self.fold][0]:folds[self.fold][1]]
