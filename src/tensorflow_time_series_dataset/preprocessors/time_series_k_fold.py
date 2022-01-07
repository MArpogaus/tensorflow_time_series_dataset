# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : time_series_k_fold.py
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
# REQUIRED PYTHON MODULES #####################################################
import numpy as np
import pandas as pd


class TimeSeriesKFold:
    def __init__(self, fold, n_folds=20):
        self.n_folds = n_folds
        self.fold = fold

    def __call__(self, data):
        days = pd.date_range(data.index.min(), data.index.max(), freq="D")
        fold_idx = np.array_split(days.to_numpy(), self.n_folds)
        folds = {
            f: (idx[[0, -1]].astype("datetime64[m]") + [0, 60 * 24 - 1])
            .astype(str)
            .tolist()
            for f, idx in enumerate(fold_idx)
        }

        return data[folds[self.fold][0] : folds[self.fold][1]]
