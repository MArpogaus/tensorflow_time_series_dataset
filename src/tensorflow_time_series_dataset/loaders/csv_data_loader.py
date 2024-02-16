# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : csv_data_loader.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-16 10:30:32 (Marcel Arpogaus)
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
import pandas as pd


def _read_csv_file(file_path, date_time_col="date_time", **kwds):
    file_path = file_path
    load_data = pd.read_csv(
        file_path,
        parse_dates=[date_time_col],
        index_col=[date_time_col],
        **kwds,
    )

    if load_data.isnull().any().sum() != 0:
        raise ValueError("Data contains NaNs")

    return load_data


class CSVDataLoader:
    def __init__(self, file_path, **kwds):
        self.file_path = file_path
        self.kwds = kwds

    def __call__(self):
        return _read_csv_file(self.file_path, **self.kwds)
