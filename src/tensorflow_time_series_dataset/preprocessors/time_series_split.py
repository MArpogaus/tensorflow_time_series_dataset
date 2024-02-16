# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : time_series_split.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-16 10:28:30 (Marcel Arpogaus)
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


class TimeSeriesSplit:
    RIGHT = 0
    LEFT = 1

    def __init__(self, split_size, split):
        self.split_size = split_size
        self.split = split

    def __call__(self, data):
        data = data.sort_index()
        days = data.index.date
        days = days.astype("datetime64[m]")
        right = days[int(len(days) * self.split_size)]
        left = right - 1
        if self.split == self.LEFT:
            return data.loc[: str(left)]
        else:
            return data.loc[str(right) :]
