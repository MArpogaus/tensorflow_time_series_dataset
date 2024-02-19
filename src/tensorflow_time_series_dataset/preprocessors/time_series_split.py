# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : time_series_split.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-19 12:09:44 (Marcel Arpogaus)
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


from typing import Any

import numpy as np


class TimeSeriesSplit:
    RIGHT: int = 0
    LEFT: int = 1

    def __init__(self, split_size: float, split: int) -> None:
        """Initialize the TimeSeriesSplit object.

        Parameters
        ----------
        split_size : float
            Proportion of split size for the time series data.
        split : int
            Split position indicator (LEFT or RIGHT).

        """
        self.split_size: float = split_size
        self.split: int = split

    def __call__(self, data: Any) -> Any:
        """Splits the time series data based on the split position.

        Parameters
        ----------
        data : Any
            Input time series data to split.

        Returns
        -------
        Any
            Left or right split of the time series data based on the split position.

        """
        data = data.sort_index()
        days = data.index.date
        days = days.astype("datetime64[m]")
        right = days[int(len(days) * self.split_size)]
        left = right - np.timedelta64(1, "m")
        if self.split == self.LEFT:
            return data.loc[: str(left)]
        else:
            return data.loc[str(right) :]
