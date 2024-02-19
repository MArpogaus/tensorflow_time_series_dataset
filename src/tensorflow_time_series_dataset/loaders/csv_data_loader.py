# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : csv_data_loader.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-19 13:01:07 (Marcel Arpogaus)
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
from typing import Union

import pandas as pd


def _read_csv_file(
    file_path: str, date_time_col: str = "date_time", **kwargs: Union[str, bool]
) -> pd.DataFrame:
    """Read CSV file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        File path to the CSV file.
    date_time_col : str, optional
        Name of the datetime column in the CSV file.
    **kwargs
        Additional keyword arguments for pd.read_csv.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the data from the CSV file.

    Raises
    ------
    ValueError
        If the data contains NaN values.

    """
    load_data = pd.read_csv(
        file_path,
        parse_dates=[date_time_col],
        index_col=[date_time_col],
        **kwargs,
    )

    if load_data.isnull().any().sum() != 0:
        raise ValueError("Data contains NaNs")

    return load_data


class CSVDataLoader:
    """Load data from a CSV file.

    Parameters
    ----------
    file_path : str
        File path to the CSV file.
    **kwargs
        Additional keyword arguments for pd.read_csv.

    """

    def __init__(self, file_path: str, **kwargs: Union[str, bool]):
        self.file_path = file_path
        self.kwargs = kwargs

    def __call__(self) -> pd.DataFrame:
        """Load data from the CSV file using _read_csv_file.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the data from the CSV file.

        """
        return _read_csv_file(self.file_path, **self.kwargs)
