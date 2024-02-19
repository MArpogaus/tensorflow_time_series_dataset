# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : cyclical_feature_encoder.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-19 12:53:30 (Marcel Arpogaus)
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
from typing import Any, Callable, Tuple

import numpy as np
import pandas as pd


def default_cycl_getter(df: pd.DataFrame, k: str) -> Any:
    """Get the cyclical feature 'k' from a DataFrame or its index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame object.
    k : str
        Key to get the cyclical feature.

    Returns
    -------
    Any
        Cyclical feature corresponding to 'k'.

    """
    try:
        cycl = getattr(df, k)
    except AttributeError:
        cycl = getattr(df.index, k)
    return cycl


class CyclicalFeatureEncoder:
    def __init__(
        self,
        cycl_name: str,
        cycl_max: int,
        cycl_min: int = 0,
        cycl_getter: Callable = default_cycl_getter,
    ) -> None:
        """Initialize the CyclicalFeatureEncoder object.

        Parameters
        ----------
        cycl_name : str
            Name of the cyclical feature.
        cycl_max : int
            Maximum value of the cyclical feature.
        cycl_min : int, optional
            Minimum value of the cyclical feature, by default 0.
        cycl_getter : Callable, optional
            Function to get the cyclical feature, by default default_cycl_getter.

        """
        self.cycl_max = cycl_max
        self.cycl_min = cycl_min
        self.cycl_getter = cycl_getter
        self.cycl_name = cycl_name

    def encode(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Encode the cyclical feature into sine and cosine components.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing the cyclical feature.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Sine and cosine components of the encoded cyclical feature.

        """
        cycl = self.cycl_getter(data, self.cycl_name)
        sin = np.sin(
            2 * np.pi * (cycl - self.cycl_min) / (self.cycl_max - self.cycl_min + 1)
        )
        cos = np.cos(
            2 * np.pi * (cycl - self.cycl_min) / (self.cycl_max - self.cycl_min + 1)
        )
        assert np.allclose(cycl, self.decode(sin, cos)), (
            'Decoding failed. Is "cycl_min/max"'
            f"({self.cycl_min}/{self.cycl_max}) correct?"
        )
        return sin, cos

    def __call__(self, data: pd.DataFrame) -> pd.DataFrame:
        """Call method to encode the cyclical feature and add the sine and cosine
        components as new columns in the data.

        Parameters
        ----------
        data : pd.DataFrame
            Data containing the cyclical feature.

        Returns
        -------
        pd.DataFrame
            Updated data with sine and cosine components.

        """
        data = data.copy()
        sin, cos = self.encode(data)
        data[self.cycl_name + "_sin"] = sin
        data[self.cycl_name + "_cos"] = cos
        return data

    def decode(self, sin: np.ndarray, cos: np.ndarray) -> np.ndarray:
        """Decode the encoded sine and cosine components back to the cyclical feature.

        Parameters
        ----------
        sin : np.ndarray
            Sine component of the encoded cyclical feature.
        cos : np.ndarray
            Cosine component of the encoded cyclical feature.

        Returns
        -------
        np.ndarray
            Decoded cyclical feature.

        """
        angle = (np.arctan2(sin, cos) + 2 * np.pi) % (2 * np.pi)
        return (angle * (self.cycl_max - self.cycl_min + 1)) / (
            2 * np.pi
        ) + self.cycl_min
