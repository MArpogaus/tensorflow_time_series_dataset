import numpy as np


def default_cycl_getter(df, k):
    try:
        cycl = getattr(df, k)
    except AttributeError:
        cycl = getattr(df.index, k)
    return cycl


class CyclicalFeatureEncoder:
    def __init__(
        self, cycl_name, cycl_max, cycl_min=0, cycl_getter=default_cycl_getter
    ):
        self.cycl_max = cycl_max
        self.cycl_min = cycl_min
        self.cycl_getter = cycl_getter
        self.cycl_name = cycl_name

    def encode(self, data):
        cycl = self.cycl_getter(data, self.cycl_name)
        sin = np.sin(
            2 * np.pi * (cycl - self.cycl_min) / (self.cycl_max - self.cycl_min + 1)
        )
        cos = np.cos(
            2 * np.pi * (cycl - self.cycl_min) / (self.cycl_max - self.cycl_min + 1)
        )
        assert np.allclose(
            cycl, self.decode(sin, cos)
        ), f'Decoding failed. Is "cycl_min/max" ({self.cycl_min}/{self.cycl_max}) correct?'
        return sin, cos

    def __call__(self, data):
        data = data.copy()
        sin, cos = self.encode(data)
        data[self.cycl_name + "_sin"] = sin
        data[self.cycl_name + "_cos"] = cos
        return data

    def decode(self, sin, cos):
        angle = (np.arctan2(sin, cos) + 2 * np.pi) % (2 * np.pi)
        return (angle * (self.cycl_max - self.cycl_min + 1)) / (
            2 * np.pi
        ) + self.cycl_min