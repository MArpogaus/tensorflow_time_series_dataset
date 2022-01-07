import pandas as pd


class TimeSeriesSplit:
    RIGHT = 0
    LEFT = 1

    def __init__(self, split_size, split):
        self.split_size = split_size
        self.split = split

    def __call__(self, data):
        days = pd.date_range(data.index.min(), data.index.max(), freq="D")
        days = days.to_numpy().astype("datetime64[m]")
        right = days[int(len(days) * self.split_size)]
        left = right - 1
        if self.split == self.LEFT:
            return data.loc[: str(left)]
        else:
            return data.loc[str(right) :]
