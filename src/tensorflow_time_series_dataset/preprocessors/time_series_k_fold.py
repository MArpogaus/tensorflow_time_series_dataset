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
