import pandas as pd


def _read_csv_file(file_path, date_time_col="date_time", **kwds):
    file_path = file_path
    load_data = pd.read_csv(
        file_path,
        parse_dates=[date_time_col],
        infer_datetime_format=True,
        index_col=[date_time_col],
        **kwds
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
