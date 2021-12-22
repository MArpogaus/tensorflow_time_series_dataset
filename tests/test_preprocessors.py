import pandas as pd
import numpy as np

from tensorflow_time_series_dataset.preprocessors import (
    CyclicalFeatureEncoder,
    TimeSeriesSplit,
    GroupbyDatasetGenerator,
)

periods = 48 * 10
test_df = pd.DataFrame(
    index=pd.date_range(start="1/1/1992", periods=periods, freq="30T"),
    data=np.stack(
        [np.random.randint(0, 10, periods), np.random.randint(20, 100, periods)],
        axis=1,
    ),
    columns=["x1", "x2"],
)


def test_cyclical_data_encoder():
    encs = {
        "x1": [9],
        "x2": [99, 20],
        "weekday": [6],
        "dayofyear": [366, 1],
        "month": [12, 1],
        "time": [24 * 60 - 1],
    }
    for name in encs.keys():
        if name == "time":
            cycl = test_df.index.hour * 60 + test_df.index.minute
        else:
            try:
                cycl = getattr(test_df, name)
            except AttributeError:
                cycl = getattr(test_df.index, name)
        enc = CyclicalFeatureEncoder(*encs[name])
        enc_dat = enc.encode(cycl)


def test_time_series_split():
    df = test_df.assign(id=1)
    df = df.combine_first(df.assign(id=2))
    l_splitter = TimeSeriesSplit(0.5, TimeSeriesSplit.LEFT)
    r_splitter = TimeSeriesSplit(0.5, TimeSeriesSplit.RIGHT)
    l_split = l_splitter(df)
    r_split = r_splitter(df)
    assert not l_split.index.isin(r_split.index).any(), "Splits overlap"
    assert l_split.index.max() < r_split.index.min(), "wrong split"
    assert np.unique(l_split.index.date).size == np.floor(
        np.unique(df.index.date).size / 2
    ) and np.unique(r_split.index.date).size == np.ceil(
        np.unique(df.index.date).size / 2
    ), "Unexpected split size"
    assert (r_split.groupby([r_split.index.date, "id"]).x1.count() == 48).all() and (
        l_split.groupby([l_split.index.date, "id"]).x1.count() == 48
    ).all(), "Incomplete Days in Split"


def test_groupby_dataset_generator(time_series_df):
    time_series_df.set_index("date_time", inplace=True)
    columns = list(sorted(time_series_df.columns))
    idx_from_column = {c: i for i, c in enumerate(columns)}
    gen = GroupbyDatasetGenerator("id", columns=columns)
    for d in gen(time_series_df):
        id = int(d[0, idx_from_column["load"]] // 1e6)
        assert np.allclose(
            time_series_df[time_series_df.id == id][columns].values, d.numpy()
        ), "Error: DatasetGenerator failed"
