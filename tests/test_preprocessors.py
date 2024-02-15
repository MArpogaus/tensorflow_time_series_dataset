import numpy as np
import pandas as pd
import pytest
from tensorflow_time_series_dataset.preprocessors import (
    CyclicalFeatureEncoder,
    GroupbyDatasetGenerator,
    TimeSeriesSplit,
)


@pytest.fixture
def cycl_df():
    date_range = pd.date_range(start="1/1/1992", end="31/12/1992", freq="30T")
    test_df = pd.DataFrame(
        index=date_range,
        data=np.stack(
            [
                np.random.randint(0, 10, date_range.size),
                np.random.randint(20, 100, date_range.size),
            ],
            axis=1,
        ),
        columns=["x1", "x2"],
    )
    return test_df


def test_cyclical_data_encoder(cycl_df):
    encs = {
        "x1": dict(cycl_max=9),
        "x2": dict(cycl_max=99, cycl_min=20),
        "weekday": dict(cycl_max=6),
        "dayofyear": dict(cycl_max=366, cycl_min=1),
        "month": dict(cycl_max=12, cycl_min=1),
        "time": dict(
            cycl_max=24 * 60 - 1,
            cycl_getter=lambda df, k: df.index.hour * 60 + df.index.minute,
        ),
    }
    for name, kwds in encs.items():
        enc = CyclicalFeatureEncoder(name, **kwds)
        enc_dat = enc(cycl_df)
        cycl = enc.cycl_getter(cycl_df, name)
        assert np.isclose(
            enc.decode(enc_dat[name + "_sin"], enc_dat[name + "_cos"]), cycl
        ).all(), "Decoding failed"


def test_cyclical_data_encoder_except(cycl_df):
    encs = {
        "x1": dict(cycl_max=cycl_df.x1.max() - 1),
        "weekday": dict(cycl_max=cycl_df.index.weekday.max() - 1),
        "dayofyear": dict(cycl_max=cycl_df.index.dayofyear.max() - 1),
        "month": dict(cycl_max=cycl_df.index.month.max() - 1),
        "time": dict(
            cycl_max=(cycl_df.index.hour * 60 + cycl_df.index.minute).max() - 1,
            cycl_getter=lambda df, k: df.index.hour * 60 + df.index.minute,
        ),
    }
    for name, kwds in encs.items():
        enc = CyclicalFeatureEncoder(name, **kwds)
        with pytest.raises(AssertionError):
            enc(cycl_df)


def test_time_series_split(time_series_df_with_id):
    df = time_series_df_with_id.set_index("date_time")
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


def test_groupby_dataset_generator(time_series_df_with_id):
    df = time_series_df_with_id.set_index("date_time")
    columns = list(sorted(df.columns))
    df = df[columns]

    records_per_id = df.groupby("id").size().max()
    expected_shape = (records_per_id, len(columns))

    idx_from_column = {c: i for i, c in enumerate(columns)}
    gen = GroupbyDatasetGenerator("id", columns=columns)
    for d in gen(df.sample(frac=1)):
        assert d.shape == expected_shape, "Wrong shape"
        id = int(d[0, idx_from_column["ref"]] // 1e5)
        expected_values = df[df.id == id].sort_index().values
        assert np.all(
            d.numpy() == expected_values,
        ), "Error: DatasetGenerator failed"
