import itertools

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

columns = ["ref", "x1", "x2"]

seed = 1
tf.random.set_seed(seed)
np.random.seed(seed)


def gen_value(column, line, id=0):
    if column == "ref":
        return id * 1e5 + line
    else:
        return np.random.randint(0, 1000)


def gen_df(columns, date_range, id=0):
    periods = date_range.size
    df = pd.DataFrame(
        {
            "date_time": date_range,
            **{
                col: [gen_value(col, line, id) for line in range(periods)]
                for col in columns
            },
        }
    )
    return df


def gen_df_with_id(ids, columns, date_range):
    dfs = []
    for i in ids:
        df = gen_df(columns, date_range, i)
        df["id"] = i
        dfs.append(df)
    df = pd.concat(dfs)

    return df


@pytest.fixture(
    scope="function",
    params=[(columns, 48 * 30 * 3)],
)
def time_series_df(request):
    df = gen_df(
        columns=request.param[0],
        date_range=pd.date_range("1/1/1", periods=request.param[1], freq="30min"),
    )
    return df


@pytest.fixture(
    scope="function",
    params=[
        (list(range(5)), columns, 48 * 30 * 3),
    ],
)
def time_series_df_with_id(request):
    ids, columns, periods = request.param
    df = gen_df_with_id(
        ids=ids,
        columns=columns,
        date_range=pd.date_range("1/1/1", periods=periods, freq="30min"),
    )
    return df


@pytest.fixture(scope="function")
def tmp_csv(tmpdir_factory, time_series_df):
    file_path = tmpdir_factory.mktemp("csv_data") / "test.csv"

    time_series_df.to_csv(file_path, index=False)

    return file_path, time_series_df


@pytest.fixture(scope="function")
def tmp_csv_with_id(tmpdir_factory, time_series_df_with_id):
    file_path = tmpdir_factory.mktemp("csv_data") / "test.csv"

    time_series_df_with_id.to_csv(file_path, index=False)

    return file_path, time_series_df_with_id


@pytest.fixture(params=[0, 1, 48])
def history_size(request):
    return request.param


@pytest.fixture
def prediction_size(history_size):
    return history_size


@pytest.fixture
def shift(prediction_size):
    return prediction_size


@pytest.fixture(params=[32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[[], ["ref"], ["x2", "ref"], columns])
def history_columns(request):
    return request.param


@pytest.fixture(params=[[], ["ref", "x1"]])
def meta_columns(request):
    return request.param


@pytest.fixture(
    params=[
        [],
        list(
            itertools.chain(
                ["ref"],
                *[
                    [c + "_sin", c + "_cos"]
                    for c in ["weekday", "dayofyear", "time", "month"]
                ],
            )
        ),
    ]
)
def meta_columns_cycle(request):
    return request.param


@pytest.fixture
def prediction_columns(history_columns):
    return history_columns
