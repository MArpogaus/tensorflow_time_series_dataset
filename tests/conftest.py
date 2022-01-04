import pytest
import pandas as pd
import numpy as np

# ref.: https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
def encode(data, cycl_name, cycl_max, cycl_min=0, cycl=None):
    if cycl is None:
        cycl = getattr(data.index, cycl_name)
    data[cycl_name + "_sin"] = np.float32(
        np.sin(2 * np.pi * (cycl - cycl_min) / (cycl_max - cycl_min + 1))
    )
    data[cycl_name + "_cos"] = np.float32(
        np.cos(2 * np.pi * (cycl - cycl_min) / (cycl_max - cycl_min + 1))
    )
    return data


def get_value_generator(date_range, columns):
    drf = date_range.to_frame()
    drf_weekday = encode(drf, "weekday", 6, cycl=drf.index.weekday)
    drf_dayofyear = encode(drf, "dayofyear", 366, 1)
    drf_time = encode(
        drf, "time", 24 * 60 - 1, cycl=drf.index.hour * 60 + drf.index.minute
    )

    def gen_value(column, line):
        if column == "weekday":
            return date_range[line].weekday()
        elif "weekday" in column:
            return drf_weekday.loc[date_range[line], column]
        elif "dayofyear" in column:
            return drf_dayofyear.loc[date_range[line], column]
        elif "time" in column:
            return drf_time.loc[date_range[line], column]
        else:
            col_num = columns.index(column)
            return int(f"{col_num:02d}{line:04d}")

    return gen_value


def gen_df(columns, date_range):
    gen_value = get_value_generator(date_range, columns)
    periods = date_range.size
    df = pd.DataFrame(
        {
            "date_time": date_range,
            **{c: [gen_value(c, l) for l in range(periods)] for c in columns},
        }
    )
    return df


def gen_df_with_id(ids, columns, date_range):
    dfs = []
    for i in ids:
        df = gen_df(columns, date_range)
        df["id"] = i
        df[columns] += i * 1e6
        dfs.append(df)
    df = pd.concat(dfs)

    return df


@pytest.fixture(
    scope="function", params=[(["x1", "x2"], 48 * 30), (["x1", "x2"], 48 * 30 * 6)]
)
def time_series_df(request):
    df = gen_df(
        columns=request.param[0],
        date_range=pd.date_range("1/1/1", periods=request.param[1], freq="30T"),
    )
    return df


@pytest.fixture(
    scope="function",
    params=[
        (list(range(10)), ["x1", "x2"], 48 * 30),
        (list(range(10)), ["x1", "x2"], 48 * 30 * 6),
    ],
)
def time_series_df_with_id(request):
    ids, columns, periods = request.param
    df = gen_df_with_id(
        ids=ids,
        columns=columns,
        date_range=pd.date_range("1/1/1", periods=periods, freq="30T"),
    )
    return df


@pytest.fixture(scope="function")
def tmp_csv_with_id(tmpdir_factory, time_series_df_with_id):
    file_path = tmpdir_factory.mktemp("csv_data") / "test.csv"

    time_series_df_with_id.to_csv(file_path, index=False)

    return file_path, time_series_df_with_id


@pytest.fixture(params=[1] + list(range(0, 48 * 2, 48)))
def history_size(request):
    return request.param


@pytest.fixture(params=[0, 1, 48])
def prediction_size(request):
    return request.param


@pytest.fixture
def shift(prediction_size):
    return prediction_size


@pytest.fixture(params=[4, 32])
def batch_size(request):
    return request.param
