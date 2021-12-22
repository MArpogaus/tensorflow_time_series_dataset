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

    def gen_value(id, column, line):
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
            return int(f"{id:02d}{col_num:02d}{line:04d}")

    return gen_value


def gen_df(ids, columns, date_range):
    gen_value = get_value_generator(date_range, columns)
    periods = date_range.size
    dfs = []
    for i in ids:
        df = pd.DataFrame(
            {
                "date_time": date_range,
                "id": i,
                **{
                    c: [gen_value(int(i), c, l) for l in range(periods)]
                    for n, c in enumerate(columns)
                    if c != "weekday"
                },
            }
        )
        dfs.append(df)
    df = pd.concat(dfs)
    df["weekday"] = df.date_time.dt.weekday

    return df


@pytest.fixture(scope="function", params=[48 * 30, 48 * 30 * 6])
def time_series_df(request):
    df = gen_df(
        ids=list(range(10)),
        columns=["load", "weekday", "is_holiday"],
        date_range=pd.date_range("1/1/1", periods=request.param, freq="30T"),
    )
    return df


@pytest.fixture(scope="function")
def tmpdf(tmpdir_factory, time_series_df):
    file_path = tmpdir_factory.mktemp("csv_data") / "test.csv"

    time_series_df.to_csv(file_path, index=False)

    return file_path, time_series_df
