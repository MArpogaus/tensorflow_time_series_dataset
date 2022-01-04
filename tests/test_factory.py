from tensorflow_time_series_dataset import WindowedTimeSeriesDataSetFactory
from tensorflow_time_series_dataset.loaders import CSVDataLoader
from tensorflow_time_series_dataset.preprocessors import (
    GroupbyDatasetGenerator,
    TimeSeriesSplit,
)
from tensorflow_time_series_dataset.utils.test import get_ctxmgr, validate_dataset


def test_windowed_time_series_dataset_factory(
    time_series_df, batch_size, history_size, prediction_size, shift
):

    df = time_series_df.set_index("date_time")
    columns = list(sorted(df.columns))

    common_kwds = dict(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=columns[:1],
        meta_columns=columns[1:],
        prediction_columns=columns[:1],
        batch_size=batch_size,
    )
    factory_kwds = dict(
        shift=shift,
        cycle_length=1,
        shuffle_buffer_size=100,
        seed=1,
    )
    with get_ctxmgr(prediction_size):
        factory = WindowedTimeSeriesDataSetFactory(**common_kwds, **factory_kwds)
        ds = factory(df)
        ds
        validate_dataset(
            df,
            ds,
            **common_kwds,
        )


def test_windowed_time_series_dataset_factory_groupby(
    time_series_df_with_id, batch_size, history_size, prediction_size, shift
):

    df = time_series_df_with_id.set_index("date_time")

    ids = df.id.unique()
    columns = sorted([c for c in df.columns if c != "id"])

    history_columns = columns[:1]
    meta_columns = columns[1:]
    prediction_columns = columns[:1]
    common_kwds = dict(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
        batch_size=batch_size,
    )
    factory_kwds = dict(
        shift=shift,
        cycle_length=len(ids),
        shuffle_buffer_size=100,
        seed=1,
    )
    with get_ctxmgr(prediction_size):
        factory = WindowedTimeSeriesDataSetFactory(**common_kwds, **factory_kwds)
        factory.add_preprocessor(
            GroupbyDatasetGenerator(
                "id", columns=history_columns + meta_columns + prediction_columns
            )
        )

        ds = factory(df)
        ds
        validate_dataset(
            df,
            ds,
            **common_kwds,
        )


def test_windowed_time_series_dataset_factory_csv_loader(
    tmp_csv, batch_size, history_size, prediction_size, shift
):

    test_data_path, df = tmp_csv
    df = df.set_index("date_time")
    columns = list(sorted(df.columns))

    common_kwds = dict(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=columns[:1],
        meta_columns=columns[1:],
        prediction_columns=columns[:1],
        batch_size=batch_size,
    )
    factory_kwds = dict(
        shift=shift,
        cycle_length=1,
        shuffle_buffer_size=100,
        seed=1,
    )
    with get_ctxmgr(prediction_size):
        factory = WindowedTimeSeriesDataSetFactory(**common_kwds, **factory_kwds)
        factory.set_data_loader(CSVDataLoader(file_path=test_data_path))
        ds = factory()
        ds
        validate_dataset(
            df,
            ds,
            **common_kwds,
        )


def test_windowed_time_series_dataset_factory_csv_loader_with_preprocessors(
    tmp_csv_with_id, batch_size, history_size, prediction_size, shift
):

    test_data_path, df = tmp_csv_with_id
    df = df.set_index("date_time")

    splitter = TimeSeriesSplit(0.5, TimeSeriesSplit.LEFT)
    df = splitter(df)

    ids = df.id.unique()
    columns = sorted([c for c in df.columns if c != "id"])

    history_columns = columns[:1]
    meta_columns = columns[1:]
    prediction_columns = columns[:1]
    common_kwds = dict(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
        batch_size=batch_size,
    )
    factory_kwds = dict(
        shift=shift,
        cycle_length=len(ids),
        shuffle_buffer_size=100,
        seed=1,
    )
    with get_ctxmgr(prediction_size):
        factory = WindowedTimeSeriesDataSetFactory(**common_kwds, **factory_kwds)
        factory.set_data_loader(CSVDataLoader(file_path=test_data_path))
        factory.add_preprocessor(splitter)
        factory.add_preprocessor(
            GroupbyDatasetGenerator(
                "id", columns=history_columns + meta_columns + prediction_columns
            )
        )
        ds = factory()
        ds
        validate_dataset(
            df,
            ds,
            **common_kwds,
        )
