from tensorflow_time_series_dataset import WindowedTimeSeriesDatasetFactory
from tensorflow_time_series_dataset.loaders import CSVDataLoader
from tensorflow_time_series_dataset.preprocessors import (
    GroupbyDatasetGenerator,
    TimeSeriesSplit,
    CyclicalFeatureEncoder,
)
from tensorflow_time_series_dataset.utils.test import get_ctxmgr, validate_dataset


def test_windowed_time_series_dataset_factory(
    time_series_df,
    batch_size,
    history_size,
    prediction_size,
    shift,
    history_columns,
    meta_columns,
    prediction_columns,
):

    df = time_series_df.set_index("date_time")

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
        cycle_length=1,
        shuffle_buffer_size=100,
        seed=1,
    )

    with get_ctxmgr(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
    ):
        factory = WindowedTimeSeriesDatasetFactory(**common_kwds, **factory_kwds)
        ds = factory(df)
        ds
        validate_dataset(
            df,
            ds,
            **common_kwds,
        )


def test_windowed_time_series_dataset_factory_groupby(
    time_series_df_with_id,
    batch_size,
    history_size,
    prediction_size,
    shift,
    history_columns,
    meta_columns,
    prediction_columns,
):

    df = time_series_df_with_id.set_index("date_time")

    ids = df.id.unique()
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

    with get_ctxmgr(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
    ):

        factory = WindowedTimeSeriesDatasetFactory(**common_kwds, **factory_kwds)
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
    tmp_csv,
    batch_size,
    history_size,
    prediction_size,
    shift,
    history_columns,
    meta_columns,
    prediction_columns,
):

    test_data_path, df = tmp_csv
    df = df.set_index("date_time")

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
        cycle_length=1,
        shuffle_buffer_size=100,
        seed=1,
    )
    with get_ctxmgr(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
    ):
        factory = WindowedTimeSeriesDatasetFactory(**common_kwds, **factory_kwds)
        factory.set_data_loader(CSVDataLoader(file_path=test_data_path))
        ds = factory()
        ds
        validate_dataset(
            df,
            ds,
            **common_kwds,
        )


def test_windowed_time_series_dataset_factory_csv_loader_with_preprocessors(
    tmp_csv_with_id,
    batch_size,
    history_size,
    prediction_size,
    shift,
    history_columns,
    meta_columns_cycle,
    prediction_columns,
):
    # define encoder args
    # [name: kwds]
    encs = {
        "weekday": dict(cycl_max=6),
        "dayofyear": dict(cycl_max=366, cycl_min=1),
        "month": dict(cycl_max=12, cycl_min=1),
        "time": dict(
            cycl_max=24 * 60 - 1,
            cycl_getter=lambda df, k: df.index.hour * 60 + df.index.minute,
        ),
    }
    test_data_path, df = tmp_csv_with_id
    test_df = df.set_index("date_time")
    for name, kwds in encs.items():
        enc = CyclicalFeatureEncoder(name, **kwds)
        test_df = enc(test_df)
    splitter = TimeSeriesSplit(0.5, TimeSeriesSplit.LEFT)
    test_df = splitter(test_df)

    ids = df.id.unique()

    common_kwds = dict(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns_cycle,
        prediction_columns=prediction_columns,
        batch_size=batch_size,
    )
    factory_kwds = dict(
        shift=shift,
        cycle_length=len(ids),
        shuffle_buffer_size=100,
        seed=1,
    )
    with get_ctxmgr(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns_cycle,
        prediction_columns=prediction_columns,
    ):
        factory = WindowedTimeSeriesDatasetFactory(**common_kwds, **factory_kwds)
        factory.set_data_loader(CSVDataLoader(file_path=test_data_path))
        factory.add_preprocessor(splitter)
        for name, kwds in encs.items():
            factory.add_preprocessor(CyclicalFeatureEncoder(name, **kwds))
        factory.add_preprocessor(
            GroupbyDatasetGenerator(
                "id", columns=history_columns + meta_columns_cycle + prediction_columns
            )
        )
        ds = factory()
        ds
        validate_dataset(
            test_df,
            ds,
            **common_kwds,
        )
