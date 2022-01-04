from tensorflow_time_series_dataset import WindowedTimeSeriesDataSetFactory
from tensorflow_time_series_dataset.utils.test import get_ctxmgr, validate_dataset


def test_windowed_time_series_dataset_factory(
    time_series_df, batch_size, history_size, prediction_size, shift
):

    df = time_series_df.set_index("date_time")
    columns = list(sorted(df.columns))

    batch_kwds = dict(
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
        factory = WindowedTimeSeriesDataSetFactory(**batch_kwds, **factory_kwds)
        ds = factory(df)
        ds
        validate_dataset(
            df,
            ds,
            **batch_kwds,
        )
