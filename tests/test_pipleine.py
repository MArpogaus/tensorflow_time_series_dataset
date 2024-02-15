import numpy as np
import pytest
import tensorflow as tf
from tensorflow_time_series_dataset.pipeline.patch_generator import PatchGenerator
from tensorflow_time_series_dataset.pipeline.patch_processor import PatchPreprocessor
from tensorflow_time_series_dataset.pipeline.windowed_time_series_pipeline import (
    WindowedTimeSeriesPipeline,
)
from tensorflow_time_series_dataset.preprocessors.groupby_dataset_generator import (
    GroupbyDatasetGenerator,
)
from tensorflow_time_series_dataset.utils.test import get_ctxmgr, validate_dataset

# FIXTURES ####################################################################


@pytest.fixture
def groupby_dataset(
    time_series_df_with_id, history_columns, meta_columns, prediction_columns
):
    time_series_df_with_id.set_index("date_time", inplace=True)
    used_cols = set(history_columns + meta_columns + prediction_columns)
    gen = GroupbyDatasetGenerator("id", columns=used_cols)
    return gen(time_series_df_with_id), time_series_df_with_id[list(used_cols) + ["id"]]


@pytest.fixture
def patched_dataset(
    request, time_series_df, history_columns, meta_columns, prediction_columns
):
    window_size, shift = request.param
    df = time_series_df.set_index("date_time")
    used_cols = set(history_columns + meta_columns + prediction_columns)

    ds = tf.data.Dataset.from_tensors(df[sorted(used_cols)])
    ds = ds.interleave(
        PatchGenerator(window_size=window_size, shift=shift),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds, df, window_size, shift


# TESTS #######################################################################
@pytest.mark.parametrize("window_size,shift", [(2 * 48, 48), (48 + 1, 1)])
def test_patch_generator(time_series_df, window_size, shift):
    df = time_series_df.set_index("date_time")

    initial_size = window_size - shift
    data_size = df.index.size - initial_size
    patches = data_size // shift

    expected_shape = (window_size, len(df.columns))

    ds = tf.data.Dataset.from_tensors(df)
    ds_patched = ds.interleave(
        PatchGenerator(window_size=window_size, shift=shift),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    for i, patch in enumerate(ds_patched.as_numpy_iterator()):
        assert patch.shape == expected_shape, "Wrong shape"
        x1 = patch[0, 0]
        idx = int(x1 % 1e5)
        expected_values = df.iloc[idx : idx + window_size]
        assert np.all(patch == expected_values), "Patch contains wrong data"
    assert i + 1 == patches, "Not enough patches"


@pytest.mark.parametrize("window_size,shift", [(2 * 48, 48), (48 + 1, 1)])
def test_patch_generator_groupby(groupby_dataset, window_size, shift):
    ds, df = groupby_dataset
    records_per_id = df.groupby("id").size().max()
    ids = df.id.unique()
    columns = sorted([c for c in df.columns if c != "id"])

    initial_size = window_size - shift
    data_size = records_per_id - initial_size
    patches = data_size / shift * len(ids)
    expected_shape = (window_size, len(columns))

    ds_patched = ds.interleave(
        PatchGenerator(window_size=window_size, shift=shift),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    for i, patch in enumerate(ds_patched.as_numpy_iterator()):
        assert patch.shape == expected_shape, "Wrong shape"
        if len(columns):
            x1 = patch[0, 0]
            id = int(x1 // 1e5)
            idx = int(x1 % 1e5)
            expected_values = df[df.id == id].iloc[idx : idx + window_size]
            assert np.all(
                patch == expected_values[columns].values
            ), "Patch contains wrong data"
    assert i + 1 == patches, "Not enough patches"


@pytest.mark.parametrize(
    "patched_dataset", [(2 * 48, 48), (48 + 1, 1)], indirect=["patched_dataset"]
)
def test_batch_processor(
    patched_dataset,
    history_size,
    batch_size,
    history_columns,
    meta_columns,
    prediction_columns,
):
    ds, df, window_size, shift = patched_dataset
    prediction_size = window_size - history_size

    batch_kwds = dict(
        history_size=history_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
    )

    with get_ctxmgr(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
    ):
        ds_batched = ds.map(
            PatchPreprocessor(**batch_kwds),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        ).batch(batch_size)

        validate_dataset(
            df,
            ds_batched,
            batch_size=batch_size,
            prediction_size=prediction_size,
            shift=shift,
            drop_remainder=False,
            **batch_kwds,
        )


def test_windowed_time_series_pipeline(
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
    used_cols = sorted(
        set(
            history_columns + meta_columns + prediction_columns,
        )
    )

    batch_kwds = dict(
        history_size=history_size,
        prediction_size=prediction_size,
        shift=shift,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
        batch_size=batch_size,
        drop_remainder=True,
    )
    pipeline_kwds = dict(cycle_length=1, shuffle_buffer_size=100, cache=True)

    with get_ctxmgr(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
    ):
        pipeline = WindowedTimeSeriesPipeline(**batch_kwds, **pipeline_kwds)
        ds = tf.data.Dataset.from_tensors(df[used_cols])
        ds = pipeline(ds)
        validate_dataset(
            df,
            ds,
            **batch_kwds,
        )


def test_windowed_time_series_pipeline_groupby(
    groupby_dataset,
    batch_size,
    history_size,
    prediction_size,
    shift,
    history_columns,
    meta_columns,
    prediction_columns,
):
    ds, df = groupby_dataset

    ids = df.id.unique()

    batch_kwds = dict(
        history_size=history_size,
        prediction_size=prediction_size,
        shift=shift,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
        batch_size=batch_size,
        drop_remainder=False,
    )
    pipeline_kwds = dict(cycle_length=len(ids), shuffle_buffer_size=1000, cache=True)

    with get_ctxmgr(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
    ):
        pipeline = WindowedTimeSeriesPipeline(**batch_kwds, **pipeline_kwds)
        ds = pipeline(ds)
        validate_dataset(
            df,
            ds,
            **batch_kwds,
        )
