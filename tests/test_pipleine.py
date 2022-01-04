import numpy as np
import pytest
import tensorflow as tf
from tensorflow_time_series_dataset.pipeline.batch_processor import BatchPreprocessor
from tensorflow_time_series_dataset.pipeline.patch_generator import PatchGenerator
from tensorflow_time_series_dataset.pipeline.windowed_time_series_pipeline import (
    WindowedTimeSeriesPipeline,
)
from tensorflow_time_series_dataset.preprocessors.groupby_dataset_generator import (
    GroupbyDatasetGenerator,
)


def get_id_and_idx(val):
    id = val // 1e6
    idx = val % 1e4
    return id, idx


def gen_patch(df, idx, size):
    return df.values[np.int(idx) : np.int(idx + size)]


def gen_batch(df, columns, size, ids, lines):
    batch = []
    for id, line in zip(ids, lines):
        if "id" in df.columns:
            p = gen_patch(df[df.id == id[0]][columns], line, size)
        else:
            p = gen_patch(df[columns], line, size)
        batch.append(p)
    return np.float32(batch)


def validate_batch(
    df,
    ds,
    batch_size,
    history_size,
    prediction_size,
    history_columns,
    meta_columns,
    prediction_columns,
):
    x1_shape = (batch_size, history_size, len(history_columns))
    x2_shape = (batch_size, 1, len(meta_columns))
    y_shape = (batch_size, prediction_size, len(meta_columns))
    for b, (x, y) in enumerate(ds.as_numpy_iterator()):
        x1, x2 = None, None
        if history_size and len(history_columns) and len(meta_columns):
            x1, x2 = x
        elif history_size and len(history_columns):
            x1 = x
        elif len(meta_columns):
            x2 = x

        if x1 is not None:
            assert x1.shape == x1_shape, f"Wrong shape: history ({b})"
            first_val = x1[:, 0]
            ids, lines = get_id_and_idx(first_val)

            assert np.all(
                x1 == gen_batch(df, history_columns, history_size, ids, lines)
            ), f"Wrong data: history ({b})"
            if x2 is not None:
                assert np.all(
                    x2 == gen_batch(df, meta_columns, 1, ids, lines + history_size)
                ), f"Wrong data: meta ({b})"

            if history_columns == prediction_columns:
                y_test = x1[:, -prediction_size:] + prediction_size
                assert np.all(y == y_test), f"Wrong data: prediction ({b})"

        if x2 is not None:
            first_val = x2[:, 0]
            ids, lines = get_id_and_idx(first_val)
            assert x2.shape == x2_shape, f"Wrong shape: meta ({b})"
            assert np.all(
                x2 == gen_batch(df, meta_columns, 1, ids, lines)
            ), f"Wrong data: meta ({b})"

        assert y.shape == y_shape, f"Wrong shape: prediction ({b})"
        first_val = y[:, 0]
        ids, lines = get_id_and_idx(first_val)
        assert np.all(
            y == gen_batch(df, prediction_columns, prediction_size, ids, lines)
        ), f"Wrong data: prediction ({b})"


# FIXTURES ####################################################################
@pytest.fixture
def groupby_dataset(time_series_df_with_id):
    time_series_df_with_id.set_index("date_time", inplace=True)
    columns = list(sorted([c for c in time_series_df_with_id.columns if c != "id"]))
    gen = GroupbyDatasetGenerator("id", columns=columns)
    return gen(time_series_df_with_id), time_series_df_with_id


@pytest.fixture
def patched_dataset(request, time_series_df):
    window_size, shift = request.param
    df = time_series_df.set_index("date_time")

    ds = tf.data.Dataset.from_tensors(df)
    ds = ds.interleave(
        PatchGenerator(window_size=window_size, shift=shift),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    return ds, df, window_size, shift


@pytest.fixture(params=list(range(0, 48 * 2, 48)))
def history_size(request):
    return request.param


@pytest.fixture(params=[4, 32])
def batch_size(request):
    return request.param


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
        idx = int(x1 % 1e4)
        expected_values = df.iloc[idx : idx + window_size]
        assert np.all(patch == expected_values), "Patch contains wrong data"
    assert i + 1 == patches, "Not enough patches"


@pytest.mark.parametrize("window_size,shift", [(2 * 48, 48), (48 + 1, 1)])
def test_patch_generator_groupby(groupby_dataset, window_size, shift):
    ds, df = groupby_dataset
    records_per_id = df.groupby("id").x1.size().max()
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
        x1 = patch[0, 0]
        id = int(x1 // 1e6)
        idx = int(x1 % 1e4)
        expected_values = df[df.id == id].iloc[idx : idx + window_size]
        assert np.all(
            patch == expected_values[columns].values
        ), "Patch contains wrong data"
    assert i + 1 == patches, "Not enough patches"


@pytest.mark.parametrize(
    "patched_dataset", [(2 * 48, 48), (48 + 1, 1)], indirect=["patched_dataset"]
)
def test_batch_processor(patched_dataset, history_size, batch_size):
    ds, df, window_size, shift = patched_dataset
    prediction_size = window_size - history_size

    columns = list(sorted(df.columns))

    batch_kwds = dict(
        history_size=history_size,
        history_columns=columns[:1],
        meta_columns=columns[1:],
        prediction_columns=columns[:1],
    )

    ds_batched = ds.batch(batch_size, drop_remainder=True)
    ds_batched = ds_batched.map(
        BatchPreprocessor(**batch_kwds),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    validate_batch(
        df,
        ds_batched,
        batch_size=batch_size,
        prediction_size=prediction_size,
        **batch_kwds,
    )


@pytest.mark.parametrize("prediction_size,shift", [(1, 1), (48, 1), (48, 48)])
def test_windowed_time_series_pipeline(
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
    pipeline_kwds = dict(
        shift=shift,
        cycle_length=1,
        shuffle_buffer_size=100,
        seed=1,
    )
    pipeline = WindowedTimeSeriesPipeline(**batch_kwds, **pipeline_kwds)
    ds = tf.data.Dataset.from_tensors(df)
    ds = pipeline(ds)
    ds
    validate_batch(
        df,
        ds,
        **batch_kwds,
    )


@pytest.mark.parametrize("prediction_size,shift", [(1, 1), (48, 1), (48, 48)])
def test_windowed_time_series_pipeline_groupby(
    groupby_dataset, batch_size, history_size, prediction_size, shift
):
    ds, df = groupby_dataset

    ids = df.id.unique()
    columns = sorted([c for c in df.columns if c != "id"])

    batch_kwds = dict(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=columns[:1],
        meta_columns=columns[1:],
        prediction_columns=columns[:1],
        batch_size=batch_size,
    )
    pipeline_kwds = dict(
        shift=shift,
        cycle_length=len(ids),
        shuffle_buffer_size=1000,
        seed=1,
    )
    pipeline = WindowedTimeSeriesPipeline(**batch_kwds, **pipeline_kwds)
    ds = pipeline(ds)
    ds
    validate_batch(
        df,
        ds,
        **batch_kwds,
    )
