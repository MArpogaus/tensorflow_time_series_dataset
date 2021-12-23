import pytest
import tensorflow as tf
import numpy as np

from tensorflow_time_series_dataset.pipeline.patch_generator import PatchGenerator
from tensorflow_time_series_dataset.pipeline.batch_processor import BatchPreprocessor
from tensorflow_time_series_dataset.preprocessors.groupby_dataset_generator import (
    GroupbyDatasetGenerator,
)


def get_value(l, c):
    return int(f"{c:02d}{l:04d}")


def sorted_ds():
    patch = [
        [gen_value(int(id), c, l) for c in patch_columns]
        for l in range(line, line + patch_size)
    ]
    return np.float32(patch)


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
    return ds, df


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


def get_id_and_idx(val):
    id = val // 1e6
    idx = val % 1e4
    return id, idx


def gen_patch(df, idx, size):
    return df.values[np.int(idx) : np.int(idx + size)]


def gen_batch(df, columns, size, ids, lines):
    batch = []
    print(columns)
    for id, line in zip(ids, lines):
        if "id" in df.columns:
            p = gen_patch(df[df.id == id][columns], line, size)
        else:
            p = gen_patch(df[columns], line, size)
        batch.append(p)
    return np.float32(batch)


def validate_batch(
    df,
    ds,
    history_size,
    history_columns,
    meta_columns,
    prediction_columns,
):
    for b, ((x1, x2), y) in enumerate(ds.as_numpy_iterator()):
        prediction_size = y.shape[1]

        first_val = x1[:, 0]
        ids, lines = get_id_and_idx(first_val)

        assert np.all(
            x1 == gen_batch(df, history_columns, history_size, ids, lines)
        ), f"Wrong data: history ({b})"
        assert np.all(
            x2 == gen_batch(df, meta_columns, 1, ids, lines + history_size)
        ), f"Wrong data: meta ({b})"

        print(x1.shape)
        y_test = x1[:, -prediction_size:] + prediction_size
        assert np.all(y == y_test), f"Wrong data: prediction ({b})"

        first_val = y[:, 0]
        ids, lines = get_id_and_idx(first_val)

        assert np.all(
            y == gen_batch(df, prediction_columns, prediction_size, ids, lines)
        ), f"Wrong data: prediction ({b})"


@pytest.mark.parametrize(
    "patched_dataset", [(2 * 48, 48), (48 + 1, 1)], indirect=["patched_dataset"]
)
def test_batch_processor(patched_dataset):
    ds, df = patched_dataset

    columns = list(sorted(df.columns))

    batch_kwds = dict(
        history_size=48,
        history_columns=columns[:1],
        meta_columns=columns[1:],
        prediction_columns=columns[:1],
    )
    print(batch_kwds)
    ds_batched = ds.batch(4, drop_remainder=True)
    ds_batched = ds_batched.map(
        BatchPreprocessor(**batch_kwds),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    validate_batch(df, ds_batched, **batch_kwds)
