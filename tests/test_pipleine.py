import pytest
import tensorflow as tf
import numpy as np

from tensorflow_time_series_dataset.pipeline.patch_generator import PatchGenerator
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


@pytest.fixture
def groupby_dataset(time_series_df_with_id):
    time_series_df_with_id.set_index("date_time", inplace=True)
    columns = list(sorted([c for c in time_series_df_with_id.columns if c != "id"]))
    gen = GroupbyDatasetGenerator("id", columns=columns)
    return gen(time_series_df_with_id), time_series_df_with_id


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
