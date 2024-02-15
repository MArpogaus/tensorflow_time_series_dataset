# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2024-02-15 17:03:39 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# ...
# LICENSE #####################################################################
# Copyright 2022 Marcel Arpogaus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest


def get_idx(ref, val):
    idx = np.where(ref == val)[0]
    assert len(idx) == 1, "Could not determine index from refference value"
    return int(idx)


def gen_patch(df, idx, size):
    x = df.values[idx : idx + size]
    assert len(x) == size, "Could not generate patch from reference data"
    return x


def gen_batch(df, columns, size, refs, ref_col):
    batch = []
    columns = list(sorted(columns))
    for ref in refs:
        if "id" in df.columns:
            id = ref // 1e5
            idx = get_idx(df[df.id == id][ref_col], ref)
            p = gen_patch(df[df.id == id][columns], idx, size)
        else:
            idx = get_idx(df[ref_col], ref)
            p = gen_patch(df[columns], idx, size)
        batch.append(p)
    return np.float32(batch)


def validate_dataset(
    df,
    ds,
    batch_size,
    history_size,
    prediction_size,
    shift,
    history_columns,
    meta_columns,
    prediction_columns,
    drop_remainder,
    history_reference_column="ref",
    meta_reference_column="ref",
    prediction_reference_column="ref",
    **kwds,
):
    df = df.sort_index()

    if "id" in df.columns:
        ids = len(df.id.unique())
    else:
        ids = 1

    window_size = history_size + prediction_size
    initial_size = window_size - shift
    patch_data = df.index.unique().size - initial_size
    patches = patch_data / shift * ids
    has_remainder = (patches % batch_size != 0) and not drop_remainder
    expected_batches = int(patches // batch_size) + int(has_remainder)

    history_columns = list(sorted(history_columns))
    meta_columns = list(sorted(meta_columns))
    prediction_columns = list(sorted(prediction_columns))

    history_columns_idx = {c: i for i, c in enumerate(history_columns)}
    meta_columns_idx = {c: i for i, c in enumerate(meta_columns)}
    prediction_columns_idx = {c: i for i, c in enumerate(prediction_columns)}

    assert (
        len(set(history_columns + meta_columns + prediction_columns) - set(df.columns))
        == 0
    ), "Not all columns in test df"
    assert (
        history_reference_column in df.columns
    ), f"history_reference_column ({history_reference_column}) not in df.columns"
    assert (
        meta_reference_column in df.columns
    ), f"meta_reference_column ({meta_reference_column}) not in df.columns"
    assert (
        prediction_reference_column in df.columns
    ), f"prediction_reference_column ({prediction_reference_column}) not in df.columns"

    for batch_no, (x, y) in enumerate(ds.as_numpy_iterator(), start=1):
        if batch_no == expected_batches and has_remainder:
            bs = int(patches % batch_size)
        else:
            bs = batch_size
        x1_shape = (bs, history_size, len(history_columns))
        x2_shape = (bs, 1, len(meta_columns))
        y_shape = (bs, prediction_size, len(prediction_columns))

        x1, x2 = None, None
        if history_size and len(history_columns) and len(meta_columns):
            x1, x2 = x
        elif history_size and len(history_columns):
            x1 = x
        elif len(meta_columns):
            x2 = x

        if x1 is not None:
            assert (
                x1.shape == x1_shape
            ), f"Wrong shape: history ({batch_no}: {x1.shape} != {x1_shape})"
            ref = x1[:, 0, history_columns_idx[history_reference_column]]
            assert np.all(
                x1
                == gen_batch(
                    df, history_columns, history_size, ref, history_reference_column
                )
            ), f"Wrong data: history ({batch_no})"
            if x2 is not None:
                assert np.all(
                    x2
                    == gen_batch(
                        df,
                        meta_columns,
                        1,
                        ref + history_size,
                        history_reference_column,
                    )
                ), f"wrong data: meta not consecutive ({batch_no})"

            last_val = x1[:, -1, history_columns_idx[history_reference_column]]
            y_test = gen_batch(
                df,
                prediction_columns,
                prediction_size,
                last_val + 1,
                history_reference_column,
            )
            assert np.all(
                y == y_test
            ), f"Wrong data: prediction not consecutive ({batch_no})"

        if x2 is not None:
            first_val = x2[:, 0, meta_columns_idx[meta_reference_column]]
            assert x2.shape == x2_shape, f"Wrong shape: meta ({batch_no})"
            assert np.all(
                x2 == gen_batch(df, meta_columns, 1, first_val, meta_reference_column)
            ), f"Wrong data: meta ({batch_no})"

        assert y.shape == y_shape, f"Wrong shape: prediction ({batch_no})"
        first_val = y[:, 0, prediction_columns_idx[prediction_reference_column]]
        assert np.all(
            y
            == gen_batch(
                df,
                prediction_columns,
                prediction_size,
                first_val,
                prediction_reference_column,
            )
        ), f"Wrong data: prediction ({batch_no})"

    assert batch_no is not None, "No iteration. Is there enough data?"

    assert (
        batch_no == expected_batches
    ), f"Not enough batches. {batch_no} != {expected_batches}"

    return batch_no


def get_ctxmgr(
    prediction_size, history_columns, meta_columns, history_size, prediction_columns
):
    if prediction_size <= 0:
        ctxmgr = pytest.raises(
            AssertionError,
            match="prediction_size must be a positive integer greater than zero",
        )
    elif len(set(history_columns + meta_columns)) == 0:
        ctxmgr = pytest.raises(
            AssertionError,
            match="No feature columns provided",
        )
    elif len(meta_columns) == 0 and history_size <= 0:
        ctxmgr = pytest.raises(
            AssertionError,
            match="history_size must be a positive integer greater than zero,"
            " when no meta date is used",
        )
    elif history_size < 0:
        ctxmgr = pytest.raises(
            AssertionError,
            match="history_size must be a positive integer",
        )
    elif len(prediction_columns) == 0:
        ctxmgr = pytest.raises(
            AssertionError,
            match="No prediction columns provided",
        )
    else:
        ctxmgr = does_not_raise()
    return ctxmgr
