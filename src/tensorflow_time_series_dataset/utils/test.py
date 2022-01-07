# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-07 09:02:38 (Marcel Arpogaus)
# changed : 2022-01-07 14:50:51 (Marcel Arpogaus)
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
            p = gen_patch(df[df.id == id][columns], line, size)
        else:
            p = gen_patch(df[columns], line, size)
        batch.append(p)
    return np.float32(batch)


def validate_dataset(
    df,
    ds,
    batch_size,
    history_size,
    prediction_size,
    history_columns,
    meta_columns,
    prediction_columns,
    history_reference_column="ref",
    meta_reference_column="ref",
    prediction_reference_column="ref",
):
    x1_shape = (batch_size, history_size, len(history_columns))
    x2_shape = (batch_size, 1, len(meta_columns))
    y_shape = (batch_size, prediction_size, len(prediction_columns))
    history_columns = list(sorted(history_columns))
    meta_columns = list(sorted(meta_columns))
    prediction_columns = list(sorted(prediction_columns))
    history_columns_idx = {c: i for i, c in enumerate(history_columns)}
    meta_columns_idx = {c: i for i, c in enumerate(meta_columns)}
    prediction_columns_idx = {c: i for i, c in enumerate(prediction_columns)}

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
            first_val = x1[:, 0, history_columns_idx[history_reference_column]]
            ids, lines = get_id_and_idx(first_val)
            assert np.all(
                x1 == gen_batch(df, history_columns, history_size, ids, lines)
            ), f"Wrong data: history ({b})"
            if x2 is not None:
                assert np.all(
                    x2 == gen_batch(df, meta_columns, 1, ids, lines + history_size)
                ), f"wrong data: meta not consecutive ({b})"

            last_val = x1[:, -1, history_columns_idx[history_reference_column]]
            ids, lines = get_id_and_idx(last_val)
            y_test = gen_batch(df, prediction_columns, prediction_size, ids, lines + 1)
            assert np.all(y == y_test), f"Wrong data: prediction not consecutive ({b})"

        if x2 is not None:
            first_val = x2[:, 0, meta_columns_idx[meta_reference_column]]
            ids, lines = get_id_and_idx(first_val)
            assert x2.shape == x2_shape, f"Wrong shape: meta ({b})"
            assert np.all(
                x2 == gen_batch(df, meta_columns, 1, ids, lines)
            ), f"Wrong data: meta ({b})"

        assert y.shape == y_shape, f"Wrong shape: prediction ({b})"
        first_val = y[:, 0, prediction_columns_idx[prediction_reference_column]]
        ids, lines = get_id_and_idx(first_val)
        assert np.all(
            y == gen_batch(df, prediction_columns, prediction_size, ids, lines)
        ), f"Wrong data: prediction ({b})"


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
            match="history_size must be a positive integer greater than zero, when no meta date is used",
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
