# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf

from tensorflow_time_series_dataset.pipeline.batch_processor import BatchPreprocessor
from tensorflow_time_series_dataset.pipeline.patch_generator import PatchGenerator


class WindowedTimeSeriesPipeline:
    def __init__(
        self,
        history_size,
        prediction_size,
        history_columns,
        meta_columns,
        prediction_columns,
        shift,
        batch_size,
        cycle_length,
        shuffle_buffer_size,
        seed,
    ):
        assert (
            prediction_size > 0
        ), "prediction_size must be a positive integer greater than zero"
        self.history_size = history_size
        self.prediction_size = prediction_size
        self.window_size = history_size + prediction_size
        self.history_columns = history_columns
        self.meta_columns = meta_columns
        self.prediction_columns = prediction_columns
        self.shift = shift
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed

    def __call__(self, ds):

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(
                self.cycle_length * self.shuffle_buffer_size, seed=self.seed
            )

        ds = ds.interleave(
            PatchGenerator(self.window_size, self.shift),
            cycle_length=self.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(self.batch_size * self.shuffle_buffer_size, seed=self.seed)

        ds = ds.batch(self.batch_size, drop_remainder=True)

        ds = ds.map(
            BatchPreprocessor(
                self.history_size,
                self.history_columns,
                self.meta_columns,
                self.prediction_columns,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds
