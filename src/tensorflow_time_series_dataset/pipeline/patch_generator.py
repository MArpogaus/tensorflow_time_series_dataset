import tensorflow as tf


class PatchGenerator:
    def __init__(self, window_size, shift):
        self.window_size = window_size
        self.shift = shift

    def __call__(self, data):
        def sub_to_patch(sub):
            return sub.batch(self.window_size, drop_remainder=True)

        data_set = tf.data.Dataset.from_tensor_slices(data)

        data_set = data_set.window(
            size=self.window_size,
            shift=self.shift,
            drop_remainder=True,
        )
        data_set = data_set.flat_map(sub_to_patch)

        return data_set
