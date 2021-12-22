import tensorflow as tf
import numpy as np


class CyclicalFeatureEncoder:
    def __init__(
        self, cycl_max, cycl_min=0, name="CyclicalFeatureEncoder", dtype=tf.float32
    ):
        self.cycl_max = cycl_max
        self.cycl_min = cycl_min
        self.name = name
        self.dtype = dtype

    def encode(self, cycl):
        with tf.name_scope(self.name + "::decode"):
            cycl = tf.convert_to_tensor(cycl, name="cycl", dtype=self.dtype)
            sin = tf.sin(
                2 * np.pi * (cycl - self.cycl_min) / (self.cycl_max - self.cycl_min + 1)
            )
            cos = tf.cos(
                2 * np.pi * (cycl - self.cycl_min) / (self.cycl_max - self.cycl_min + 1)
            )
            assert np.allclose(
                cycl, self.decode(sin, cos)
            ), f'Decoding failed. is "cycl_min/max" ({self.cycl_min}/{self.cycl_max}) correct?'
            return tf.stack([sin, cos], axis=0)

    def decode(self, sin, cos):
        with tf.name_scope(self.name + "::encode"):
            angle = (tf.math.atan2(sin, cos) + 2 * np.pi) % (2 * np.pi)
            return (angle * (self.cycl_max - self.cycl_min + 1)) / (
                2 * np.pi
            ) + self.cycl_min
