import tensorflow as tf


class GroupbyDatasetGenerator:
    def __init__(self, groupby, columns, dtype=tf.float32, test_mode=False):
        self.groupby = groupby
        self.columns = sorted(list(set(columns)))
        self.dtype = dtype
        self.test_mode = test_mode

    def get_generator(self, df):
        df.sort_index(inplace=True)
        if self.test_mode:
            ids = df[self.groupby].unique()
            ids = ids[:2]
            df = df[df[self.groupby].isin(ids)]

        grpd = df.groupby(self.groupby)

        def generator():
            for _, d in grpd:
                yield d[self.columns].values

        return generator

    def __call__(self, df):
        ds = tf.data.Dataset.from_generator(
            self.get_generator(df),
            output_signature=(
                tf.TensorSpec(shape=[None, len(self.columns)], dtype=self.dtype)
            ),
        )
        return ds
