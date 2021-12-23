import tensorflow as tf


class BatchPreprocessor:
    def __init__(
        self,
        history_size,
        history_columns,
        meta_columns,
        prediction_columns,
    ):
        self.history_size = history_size
        self.history_columns = history_columns
        self.meta_columns = meta_columns
        self.prediction_columns = prediction_columns

        columns = sorted(list(set(history_columns + prediction_columns + meta_columns)))
        self.column_idx = {c: i for i, c in enumerate(columns)}

    def __call__(self, batch):
        y = []
        x_hist = []
        x_meta = []

        x_columns = sorted(set(self.history_columns + self.meta_columns))
        y_columns = sorted(self.prediction_columns)

        for c in y_columns:
            column = batch[:, self.history_size :, self.column_idx[c]]
            y.append(column)

        if len(x_columns) == 0:
            ValueError("No feature columns provided")

        for c in x_columns:
            column = batch[:, :, self.column_idx[c], None]
            if c in self.history_columns:
                x_hist.append(column[:, : self.history_size, 0])
            if c in self.meta_columns:
                x_meta.append(column[:, self.history_size, None, ...])

        y = tf.stack(y, axis=2)
        x_hist = tf.stack(x_hist, axis=2)
        x_meta = tf.concat(x_meta, axis=2)

        return (x_hist, x_meta), y
