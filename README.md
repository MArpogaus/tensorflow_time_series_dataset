[![img](https://img.shields.io/github/contributors/MArpogaus/tensorflow-timeseries-dataset.svg?style=flat-square)](https://github.com/MArpogaus/tensorflow-timeseries-dataset/graphs/contributors)
[![img](https://img.shields.io/github/forks/MArpogaus/tensorflow-timeseries-dataset.svg?style=flat-square)](https://github.com/MArpogaus/tensorflow-timeseries-dataset/network/members)
[![img](https://img.shields.io/github/stars/MArpogaus/tensorflow-timeseries-dataset.svg?style=flat-square)](https://github.com/MArpogaus/tensorflow-timeseries-dataset/stargazers)
[![img](https://img.shields.io/github/issues/MArpogaus/tensorflow-timeseries-dataset.svg?style=flat-square)](https://github.com/MArpogaus/tensorflow-timeseries-dataset/issues)
[![img](https://img.shields.io/github/license/MArpogaus/tensorflow-timeseries-dataset.svg?style=flat-square)](https://github.com/MArpogaus/tensorflow-timeseries-dataset/blob/main/LICENSE)
[![img](https://img.shields.io/github/actions/workflow/status/MArpogaus/tensorflow-timeseries-dataset/test.yaml.svg?label=test&style=flat-square)](https://github.com/MArpogaus/tensorflow-timeseries-dataset/actions/workflows/test.yaml)
[![img](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg?logo=pre-commit&style=flat-square)](https://github.com/MArpogaus/tensorflow-timeseries-dataset/blob/main/.pre-commit-config.yaml)
[![img](https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555)](https://linkedin.com/in/MArpogaus)


# TensorFlow time-series Dataset

1.  [About The Project](#about-the-project)
2.  [Usage](#org0301ec0)
    1.  [Example Data](#org1397d97)
    2.  [Single-Step Prediction](#org8e15547)
    3.  [Multi-Step Prediction](#org3dd7dc8)
    4.  [Preprocessing: Add Metadata features](#org7b66c9a)
3.  [Contributing](#org141ef05)
4.  [License](#orgd3ceeaa)
5.  [Contact](#orgd84778f)
6.  [Acknowledgments](#org8e17169)


<a id="about-the-project"></a>

## About The Project

This python package should help you to create TensorFlow datasets for time-series data.


<a id="org0301ec0"></a>

## Usage


<a id="org1397d97"></a>

### Example Data

Suppose you have a dataset in the following form:

    import numpy as np
    import pandas as pd

    # make things determeinisteic
    np.random.seed(1)

    columns=['x1', 'x2', 'x3']
    periods=48 * 14
    test_df=pd.DataFrame(
        index=pd.date_range(
            start='1/1/1992',
            periods=periods,
            freq='30T'
        ),
        data=np.stack(
            [
                np.random.normal(0,0.5,periods),
                np.random.normal(1,0.5,periods),
                np.random.normal(2,0.5,periods)
            ],
            axis=1
        ),
        columns=columns
    )
    test_df.head()

                               x1        x2        x3
    1992-01-01 00:00:00  0.812173  1.205133  1.578044
    1992-01-01 00:30:00 -0.305878  1.429935  1.413295
    1992-01-01 01:00:00 -0.264086  0.550658  1.602187
    1992-01-01 01:30:00 -0.536484  1.159828  1.644974
    1992-01-01 02:00:00  0.432704  1.159077  2.005718


<a id="org8e15547"></a>

### Single-Step Prediction

The factory class `WindowedTimeSeriesDatasetFactory` is used to create a TensorFlow dataset from pandas dataframes, or other data sources as we will see later.
We will use it now to create a dataset with `48` historic time-steps as the input to predict a single time-step in the future.

    from tensorflow_time_series_dataset.factory import WindowedTimeSeriesDatasetFactory as Factory

    factory_kwds=dict(
        history_size=48,
        prediction_size=1,
        history_columns=['x1', 'x2', 'x3'],
        prediction_columns=['x3'],
        batch_size=4,
        drop_remainder=True,
    )
    factory=Factory(**factory_kwds)
    ds1=factory(test_df)
    ds1

This returns the following TensorFlow Dataset:

    <_PrefetchDataset element_spec=(TensorSpec(shape=(4, 48, 3), dtype=tf.float32, name=None), TensorSpec(shape=(4, 1, 1), dtype=tf.float32, name=None))>

We can plot the result with the utility function `plot_path`:

    from tensorflow_time_series_dataset.utils.visualisation import plot_patch
    fig=plot_patch(
        ds1,
        figsize=(8,4),
        **factory_kwds
    )

    fname='.images/example1.png'
    fig.savefig(fname)
    fname

![img](.images/example1.png)


<a id="org3dd7dc8"></a>

### Multi-Step Prediction

Lets now increase the prediction size to `6` half-hour time-steps.

    factory_kwds.update(dict(
        prediction_size=6
    ))
    factory=Factory(**factory_kwds)
    ds2=factory(test_df)
    ds2

This returns the following TensorFlow Dataset:

    <_PrefetchDataset element_spec=(TensorSpec(shape=(4, 48, 3), dtype=tf.float32, name=None), TensorSpec(shape=(4, 6, 1), dtype=tf.float32, name=None))>

Again, lets plot the results to see what changed:

    fig=plot_patch(
        ds2,
        figsize=(8,4),
        **factory_kwds
    )

    fname='.images/example2.png'
    fig.savefig(fname)
    fname

![img](.images/example2.png)


<a id="org7b66c9a"></a>

### Preprocessing: Add Metadata features

Preprocessors can be used to transform the data before it is fed into the model.
A Preprocessor can be any python callable.
In this case we will be using the a class called `CyclicalFeatureEncoder` to encode our one-dimensional cyclical features like the *time* or *weekday* to two-dimensional coordinates using a sine and cosine transformation as suggested in [this blogpost](<https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning>).

    import itertools
    from tensorflow_time_series_dataset.preprocessors import CyclicalFeatureEncoder
    encs = {
        "weekday": dict(cycl_max=6),
        "dayofyear": dict(cycl_max=366, cycl_min=1),
        "month": dict(cycl_max=12, cycl_min=1),
        "time": dict(
            cycl_max=24 * 60 - 1,
            cycl_getter=lambda df, k: df.index.hour * 60 + df.index.minute,
        ),
    }
    factory_kwds.update(dict(
        meta_columns=list(itertools.chain(*[[c+'_sin', c+'_cos'] for c in encs.keys()]))
    ))
    factory=Factory(**factory_kwds)
    for name, kwds in encs.items():
        factory.add_preprocessor(CyclicalFeatureEncoder(name, **kwds))
    ds3=factory(test_df)
    ds3

This returns the following TensorFlow Dataset:

    <_PrefetchDataset element_spec=((TensorSpec(shape=(4, 48, 3), dtype=tf.float32, name=None), TensorSpec(shape=(4, 1, 8), dtype=tf.float32, name=None)), TensorSpec(shape=(4, 6, 1), dtype=tf.float32, name=None))>

Again, lets plot the results to see what changed:

    fig=plot_patch(
        ds3,
        figsize=(8,4),
        **factory_kwds
    )

    fname='.images/example3.png'
    fig.savefig(fname)
    fname

![img](.images/example3.png)


<a id="org141ef05"></a>

## Contributing

Any Contributions are greatly appreciated! If you have a question, an issue or would like to contribute, please read our [contributing guidelines](CONTRIBUTING.md).


<a id="orgd3ceeaa"></a>

## License

Distributed under the [Apache License 2.0](LICENSE)


<a id="orgd84778f"></a>

## Contact

[Marcel Arpogaus](https://github.com/marpogaus) - [marcel.arpogaus@gmail.com](mailto:marcel.arpogaus@gmail.com)

Project Link:
<https://github.com/MArpogaus/tensorflow-timeseries-dataset>


<a id="org8e17169"></a>

## Acknowledgments

Parts of this work have been funded by the Federal Ministry for the Environment, Nature Conservation and Nuclear Safety due to a decision of the German Federal Parliament (AI4Grids: 67KI2012A).
