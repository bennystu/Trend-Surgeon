import numpy as np
from ts_boilerplate.params import CROSS_VAL, DATA, TRAIN
from typing import Tuple

def generate_data_monotonic_increase() -> np.ndarray:
    """Creates a monotonicly increasing time serie dataset for test purposes
    - shape is (DATA['length'], DATA['n_covariates] + DATA['n_targets']),
    - values are all equals to their respective integer index!

    e.g:
    data = array(
      [[  0.,   0.,   0.,   0.,   0.],
       [  1.,   1.,   1.,   1.,   1.],
       ...,
       [998., 998., 998., 998., 998.],
       [999., 999., 999., 999., 999.]]
    )

    """

    indexes = np.arange(0, DATA['length'])
    data = np.zeros((DATA['length'], DATA['n_covariates'] + DATA['n_targets'])) \
        + np.expand_dims(indexes, axis=1)
    return data

def generate_data_zeros_and_ones() -> np.ndarray:
    """Create a dummy data made of zeros for covariates, and ones for the targets
    e.g:
    data = array(
      [[1.,1.,0.,0.,0.],
       [1.,1.,0.,0.,0.],
       ...,
       [1.,1.,0.,0.,0.],
       [1.,1.,0.,0.,0.]]
    )
    """
    shape = (DATA['length'], DATA['n_covariates'] + DATA['n_targets'])
    data = np.zeros(shape)
    data[:, DATA["target_column_idx"]] = 1.
    return data

def generate_X_y_zeros_and_ones() -> Tuple[np.ndarray]:
    """Create a dummy (X,y) tuple made of zeros for covariates, and ones for the targets, just to check if model fit well"""
    length = round(DATA["length"] / TRAIN['stride'])

    shape_X = (length, TRAIN['input_length'], DATA['n_covariates']+DATA['n_targets'])
    X = np.zeros(shape_X)
    X[:, :, DATA["target_column_idx"]] = 1.

    shape_y = (length, TRAIN['output_length'], DATA['n_targets'])
    y = np.ones(shape_y)
    y = np.squeeze(y)

    return (X,y)

def generate_time_series(n_samples, input_length, n_features, trend=True, noise_level=0.01):
    """
    Generates a dummy time series dataset with optional trend and noise
    Returns X: (n_samples, input_length, n_features)
            y: (n_samples, output_length)
    """
    import numpy as np
    output_length = 1
    X = np.zeros((n_samples, input_length, n_features))
    y = np.zeros((n_samples, output_length))

    for i in range(n_samples):
        for f in range(n_features):
            series = np.linspace(0, 1, input_length)
            if noise_level > 0:
                series += np.random.normal(0, noise_level, size=input_length)
            X[i,:,f] = series
        y[i,0] = X[i,-1,0] + np.random.normal(0, noise_level)

    return X, y
